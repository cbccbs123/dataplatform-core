"""골든 관계셋 데이터 모델 + 순수 파싱·검증 (spec 031 T001).

골든 픽스처는 사람이 검증한 정답 관계셋이다.
- `pairs`: 관계가 있어야 하는 자산 쌍(a,b)과 정답 `kind`.
- `isolated`: 어떤 관계도 없어야 하는 자산(고립 처리 정확도 측정용).

자산은 **fs_path 또는 content_hash 키**로 지정한다(재적재에도 안정 — FR-001).
실제 `asset_id`로의 해소(`resolve_asset_keys`)는 DB가 필요하므로 러너/사람 몫(T006).
이 모듈의 `parse_golden`은 LLM/DB 불요 순수 함수다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from psycopg import Connection

# 허용 키 타입 — fs_path(파일 경로) 또는 content_hash(파일 해시).
_KEY_TYPES = ("fs_path", "content_hash")
# key_type → asset 컬럼. 화이트리스트라 SQL 식별자 보간 안전(주입 불가).
_KEY_COLUMN = {"fs_path": "fs_path", "content_hash": "file_hash"}


@dataclass(frozen=True)
class GoldenPair:
    """정답 관계 쌍 — 자산 키 a·b 와 관계 kind(+선택 메모)."""
    a: str
    b: str
    kind: str
    note: str = ""


@dataclass(frozen=True)
class Golden:
    """파싱·검증된 골든 관계셋(불변)."""
    key_type: str
    pairs: tuple[GoldenPair, ...]
    isolated: tuple[str, ...]


def parse_golden(data: dict) -> Golden:
    """골든 dict를 검증해 `Golden`으로 변환한다.

    결함(버전 불일치·미지원 key_type·필드 누락·자기-쌍)은 `ValueError`.
    """
    if data.get("version") != 1:
        raise ValueError(f"golden version must be 1: {data.get('version')!r}")
    kt = data.get("key_type")
    if kt not in _KEY_TYPES:
        raise ValueError(f"key_type must be one of {_KEY_TYPES}: {kt!r}")
    pairs: list[GoldenPair] = []
    for p in data.get("pairs", []):
        a, b, kind = p.get("a"), p.get("b"), p.get("kind")
        if not a or not b or not kind:
            raise ValueError(f"pair needs a/b/kind: {p!r}")
        if a == b:
            raise ValueError(f"self-pair not allowed: {p!r}")
        pairs.append(GoldenPair(str(a), str(b), str(kind), str(p.get("note") or "")))
    isolated = tuple(str(x) for x in data.get("isolated", []))
    return Golden(kt, tuple(pairs), isolated)


def resolve_asset_keys(
    conn: Connection[Any], golden: Golden
) -> tuple[dict[str, str], list[str]]:
    """골든 키(fs_path|content_hash)를 현재 ``asset_id``로 해소한다 (T006·DB·읽기 전용).

    ``key_type=fs_path`` → ``asset.fs_path``, ``content_hash`` → ``asset.file_hash`` 와 매칭한다.
    트랜잭션 경계는 호출자가 제어한다(conn-우선 관례). graph 무기록(측정 전용·SC-004).

    Returns:
        ``(mapping, missing)`` — ``mapping``=키→asset_id, ``missing``=해소 안 된 키(정렬·결정적).
    """
    col = _KEY_COLUMN[golden.key_type]  # 화이트리스트(parse_golden 검증 통과 key_type)
    keys = {k for p in golden.pairs for k in (p.a, p.b)} | set(golden.isolated)
    mapping: dict[str, str] = {}
    if keys:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {col} AS k, asset_id FROM asset WHERE {col} = ANY(%s)",  # noqa: S608 (화이트리스트 식별자)
                (list(keys),),
            )
            for k, aid in cur.fetchall():
                mapping[str(k)] = str(aid)
    missing = sorted(k for k in keys if k not in mapping)
    return mapping, missing
