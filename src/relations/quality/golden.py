"""정답 관계셋(골든)의 데이터 모델과 순수 파싱·검증.

골든은 **사람이 확인한 정답**이다 — `pairs` 는 관계가 있어야 하는 쌍, `isolated` 는 어떤 관계도
없어야 하는 자산이다.

자산을 asset_id 가 아니라 **파일 경로나 해시로** 지정하는 이유: 재적재하면 id 가 바뀌지만 파일은
그대로라, id 로 적으면 골든이 통째로 무효가 된다. 실제 id 로의 변환은 DB 가 필요해 별도 함수다.
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
    """골든 dict 를 검증해 ``Golden`` 으로 변환한다(순수 함수).

    Args:
        data: 골든 픽스처 JSON 을 읽은 dict. ``version``·``key_type``·``pairs``·``isolated`` 를 본다.

    Returns:
        불변 ``Golden`` 객체.

    Raises:
        ValueError: 버전 불일치 · 미지원 ``key_type`` · 쌍의 필드 누락 · 자기 자신과의 쌍.
            **관대하게 넘기지 않는다** — 잘못된 골든으로 잰 수치는 틀린 결론을 낳기 때문이다.
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
    """골든 키(파일 경로 또는 해시)를 현재 ``asset_id`` 로 바꾼다(DB 조회·읽기 전용).

    ``key_type=fs_path`` → ``asset.fs_path``, ``content_hash`` → ``asset.file_hash`` 와 매칭한다.
    **조회 전용**(그래프에 아무것도 쓰지 않는다 — 측정 전용). 트랜잭션 경계는 호출자가 제어한다.

    Args:
        golden: ``parse_golden`` 을 통과한 골든셋. ``key_type`` 이 어느 컬럼과 맞출지 결정한다.

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
