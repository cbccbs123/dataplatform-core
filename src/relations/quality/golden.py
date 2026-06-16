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

# 허용 키 타입 — fs_path(파일 경로) 또는 content_hash(파일 해시).
_KEY_TYPES = ("fs_path", "content_hash")


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
