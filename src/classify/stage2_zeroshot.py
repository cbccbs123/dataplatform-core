"""Stage 2 — 어휘 사전 기반 zero-shot 점수(도메인-불가지).

주어진 어휘 사전(lexicon)에 대해 텍스트의 hit 수를 센다(학습 없음). 도메인 판정
(임계·margin)은 cascade 엔진이 담당하고, 본 모듈은 순수 카운팅만 한다.

매칭 규칙
    - 영문/ASCII 어휘는 **단어 경계 매칭**(예: 'contact' 의 'ct' 오탐 방지).
    - 한글 어휘는 부분 문자열 매칭(어절 경계 모호).
"""
from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=64)
def _compiled(lexicon: frozenset[str]):
    """어휘 사전 → (라틴 단어경계 정규식 | None, 한글 용어 집합). 사전별 캐시."""
    latin = sorted({t for t in lexicon if t.isascii()}, key=len, reverse=True)
    korean = frozenset(t for t in lexicon if not t.isascii())
    latin_re = (
        re.compile(r"\b(?:" + "|".join(re.escape(t) for t in latin) + r")\b", re.IGNORECASE)
        if latin
        else None
    )
    return latin_re, korean


def count_hits(text: str, lexicon: frozenset[str]) -> tuple[int, list[str]]:
    """(hit 수, 매칭 용어 ≤20). 어휘 사전당 단어경계(ASCII)/부분문자열(한글) 매칭."""
    text = text or ""
    latin_re, korean = _compiled(lexicon)
    hits: set[str] = set()
    if latin_re is not None:
        hits |= {m.group(0).lower() for m in latin_re.finditer(text)}
    hits |= {t for t in korean if t in text}
    return len(hits), sorted(hits)[:20]
