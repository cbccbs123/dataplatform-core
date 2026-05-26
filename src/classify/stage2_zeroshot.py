"""F-5.1 Stage 2 — 의료 어휘 사전 기반 zero-shot 점수.

텍스트(파일명 + 추출 텍스트)에서 의료 어휘 빈도로 판정한다(학습 없음).
- 의료 어휘 hit ≥ 2 → medical 확정
- hit == 0 → general 확정
- hit == 1 → 모호(None) → Stage 3 로 위임

매칭 규칙
    - 영문/ASCII 어휘(ct, mri, patient …)는 **단어 경계 매칭**으로 부분문자열 오탐 방지
      (예: "contact" 의 "ct" 를 의료로 오인하지 않음).
    - 한글 어휘는 부분 문자열 매칭(어절 경계가 모호하므로).
"""

from __future__ import annotations

import re
from typing import Any

from src.classify.medical_terms import MEDICAL_TERMS
from src.classify.types import DOMAIN_GENERAL, DOMAIN_MEDICAL

_MEDICAL_HIT_THRESHOLD = 2

# ASCII(영문/약어)와 비ASCII(한글) 분리.
_LATIN_TERMS = sorted({t for t in MEDICAL_TERMS if t.isascii()}, key=len, reverse=True)
_KOREAN_TERMS = frozenset(t for t in MEDICAL_TERMS if not t.isascii())
# 단어 경계로 영문 약어 매칭(대소문자 무시).
_LATIN_RE = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in _LATIN_TERMS) + r")\b", re.IGNORECASE)


def score(text: str) -> tuple[str | None, dict[str, Any]]:
    """(label, signal). label 은 'medical'|'general'|None(모호)."""
    text = text or ""
    hits: set[str] = {m.group(0).lower() for m in _LATIN_RE.finditer(text)}
    hits |= {t for t in _KOREAN_TERMS if t in text}
    n = len(hits)
    signal: dict[str, Any] = {"medical_term_hits": n, "terms": sorted(hits)[:20]}
    if n >= _MEDICAL_HIT_THRESHOLD:
        return DOMAIN_MEDICAL, signal
    if n == 0:
        return DOMAIN_GENERAL, signal
    return None, signal  # 모호 → Stage 3
