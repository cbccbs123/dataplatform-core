"""F-5.1 Stage 2 — 의료 어휘 사전 기반 zero-shot 점수.

텍스트(파일명 + 내용 스니펫)에서 의료 어휘 빈도로 판정한다(학습 없음).
- 의료 어휘 hit ≥ 2 → medical 확정
- hit == 0 → general 확정
- hit == 1 → 모호(None) → Stage 3 로 위임

(설계의 CLIP/ST zero-shot 임베딩 보강은 후속. 현재는 결정적 규칙 매칭.)
"""

from __future__ import annotations

from typing import Any

from src.classify.medical_terms import MEDICAL_TERMS
from src.classify.types import DOMAIN_GENERAL, DOMAIN_MEDICAL

_MEDICAL_HIT_THRESHOLD = 2


def score(text: str) -> tuple[str | None, dict[str, Any]]:
    """(label, signal). label 은 'medical'|'general'|None(모호)."""
    low = (text or "").lower()
    hits = sorted({t for t in MEDICAL_TERMS if t in low})
    n = len(hits)
    signal: dict[str, Any] = {"medical_term_hits": n, "terms": hits[:20]}
    if n >= _MEDICAL_HIT_THRESHOLD:
        return DOMAIN_MEDICAL, signal
    if n == 0:
        return DOMAIN_GENERAL, signal
    return None, signal  # 모호 → Stage 3
