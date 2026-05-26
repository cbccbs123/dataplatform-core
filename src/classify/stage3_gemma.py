"""F-5.1 Stage 3 — 온프레미스 LLM zero-shot 도메인 판정.

**모든 LLM 은 온프레미스 전용** — 설정된 단일 엔드포인트(`cfg.openai_*`, OpenAI 호환
온프레미스 서버)를 사용한다. 외부 LLM API 미사용.
Stage 1/2 가 결정 못한 모호 케이스에만 호출(≤10% 게이트는 호출 위치로 보장).
설정 미초기화/호출 실패 시 'review'(HITL)로 폴백.
"""

from __future__ import annotations

import json
from typing import Any

from src.classify.types import DOMAIN_GENERAL, DOMAIN_MEDICAL, DOMAIN_REVIEW

_MAX_TEXT = 4000
_PROMPT = (
    "다음 텍스트가 의료(medical) 도메인인지 일반(general) 도메인인지 판정해라. "
    '반드시 JSON 만 출력: {"label": "medical" 또는 "general"}\n\n텍스트:\n'
)


def classify(text: str) -> tuple[str, dict[str, Any]]:
    """온프레미스 LLM(cfg.openai_*)으로 medical|general 판정. 실패 시 'review'."""
    try:
        from openai import OpenAI

        from src.config.settings import get_current_settings

        cfg = get_current_settings()
        client = OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)
        resp = client.chat.completions.create(
            model=cfg.meta_model,
            messages=[{"role": "user", "content": _PROMPT + text[:_MAX_TEXT]}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw) if raw else {}
        label = str(data.get("label", "")).strip().lower()
        if label in (DOMAIN_MEDICAL, DOMAIN_GENERAL):
            return label, {"stage3": "llm", "label": label}
        return DOMAIN_REVIEW, {"stage3": "llm_unclear", "raw": raw[:200]}
    except Exception as exc:  # noqa: BLE001 — 미초기화/호출 실패는 review 로 흡수
        return DOMAIN_REVIEW, {"stage3": "error", "error": str(exc)}
