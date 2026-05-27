"""Stage 3 — 온프레미스 LLM zero-shot 도메인 판정(동적 라벨).

**모든 LLM 은 온프레미스 전용**(`cfg.openai_*`, OpenAI 호환 온프레미스 서버). 외부 LLM 미사용.
허용 라벨 집합은 등록 도메인 + general 로 호출부(cascade)가 동적으로 만들어 넘긴다.
``complete`` 주입으로 네트워크 없이 테스트 가능. 실패/미초기화 → 'review'(HITL).
"""
from __future__ import annotations

import json
from typing import Any, Callable

from src.classify.types import DOMAIN_REVIEW

_MAX_TEXT = 4000


def _build_prompt(labels: list[str]) -> str:
    opts = " 또는 ".join(labels)
    return (
        f"다음 텍스트가 어느 도메인인지 판정해라. 후보: {opts}. "
        '반드시 JSON 만 출력: {"label": "<후보 중 하나>"}\n\n텍스트:\n'
    )


def _default_complete(prompt: str) -> str:
    """온프레미스 OpenAI 호환 엔드포인트 호출(지연 import)."""
    from openai import OpenAI

    from src.config.settings import get_current_settings

    cfg = get_current_settings()
    client = OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)
    resp = client.chat.completions.create(
        model=cfg.meta_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return (resp.choices[0].message.content or "").strip()


def classify(
    text: str,
    labels: list[str],
    *,
    complete: Callable[[str], str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """온프레미스 LLM 으로 labels 중 하나 판정. 허용 외/실패 → 'review'."""
    complete = complete or _default_complete
    try:
        raw = complete(_build_prompt(labels) + text[:_MAX_TEXT])
        data = json.loads(raw) if raw else {}
        label = str(data.get("label", "")).strip().lower()
        if label in labels:
            return label, {"stage3": "llm", "label": label}
        return DOMAIN_REVIEW, {"stage3": "llm_unclear", "raw": raw[:200]}
    except Exception as exc:  # noqa: BLE001 — 미초기화/호출 실패는 review 로 흡수
        return DOMAIN_REVIEW, {"stage3": "error", "error": str(exc)}
