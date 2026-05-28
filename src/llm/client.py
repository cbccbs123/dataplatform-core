"""온프레미스 LLM 호출 공통 seam.

**모든 LLM 은 온프레미스 전용**(`cfg.openai_*`, OpenAI 호환 서버). 외부 LLM 미사용, temperature=0.
호출지는 ``complete_text``/``complete_json``/``complete_vision_json`` 만 쓰고, ``client=`` 주입으로
네트워크 없이 테스트한다. 향후 정책 엔진(onprem_llm/NoExternalLLM·결정성)이 붙는 단일 chokepoint.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.config.settings import get_current_settings


def get_llm_client() -> OpenAI:
    """현재 설정의 온프레미스 OpenAI 호환 클라이언트."""
    cfg = get_current_settings()
    return OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)


def _model_or_default(model: str | None) -> str:
    """명시된 모델이 있으면 그대로, 없으면 현재 설정의 meta_model을 반환.

    settings 미초기화(테스트 등) 시 빈 문자열을 반환한다 — client 주입 테스트에서는
    실제 API 호출이 없으므로 model 값이 검증되지 않는다.
    """
    if model:
        return model
    try:
        return get_current_settings().meta_model
    except RuntimeError:
        return ""


def complete_text(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    client: OpenAI | None = None,
) -> str:
    """JSON-mode 호출 후 원문 문자열 반환('' if 빈응답). 호출부가 직접 파싱할 때."""
    client = client or get_llm_client()
    resp = client.chat.completions.create(
        model=_model_or_default(model),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_json_or_empty(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(out) if isinstance(out, dict) else {}


def complete_json(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """``complete_text`` + JSON 파싱. 빈응답/파싱실패/비객체 → ``{}``."""
    return _parse_json_or_empty(
        complete_text(prompt, model=model, temperature=temperature, client=client)
    )


def complete_vision_json(
    *,
    text: str,
    image_data_url: str,
    model: str | None = None,
    temperature: float = 0.0,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """이미지+텍스트 비전 호출 → JSON dict(빈/파싱실패 → ``{}``)."""
    client = client or get_llm_client()
    resp = client.chat.completions.create(
        model=_model_or_default(model),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return _parse_json_or_empty((resp.choices[0].message.content or "").strip())
