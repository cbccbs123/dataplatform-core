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
    """현재 설정의 온프레미스 OpenAI 호환 클라이언트.

    ``cfg.openai_base_url``은 내부 LLM 서버(Ollama / vLLM 등) 엔드포인트.
    외부 OpenAI SaaS가 아닌 온프레미스 서버를 가리키는 것이 과제 정책 요건이다.
    테스트에서는 직접 이 함수를 호출하지 않고 ``client=`` 인자로 모의 클라이언트를 주입한다.
    """
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
    """JSON-mode 호출 후 원문 문자열 반환('' if 빈응답). 호출부가 직접 파싱할 때.

    ``response_format=json_object`` 를 강제해 LLM 이 마크다운 코드블록 없이
    순수 JSON 만 출력하도록 유도한다. 단, 내용 보장은 아니므로 파싱은 호출부 책임.

    ``temperature=0.0`` 기본값은 결정 재현성 100% 보장(과제 요건)을 위한 것이다.
    변경이 필요하면 명시적으로 전달해야 하며, 의료 경로에서는 절대 0 이상으로 올리지 말 것.
    """
    client = client or get_llm_client()
    resp = client.chat.completions.create(
        model=_model_or_default(model),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_json_or_empty(raw: str) -> dict[str, Any]:
    """LLM 응답 원문을 dict 로 변환한다.

    JSON 배열·문자열·파싱 실패 등 비객체 응답은 모두 ``{}`` 로 정규화해
    호출부가 ``.get()`` 으로 안전하게 접근하도록 한다.
    """
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
    """이미지+텍스트 비전 호출 → JSON dict(빈/파싱실패 → ``{}``).

    ``image_data_url`` 은 ``data:image/jpeg;base64,...`` 형태여야 한다.
    이미지 summarizer 가 PIL 로 리사이즈 + JPEG 재인코딩 후 전달하므로
    여기서는 포맷 검증 없이 그대로 사용한다.
    비전 모델은 ``complete_text`` / ``complete_json`` 과 같은 온프레미스 엔드포인트를 공유한다
    (다른 모델명이 필요하면 ``model=`` 으로 명시).
    """
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
