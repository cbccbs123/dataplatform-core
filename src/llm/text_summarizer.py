from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from openai import OpenAI

from src.config.settings import get_current_settings
from src.file.data_loader import (
    MAX_INPUT_CHARS,
    iter_document_chunks,
    normalize_file_kind,
)


class SummaryKeywords(TypedDict):
    summary: str
    keywords: list[str]
    stt: str


def _call_openai_json(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float = 0.0,
) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": raw, "keywords": []}


def _summarize_chunk_only_openai(
    chunk_text: str,
    *,
    client: OpenAI,
    model: str,
    summary_max_chars: int,
) -> str:
    prompt = (
        "다음 텍스트를 간단히 요약해서 반드시 JSON만 출력해.\n"
        "형식:\n"
        '{ "summary": "요약" }\n'
        f"- summary는 {summary_max_chars}자 이내\n\n"
        f"텍스트:\n{chunk_text}"
        "개수/비율/합계 같은 통계 표현 금지."
    )
    data = _call_openai_json(client, model=model, prompt=prompt)
    return str(data.get("summary", "")).strip()[:summary_max_chars]


def summarize_and_extract_keywords(
    file_path: str | Path,
    *,
    file_kind: str
) -> SummaryKeywords:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    kind = normalize_file_kind(file_kind)
    if kind is None:
        raise ValueError("file_kind는 필수입니다.")

    cfg = get_current_settings()

    client = OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)

    partial_summaries: list[str] = []
    for i, ch in enumerate(
        iter_document_chunks(
            path,
            file_kind=kind,
            encoding=cfg.encoding,
            chunk_size=cfg.chunk_size,
            overlap_size=cfg.overlap_size,
            max_input_chars=MAX_INPUT_CHARS,
        )
    ):
        print(f"chunk {i}: {i * cfg.chunk_size}")
        s = _summarize_chunk_only_openai(
            ch,
            client=client,
            model=cfg.meta_model,
            summary_max_chars=cfg.summary_max_chars,
        )
        if s:
            partial_summaries.append(s)

    if not partial_summaries:
        return {"summary": "", "keywords": []}

    merged = "\n".join(f"- {s}" for s in partial_summaries)
    final_prompt = (
        "아래는 긴 문서의 청크별 요약 목록이다. 이를 종합해 반드시 JSON만 출력해.\n"
        "형식:\n"
        '{ "summary": "최종 요약", "keywords": ["키워드1", "키워드2"] }\n'
        f"- summary는 {cfg.summary_max_chars}자 이내\n"
        f"- keywords는 핵심 키워드 최대 {cfg.top_k_keywords}개\n\n"
        f"청크 요약 목록:\n{merged}"
        "개수/비율/합계 같은 통계 표현 금지."
    )
    data = _call_openai_json(client, model=cfg.meta_model, prompt=final_prompt)

    summary = str(data.get("summary", "")).strip()
    keywords_raw = data.get("keywords", [])
    if not isinstance(keywords_raw, list):
        keywords_raw = []

    keywords: list[str] = []
    for kw in keywords_raw:
        k = str(kw).strip()
        if k and k not in keywords:
            keywords.append(k)
        if len(keywords) >= cfg.top_k_keywords:
            break

    return {"summary": summary, "keywords": keywords}


def summarize_and_extract_keywords_from_audio(
    text: str,
) -> SummaryKeywords:
    cfg = get_current_settings()

    final_prompt = (
        "아래는 긴 문서의 청크별 요약 목록이다. 이를 종합해 반드시 JSON만 출력해.\n"
        "형식:\n"
        '{ "summary": "최종 요약", "keywords": ["키워드1", "키워드2"] }\n'
        f"- summary는 {cfg.summary_max_chars}자 이내\n"
        f"- keywords는 핵심 키워드 최대 {cfg.top_k_keywords}개\n\n"
        f"텍스트:\n{text}"
        "개수/비율/합계 같은 통계 표현 금지."
    )
    client = OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)
    data = _call_openai_json(client, model=cfg.meta_model, prompt=final_prompt)

    summary = str(data.get("summary", "")).strip()
    keywords_raw = data.get("keywords", [])
    if not isinstance(keywords_raw, list):
        keywords_raw = []

    keywords: list[str] = []
    for kw in keywords_raw:
        k = str(kw).strip()
        if k and k not in keywords:
            keywords.append(k)
        if len(keywords) >= cfg.top_k_keywords:
            break

    return {"summary": summary, "keywords": keywords, "stt": text}