from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from src.config.settings import get_current_settings
from src.file.data_loader import (
    MAX_INPUT_CHARS,
    iter_document_chunks,
    normalize_file_kind,
)
from src.llm.client import complete_json


class SummaryKeywords(TypedDict):
    summary: str
    keywords: list[str]
    stt: str


def _summarize_chunk_only(
    chunk_text: str,
    *,
    summary_max_chars: int,
) -> str:
    """청크 하나를 요약해 문자열로 반환한다. 공통 seam(complete_json) 사용."""
    prompt = (
        "다음 텍스트를 간단히 요약해서 반드시 JSON만 출력해.\n"
        "형식:\n"
        '{ "summary": "요약" }\n'
        f"- summary는 {summary_max_chars}자 이내\n\n"
        f"텍스트:\n{chunk_text}"
        "개수/비율/합계 같은 통계 표현 금지."
    )
    data = complete_json(prompt)
    return str(data.get("summary", "")).strip()[:summary_max_chars]


def summarize_and_extract_keywords(
    file_path: str | Path,
    *,
    file_kind: str,
) -> SummaryKeywords:
    """문서 요약·키워드 추출. LLM 은 설정된 단일 온프레미스 엔드포인트(cfg.openai_*)를 사용한다."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    kind = normalize_file_kind(file_kind)
    if kind is None:
        raise ValueError("file_kind는 필수입니다.")

    cfg = get_current_settings()

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
        s = _summarize_chunk_only(ch, summary_max_chars=cfg.summary_max_chars)
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
    data = complete_json(final_prompt)

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
    data = complete_json(final_prompt)

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