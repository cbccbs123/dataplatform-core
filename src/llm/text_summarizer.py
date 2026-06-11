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
    stt: str  # 오디오 전용 필드. 텍스트/이미지/영상 경로에서는 채우지 않는다.


# 요약 토픽화 지시(spec 026 FR-001) — image/video summarizer 와 동일 의도. 텍스트·STT 요약도
# 임베딩·BM25 의 단일 원천이라 매체 문형을 막고 내용·주제 중심으로 강제한다. 산출 JSON 키·파싱은
# 불변, 이 지시 문구만 추가한다.
_SUMMARY_TOPIC_INSTRUCTION = (
    "- summary 는 매체 자체를 언급하지 말 것 — '이미지입니다/사진입니다/영상입니다/썸네일' 같은 "
    "매체 단어·문형 금지\n"
    "- 담긴 내용·주제·개체를 명사구 중심으로 서술\n"
)


def _build_chunk_summary_prompt(chunk_text: str, *, summary_max_chars: int) -> str:
    """청크 1개 요약(map) 프롬프트(순수 — LLM·settings 미접촉). 토픽화 지시를 단위로 가드 가능하게 노출."""
    return (
        "다음 텍스트를 간단히 요약해서 반드시 JSON만 출력해.\n"
        "형식:\n"
        '{ "summary": "요약" }\n'
        f"- summary는 {summary_max_chars}자 이내\n"
        + _SUMMARY_TOPIC_INSTRUCTION
        + f"\n텍스트:\n{chunk_text}"
        + "개수/비율/합계 같은 통계 표현 금지."
    )


def _build_reduce_prompt(merged: str, *, summary_max_chars: int, top_k_keywords: int) -> str:
    """청크 요약 목록 종합(reduce) 프롬프트(순수). 토픽화 지시를 단위로 가드 가능하게 노출."""
    return (
        "아래는 긴 문서의 청크별 요약 목록이다. 이를 종합해 반드시 JSON만 출력해.\n"
        "형식:\n"
        '{ "summary": "최종 요약", "keywords": ["키워드1", "키워드2"] }\n'
        f"- summary는 {summary_max_chars}자 이내\n"
        f"- keywords는 핵심 키워드 최대 {top_k_keywords}개\n"
        + _SUMMARY_TOPIC_INSTRUCTION
        + f"\n청크 요약 목록:\n{merged}"
        + "개수/비율/합계 같은 통계 표현 금지."
    )


def _build_audio_prompt(text: str, *, summary_max_chars: int, top_k_keywords: int) -> str:
    """STT 전사 전문 요약·키워드 프롬프트(순수). 토픽화 지시를 단위로 가드 가능하게 노출."""
    return (
        "아래는 긴 문서의 청크별 요약 목록이다. 이를 종합해 반드시 JSON만 출력해.\n"
        "형식:\n"
        '{ "summary": "최종 요약", "keywords": ["키워드1", "키워드2"] }\n'
        f"- summary는 {summary_max_chars}자 이내\n"
        f"- keywords는 핵심 키워드 최대 {top_k_keywords}개\n"
        + _SUMMARY_TOPIC_INSTRUCTION
        + f"\n텍스트:\n{text}"
        + "개수/비율/합계 같은 통계 표현 금지."
    )


def _summarize_chunk_only(
    chunk_text: str,
    *,
    summary_max_chars: int,
) -> str:
    """청크 하나를 요약해 문자열로 반환한다. 공통 seam(complete_json) 사용."""
    prompt = _build_chunk_summary_prompt(chunk_text, summary_max_chars=summary_max_chars)
    data = complete_json(prompt)
    return str(data.get("summary", "")).strip()[:summary_max_chars]


def summarize_and_extract_keywords(
    file_path: str | Path,
    *,
    file_kind: str,
) -> SummaryKeywords:
    """문서 요약·키워드 추출. LLM 은 설정된 단일 온프레미스 엔드포인트(cfg.openai_*)를 사용한다.

    청크별 요약을 먼저 생성(map)한 뒤, 합쳐서 최종 요약·키워드를 추출(reduce)하는 2단계 방식.
    문서가 청크 하나에 수렴하더라도 map→reduce 경로를 동일하게 밟으므로
    긴 문서와 짧은 문서 모두 동일한 코드 경로를 따른다.

    반환값은 ``SummaryKeywords`` TypedDict 이지만 ``stt`` 필드를 채우지 않는다.
    오디오 경로는 ``summarize_and_extract_keywords_from_audio`` 를 사용할 것.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    kind = normalize_file_kind(file_kind)
    if kind is None:
        raise ValueError("file_kind는 필수입니다.")

    cfg = get_current_settings()

    partial_summaries: list[str] = []
    for _i, ch in enumerate(
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
    final_prompt = _build_reduce_prompt(
        merged, summary_max_chars=cfg.summary_max_chars, top_k_keywords=cfg.top_k_keywords
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
    """STT 전사 텍스트를 받아 요약·키워드를 추출하고 ``stt`` 필드도 채워 반환한다.

    오디오 skill 에서 Whisper STT 결과 전문(full transcript)을 그대로 넘기면
    하나의 프롬프트로 처리된다(청크 분할 없음 — 긴 STT 의 경우 max_input_chars 주의).
    ``stt`` 필드에는 원본 전사 텍스트를 그대로 보존해 embed 슬롯이 재사용할 수 있도록 한다.
    """
    cfg = get_current_settings()

    final_prompt = _build_audio_prompt(
        text, summary_max_chars=cfg.summary_max_chars, top_k_keywords=cfg.top_k_keywords
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
