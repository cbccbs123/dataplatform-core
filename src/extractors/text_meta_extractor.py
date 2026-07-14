"""텍스트/문서 본문 통계 메타(언어·인코딩·문장수·토큰수·길이) 추출 — text_skill 이 호출.

본문을 청크로 순회하며 집계만 한다. 임베딩 벡터는 ``src/embedders/text_embedder.py`` 가
별도로 만든다(추출/임베딩 분리 설계). 여기서 모델을 로드하는 이유는 임베딩이 아니라
임베딩 모델의 **토크나이저**로 토큰 수를 세기 위해서다(``count_tokens`` 참조).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

# 069 P1-7: 동명 자체 lru_cache 로더(별도 캐시)로 같은 체크포인트를 **이중 로드**하던 것을
# text_embedder 의 프로세스 캐시를 공유(re-export)해 해소한다 — 기본 채널(st)에서 추출·임베딩이
# 같은 model_name 을 쓰므로 로드 1회로 수렴(GB 단위 가중치·수 초 절감). 여기서 모델이 필요한
# 이유는 임베딩이 아니라 **토크나이저**(count_tokens)뿐이라 공유가 안전하다.
from src.embedders.text_embedder import get_embedding_model
from src.file.data_loader import (
    MAX_INPUT_CHARS,
    choose_encoding,
    iter_document_chunks,
    normalize_file_kind,
)

__all__ = ["EmbeddingTextMeta", "count_tokens", "extract_text_meta", "get_embedding_model"]


class EmbeddingTextMeta(TypedDict):
    language: str
    encoding: str
    num_sentences: int
    num_tokens: int
    length: int


def count_tokens(text: str, *, model_name: str) -> int:
    if not text:
        return 0
    model = get_embedding_model(model_name)
    token_ids = model.tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


def _detect_language_from_counts(*, hangul_count: int, latin_count: int) -> str:
    # 글자 종류 비율 기반 휴리스틱(사전·LLM 없이 결정적). 한글은 라틴보다 낮은 임계(0.3)를
    # 쓴다 — 한글 문서도 영문 용어가 흔히 섞여 한글 비중이 낮게 나오기 때문.
    # 라틴 0.5 미만이고 한글도 0.3 미만이면 판별 불가로 unknown.
    total_letters = hangul_count + latin_count
    if total_letters == 0:
        return "unknown"
    if (hangul_count / total_letters) >= 0.3:
        return "ko"
    if (latin_count / total_letters) >= 0.5:
        return "en"
    return "unknown"


def _count_sentences(text: str) -> int:
    if not text.strip():
        return 0
    parts = re.split(r"[.!?]+|\n+", text)
    return len([p for p in parts if p.strip()])


def extract_text_meta(
    file_path: str | Path,
    *,
    file_kind: str,
    encoding: str = "utf-8",
    chunk_size: int = 512,
    embedding_model_name: str = "BM-K/KoSimCSE-roberta-multitask",#"BAAI/bge-m3",
) -> EmbeddingTextMeta:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    kind = normalize_file_kind(file_kind)
    if kind is None:
        raise ValueError("file_kind는 필수입니다.")

    effective_encoding = "UTF-8"
    effective_encoding = choose_encoding(path, encoding).upper()

    num_sentences = 0
    num_tokens = 0
    length = 0
    hangul_count = 0
    latin_count = 0

    for chunk in iter_document_chunks(
        path,
        file_kind=kind,
        encoding=encoding,
        chunk_size=chunk_size,
        overlap_size=0,
        max_input_chars=MAX_INPUT_CHARS,
    ):
        if not chunk:
            continue
        length += len(chunk)
        num_sentences += _count_sentences(chunk)
        num_tokens += count_tokens(chunk, model_name=embedding_model_name)
        hangul_count += len(re.findall(r"[가-힣]", chunk))
        latin_count += len(re.findall(r"[A-Za-z]", chunk))

    language = _detect_language_from_counts(
        hangul_count=hangul_count,
        latin_count=latin_count,
    )

    return {
        "language": language,
        "encoding": effective_encoding,
        "num_sentences": num_sentences,
        "num_tokens": num_tokens,
        "length": length,
    }
