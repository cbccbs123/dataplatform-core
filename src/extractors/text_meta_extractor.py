from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Sequence, TypedDict

from sentence_transformers import SentenceTransformer

from src.file.data_loader import (
    choose_encoding,
    iter_document_chunks,
    normalize_file_kind,
    MAX_INPUT_CHARS,
)


class EmbeddingTextMeta(TypedDict):
    language: str
    encoding: str
    num_sentences: int
    num_tokens: int
    length: int


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_embedding_dimension(model_name: str) -> int:
    model = get_embedding_model(model_name)
    return int(model.get_sentence_embedding_dimension())


def ensure_embedding_dimension(model_name: str, expected_dimension: int) -> None:
    actual = get_embedding_dimension(model_name)
    if actual != expected_dimension:
        raise ValueError(
            f"임베딩 차원 불일치: model={model_name!r}의 차원은 {actual}D, "
            f"요청 차원은 {expected_dimension}D 입니다."
        )


def count_tokens(text: str, *, model_name: str) -> int:
    if not text:
        return 0
    model = get_embedding_model(model_name)
    token_ids = model.tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    normalize_embeddings: bool = True,
) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedding_model(model_name)
    vectors = model.encode(
        list(texts),
        normalize_embeddings=normalize_embeddings,
    )
    return vectors.tolist()
    

def _detect_language_from_counts(*, hangul_count: int, latin_count: int) -> str:
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
