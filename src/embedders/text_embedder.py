from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence, TypedDict

from sentence_transformers import SentenceTransformer
import numpy as np
from src.preprocess.text_embedding_normalize import normalize_text_for_embedding
from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.file.data_loader import (
    iter_document_chunks,
    iter_plain_text_chunks,
    normalize_file_kind,
    MAX_INPUT_CHARS,
)


class TextEmbeddingResult(TypedDict):
    embedding_vector: list[float]


class TextChunkEmbedding(TypedDict):
    chunk_index: int
    embedding_vector: list[float]


def pad_embedding_to_storage_dim(raw: list[float]) -> list[float]:
    """모델 출력 벡터를 DB ``vector(FIX_EMBEDDING_DIMENSION)`` 저장 형식으로 맞춘다."""
    vec = np.asarray(raw, dtype=np.float32)
    if vec.size == 0:
        vec = np.zeros((FIX_EMBEDDING_DIMENSION,), dtype=np.float32)
    if vec.shape[0] < FIX_EMBEDDING_DIMENSION:
        vec = np.pad(
            vec,
            (0, FIX_EMBEDDING_DIMENSION - vec.shape[0]),
            mode="constant",
            constant_values=0.0,
        )
    return vec.tolist()


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


def embedding_text_meta(
    file_path: str | Path,
    *,
    file_kind: str,
    encoding: str = "utf-8",
    chunk_size: int = 512,
    embedding_model_name: str = "BM-K/KoSimCSE-roberta-multitask",#"BAAI/bge-m3",
    normalize_embeddings: bool = True,
) -> TextEmbeddingResult:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    kind = normalize_file_kind(file_kind)
    if kind is None:
        raise ValueError("file_kind는 필수입니다.")

    weighted_sum: np.ndarray | None = None
    total_weight = 0.0

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

        chunk = normalize_text_for_embedding(chunk)
        if not chunk.strip():
            chunk = " "

        chunk_vector = embed_texts(
            [chunk],
            model_name=embedding_model_name,
            normalize_embeddings=normalize_embeddings,
        )[0]
        chunk_arr = np.asarray(chunk_vector, dtype=np.float32)
        weight = float(max(1, len(chunk)))
        if weighted_sum is None:
            weighted_sum = np.zeros_like(chunk_arr, dtype=np.float32)
        weighted_sum += chunk_arr * weight
        total_weight += weight

    if weighted_sum is None or total_weight == 0:
        embedding_vector = []
    else:
        embedding_vector = (weighted_sum / total_weight).tolist()

    print("embedding_vector Length:", len(embedding_vector))

    padded_embedding = pad_embedding_to_storage_dim(embedding_vector)
    print("padded_embeddings Length:", len(padded_embedding))
    return {"embedding_vector": padded_embedding}


def _iter_nonempty_chunks(
    path: Path,
    *,
    file_kind: str,
    encoding: str,
    chunk_size: int,
) -> Iterator[tuple[int, str]]:
    kind = normalize_file_kind(file_kind)
    if kind is None:
        raise ValueError("file_kind는 필수입니다.")
    idx = 0
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
        yield idx, chunk
        idx += 1


def embedding_text_chunks(
    file_path: str | Path,
    *,
    file_kind: str,
    encoding: str = "utf-8",
    chunk_size: int = 512,
    embedding_model_name: str = "BM-K/KoSimCSE-roberta-multitask",
    normalize_embeddings: bool = True,
) -> list[TextChunkEmbedding]:
    """문서를 청크 단위로 임베딩한다. ``media_chunks`` 임베딩 적재용(본문 텍스트는 DB에 저장하지 않음)."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    out: list[TextChunkEmbedding] = []
    for chunk_index, chunk in _iter_nonempty_chunks(
        path,
        file_kind=file_kind,
        encoding=encoding,
        chunk_size=chunk_size,
    ):
        clean = normalize_text_for_embedding(chunk)
        if not clean.strip():
            clean = " "
        chunk_vector = embed_texts(
            [clean],
            model_name=embedding_model_name,
            normalize_embeddings=normalize_embeddings,
        )[0]
        out.append(
            {
                "chunk_index": chunk_index,
                "embedding_vector": pad_embedding_to_storage_dim(chunk_vector),
            }
        )
    if not out:
        out.append(
            {
                "chunk_index": 0,
                "embedding_vector": pad_embedding_to_storage_dim([]),
            }
        )
    return out


def embedding_plain_text_chunks(
    text: str,
    *,
    chunk_size: int,
    embedding_model_name: str = "BM-K/KoSimCSE-roberta-multitask",
    normalize_embeddings: bool = True,
    overlap_size: int = 0,
    max_input_chars: int = MAX_INPUT_CHARS,
) -> list[TextChunkEmbedding]:
    """STT 등 단일 문자열을 청크 단위로 임베딩한다. ``media_chunks`` 임베딩 적재용(본문은 DB 미저장)."""
    out: list[TextChunkEmbedding] = []
    chunk_index = 0
    for chunk in iter_plain_text_chunks(
        text,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        max_input_chars=max_input_chars,
    ):
        if not chunk:
            continue
        clean = normalize_text_for_embedding(chunk)
        if not clean.strip():
            clean = " "
        chunk_vector = embed_texts(
            [clean],
            model_name=embedding_model_name,
            normalize_embeddings=normalize_embeddings,
        )[0]
        out.append(
            {
                "chunk_index": chunk_index,
                "embedding_vector": pad_embedding_to_storage_dim(chunk_vector),
            }
        )
        chunk_index += 1
    if not out:
        out.append(
            {
                "chunk_index": 0,
                "embedding_vector": pad_embedding_to_storage_dim([]),
            }
        )
    return out
