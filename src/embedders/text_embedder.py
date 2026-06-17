"""텍스트 임베딩 seam — SentenceTransformer 추론(학습 배제·inference only).

per-asset 파이프라인의 ST 채널 벡터를 여기서 만든다: 텍스트/오디오(STT) 본문 청크와
이미지/영상의 VLM 텍스트(캡션·키워드·라벨)가 모두 이 모듈을 거친다(``src/skills/*_skill.py``).
검색 쿼리 측(``src/search/query_embed.py`` — OpenSearch kNN 질의 임베딩)도 동일 함수
(``embed_texts``·``pad_embedding_to_storage_dim``)를 공유한다 — 인덱싱과 질의가 같은 벡터 공간이어야
코사인 유사도가 성립하기 때문이다. 따라서 모델 체크포인트는 적재·검색이 동일해야 한다.

모든 출력 벡터는 ``pad_embedding_to_storage_dim`` 으로 DB ``vector(1536)`` 차원에 맞춘다
(헌법: 1536D 통일·PG17+pgvector). 본문 텍스트 자체는 DB에 저장하지 않고 벡터만 적재한다.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.file.data_loader import (
    MAX_INPUT_CHARS,
    iter_document_chunks,
    iter_plain_text_chunks,
    normalize_file_kind,
)
from src.preprocess.text_embedding_normalize import normalize_text_for_embedding


class TextChunkEmbedding(TypedDict):
    chunk_index: int
    embedding_vector: list[float]


def pad_embedding_to_storage_dim(raw: list[float]) -> list[float]:
    """모델 출력 벡터를 DB ``vector(FIX_EMBEDDING_DIMENSION)`` 저장 형식으로 맞춘다.

    - 빈 리스트(추출 실패 fallback) → 전체 zero 벡터.
    - 모델 차원 < 1536 → 뒤를 0 으로 패딩(다른 체크포인트 전환 시 주의).
    - 모델 차원 == 1536 → 그대로 반환(정상 경로).
    패딩된 벡터는 코사인 유사도에서 방향 왜곡을 일으킬 수 있으므로,
    체크포인트는 실제 1536D 모델로 고정하는 것이 원칙이다(``embedding_constants.py`` 참조).
    """
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
    # 프로세스 내 최대 4개 체크포인트까지 캐싱 — 대부분 단일 모델 사용이므로 메모리 부담 없음.
    # 인덱싱/검색 시 동일 체크포인트를 써야 벡터 공간이 일치한다(embedding_constants.py 참조).
    return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    normalize_embeddings: bool = True,
) -> list[list[float]]:
    """문자열 배치를 그대로 ST 인코딩하는 저수준 seam(패딩 전·청크 분할 없음).

    인덱싱(skills)과 검색 쿼리(query_embed → OpenSearch kNN)가 함께 호출하는 지점 — 양측이 같은
    ``model_name``·``normalize_embeddings`` 를 줘야 벡터 공간이 일치한다.
    """
    if not texts:
        return []
    model = get_embedding_model(model_name)
    vectors = model.encode(
        list(texts),
        normalize_embeddings=normalize_embeddings,
    )
    return vectors.tolist()


def _iter_nonempty_chunks(
    path: Path,
    *,
    file_kind: str,
    encoding: str,
    chunk_size: int,
) -> Iterator[tuple[int, str]]:
    # idx 는 빈 청크를 건너뛴 뒤의 연속 번호(0,1,2,…) — 원본 문서상의 청크 위치가 아니다.
    # DB persist 가 chunk_index 로 청크를 식별하므로 빈틈 없는 0-based 순번을 보장한다.
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
