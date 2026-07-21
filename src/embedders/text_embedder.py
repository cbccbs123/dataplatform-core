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

import logging
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.embedders.text_embedding_normalize import normalize_text_for_embedding
from src.file.data_loader import (
    MAX_INPUT_CHARS,
    iter_document_chunks,
    iter_plain_text_chunks,
    normalize_file_kind,
)

_LOG = logging.getLogger(__name__)


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


def embed_texts_for(
    texts: Sequence[str],
    *,
    channel: str,
    settings: Any = None,
    normalize_embeddings: bool = True,
) -> list[list[float]]:
    """채널의 **백엔드**로 텍스트 배치를 임베딩한다(062·raw·패딩 없음 — ``embed_texts`` 대칭).

    ``backend_for_channel(channel)=='api'`` → 온프레미스 API(``embed_texts_api``·bge-m3 서빙), 그 외
    (``'st'``·``'st_bge'``) → 로컬 SentenceTransformer(``embed_texts``). 적재·질의·관계가 공유하는 단일
    라우팅 지점(018 채널 위에 얹는 직교 백엔드 축). **로컬 채널은 기존 ``embed_texts`` 그대로**라 동작 불변.
    순환 방지를 위해 settings/api 임베더는 함수 내부에서 지연 import 한다.
    """
    from src.config.settings import backend_for_channel, model_for_channel

    model = model_for_channel(channel, settings)
    if backend_for_channel(channel, settings) == "api":
        from src.config.settings import get_current_settings
        from src.embedders.text_embedder_api import embed_texts_api

        cfg = settings if settings is not None else get_current_settings()
        return embed_texts_api(
            texts,
            base_url=cfg.embed.api_base_url,
            model=model,
            api_key=(cfg.embed.api_key or None),
            timeout_s=cfg.embed.api_timeout_s,
            batch_size=cfg.embed.api_batch_size,
            max_retries=cfg.embed.api_max_retries,
            normalize_embeddings=normalize_embeddings,
        )
    return embed_texts(texts, model_name=model, normalize_embeddings=normalize_embeddings)


def _embed_many(
    texts: list[str],
    *,
    channel: str | None,
    settings: Any,
    embedding_model_name: str,
    normalize_embeddings: bool,
) -> list[list[float]]:
    """청크 여러 개를 **1회 배치**로 임베딩 — ``channel`` 있으면 백엔드 라우팅(``embed_texts_for``),
    없으면 로컬 모델(``embed_texts``). 입력 순서대로 벡터 리스트 반환.

    069 D7(P2-32): 과거 ``_embed_one`` 이 청크마다 1회(로컬 model.encode / API HTTP 요청 1건)를 돌던
    것을 전 청크 1회 배치로 대체한다. 하부 ``embed_texts``·``embed_texts_api`` 는 내부에서 batch_size 로
    나눠 처리하므로 API 요청 왕복이 청크수→⌈청크수/batch_size⌉ 로 급감한다. **수치 동일성 실측**(120
    실코퍼스 청크): 운영 API bge-m3 개별 vs 배치 bit-identical(차이 0), 로컬 bge cos이탈 8.8e-12 —
    검색 무영향·재백필 불필요(동작 불변). 빈 입력은 빈 리스트(호출 생략).

    메모리: 호출자가 문서 전체의 정제 청크 텍스트를 리스트로 모아 넘긴다 — 지배적 요인인 결과 벡터
    누적(1536D)은 기존과 동일하나 원문 청크 텍스트를 배치 호출 직전 동시에 들고 있다. embed_texts_api
    의 batch_size 분할은 **HTTP 왕복만** 줄일 뿐 이 초기 리스트 메모리는 완화하지 않는다(초대형 문서
    피크 메모리는 미측정 — 필요 시 후속에서 스트리밍 배치로 분할)."""
    if not texts:
        return []
    if channel is not None:
        return embed_texts_for(
            texts, channel=channel, settings=settings, normalize_embeddings=normalize_embeddings
        )
    return embed_texts(
        texts, model_name=embedding_model_name, normalize_embeddings=normalize_embeddings
    )


def _iter_nonempty_chunks(
    path: Path,
    *,
    file_kind: str,
    encoding: str,
    chunk_size: int,
    overlap_size: int = 0,
) -> Iterator[tuple[int, str]]:
    # idx 는 빈 청크를 건너뛴 뒤의 연속 번호(0,1,2,…) — 원본 문서상의 청크 위치가 아니다.
    # DB persist 가 chunk_index 로 청크를 식별하므로 빈틈 없는 0-based 순번을 보장한다.
    # 069 D8: overlap_size 기본 0 = 하드코딩과 동일(동작 불변). 호출자가 설정으로 조정 가능.
    kind = normalize_file_kind(file_kind)
    if kind is None:
        raise ValueError("file_kind는 필수입니다.")
    idx = 0
    for chunk in iter_document_chunks(
        path,
        file_kind=kind,
        encoding=encoding,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
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
    # 기본값은 테스트 전용 폴백 — 운영은 skills 가 active_embed_model+channel 을 항상 주입한다(D7·062).
    embedding_model_name: str = "BM-K/KoSimCSE-roberta-multitask",
    normalize_embeddings: bool = True,
    channel: str | None = None,
    settings: Any = None,
) -> list[TextChunkEmbedding]:
    """문서를 청크 단위로 임베딩한다. ``media_chunks`` 임베딩 적재용(본문 텍스트는 DB에 저장하지 않음).

    062: ``channel`` 이 주어지면 그 채널의 **백엔드**(로컬/API)로 임베딩한다(``embed_texts_for``). 미지정이면
    기존대로 로컬 ``embed_texts(embedding_model_name)`` — 기존 호출 완전 동치(회귀 0).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    # 069 D8: 청크 overlap 은 설정(embed.chunk_overlap) 단일 출처. settings 미전달(None)은 0
    # (하드코딩과 동일·동작 불변). >0 이면 인접 청크가 겹쳐 경계 문맥 손실을 줄인다(opt-in).
    overlap_size = int(settings.embed.chunk_overlap or 0) if settings is not None else 0
    # 069 D8: chunk_size 가 임베딩 모델 최대 시퀀스의 2배를 넘으면 인코딩 시 조용히 잘려(재현 불가)
    # 임베딩 품질이 저하될 수 있다 — **로컬 백엔드**에서 1회 관측 경고(동작은 불변·로그만). API
    # 백엔드(st_api)는 원격 모델의 max_seq 를 알 수 없어 생략. 판정 모델은 _embed_many 의 실제 경로를
    # 그대로 미러링한다: channel=None → 로컬 embed_texts(embedding_model_name), channel 있으면
    # backend_for_channel 로 api/local 분기 후 로컬이면 model_for_channel 의 모델. (운영 text_skill 은
    # channel=active('st') 를 넘기므로 과거 `channel is None` 조건은 운영에서 절대 발동 안 했음 — 069 리뷰)
    if channel is None:
        _is_api, _model_name = False, embedding_model_name
    else:
        from src.config.settings import backend_for_channel, model_for_channel
        _is_api = backend_for_channel(channel, settings) == "api"
        _model_name = None if _is_api else model_for_channel(channel, settings)
    if not _is_api:
        try:
            _max_seq = getattr(get_embedding_model(_model_name), "max_seq_length", None)
            if _max_seq and chunk_size > _max_seq * 2:
                _LOG.warning(
                    "chunk_size=%d 가 임베딩 모델 max_seq_length=%d 의 2배(%d)를 초과 — 청크가 "
                    "인코딩 시 잘려 임베딩 품질이 저하될 수 있음(TEXT_EMBED_CHUNK_SIZE 재검토 권장).",
                    chunk_size, _max_seq, _max_seq * 2,
                )
        except Exception:  # noqa: BLE001 — 경고는 관측용, 모델 조회 실패가 임베딩을 막으면 안 됨
            pass

    # 069 D7: 청크를 먼저 (chunk_index, clean) 로 모은 뒤 **1회 배치** 임베딩한다(과거 청크별 _embed_one
    # 대체). 정규화·빈청크 " " 치환·chunk_index 순번은 기존과 동일 — 임베딩 호출만 개별→배치.
    indexed: list[tuple[int, str]] = []
    for chunk_index, chunk in _iter_nonempty_chunks(
        path,
        file_kind=file_kind,
        encoding=encoding,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    ):
        clean = normalize_text_for_embedding(chunk)
        if not clean.strip():
            clean = " "
        indexed.append((chunk_index, clean))
    if not indexed:
        return [{"chunk_index": 0, "embedding_vector": pad_embedding_to_storage_dim([])}]
    vectors = _embed_many(
        [clean for _, clean in indexed],
        channel=channel,
        settings=settings,
        embedding_model_name=embedding_model_name,
        normalize_embeddings=normalize_embeddings,
    )
    return [
        {"chunk_index": idx, "embedding_vector": pad_embedding_to_storage_dim(vec)}
        for (idx, _clean), vec in zip(indexed, vectors, strict=True)  # 청크수==벡터수 방어
    ]


def embedding_plain_text_chunks(
    text: str,
    *,
    chunk_size: int,
    # 기본값은 테스트 전용 폴백 — 운영은 skills 가 active_embed_model+channel 을 항상 주입한다(D7·062).
    embedding_model_name: str = "BM-K/KoSimCSE-roberta-multitask",
    normalize_embeddings: bool = True,
    overlap_size: int = 0,
    max_input_chars: int = MAX_INPUT_CHARS,
    channel: str | None = None,
    settings: Any = None,
) -> list[TextChunkEmbedding]:
    """STT 등 단일 문자열을 청크 단위로 임베딩한다. ``media_chunks`` 임베딩 적재용(본문은 DB 미저장).

    062: ``channel`` 지정 시 그 채널 백엔드(로컬/API)로 임베딩(``embed_texts_for``). 미지정=기존 로컬(회귀 0).
    """
    # 069 D7: 파일 경로와 동일하게 청크를 모아 1회 배치 임베딩(과거 청크별 _embed_one 대체).
    # chunk_index 는 빈 청크를 건너뛴 뒤의 연속 순번(0,1,2,…) — 기존과 동일.
    indexed: list[tuple[int, str]] = []
    idx = 0
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
        indexed.append((idx, clean))
        idx += 1
    if not indexed:
        return [{"chunk_index": 0, "embedding_vector": pad_embedding_to_storage_dim([])}]
    vectors = _embed_many(
        [clean for _, clean in indexed],
        channel=channel,
        settings=settings,
        embedding_model_name=embedding_model_name,
        normalize_embeddings=normalize_embeddings,
    )
    return [
        {"chunk_index": i, "embedding_vector": pad_embedding_to_storage_dim(vec)}
        for (i, _clean), vec in zip(indexed, vectors, strict=True)  # 청크수==벡터수 방어
    ]
