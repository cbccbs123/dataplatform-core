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

    ⚠️ **패딩은 최후 수단이다.** 짧은 벡터 뒤를 0 으로 채우면 저장은 되지만 코사인 유사도의
    방향이 왜곡된다 — 모델은 저장 차원과 **같은 차원**인 것으로 고정하는 것이 원칙이다
    (``embedding_constants.py`` 참조).

    Args:
        raw: 모델이 낸 벡터. **빈 리스트여도 된다** — 추출이 실패한 자산은 전부 0인 벡터가
            되어, 행은 남고 검색에서는 아무것과도 닮지 않는다.

    Returns:
        저장 차원에 맞춘 벡터.
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
    """임베딩 모델을 한 번만 로드해 재사용한다.

    Args:
        model_name: 모델 이름. **캐시 키이기도 하다** — 최대 4개까지 프로세스에 남는다.

    Returns:
        로드된 모델.

    ⚠️ 적재와 검색이 **같은 모델**을 써야 벡터 공간이 일치한다. 다른 모델로 만든 벡터끼리
    비교하면 값은 나오지만 의미가 없다.
    """
    return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    normalize_embeddings: bool = True,
) -> list[list[float]]:
    """문자열 배치를 그대로 ST 인코딩하는 저수준 seam(패딩 전·청크 분할 없음).

    ⚠️ **적재와 질의가 함께 부르는 자리**다. 두 쪽이 같은 모델·같은 정규화 설정을 줘야 벡터가
    같은 공간에 놓인다 — 어긋나면 검색이 조용히 엉뚱한 결과를 낸다.

    Args:
        texts: 인코딩할 문자열들. **빈 목록이면 모델을 부르지 않는다**.
        model_name: 쓸 모델.
        normalize_embeddings: 벡터 길이를 1로 맞출지. 적재와 질의가 **같아야** 한다.

    Returns:
        입력 순서대로의 벡터 목록(패딩 전·청크 분할 없음).
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

    **채널이 "어느 모델"과 "로컬이냐 원격이냐"를 함께 정한다.** 적재·질의·관계가 이 한 지점을
    공유하므로 세 경로가 자동으로 같은 공간을 쓴다.

    설정과 원격 임베더는 함수 **안에서** import 한다 — 모듈 상단에 두면 순환 import 가 생긴다.

    Args:
        texts: 인코딩할 문자열들.
        channel: 임베딩 채널. 원격 채널이면 API 로, 그 밖이면 로컬 모델로 보낸다.
        settings: 설정. ``None`` 이면 현재 활성 설정을 쓴다.
        normalize_embeddings: 벡터 길이를 1로 맞출지(적재·질의가 같아야 한다).

    Returns:
        입력 순서대로의 벡터 목록(패딩 전).
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
    """청크 전부를 **한 번의 배치**로 임베딩한다.

    청크마다 따로 부르면 원격 채널에서 HTTP 왕복이 청크 수만큼 발생한다. 한 번에 넘기면 하부
    구현이 알맞은 크기로 나눠 처리해 왕복이 크게 줄어든다.

    ⚠️ **메모리 주의**: 호출자가 문서 전체의 청크 텍스트를 목록으로 들고 있게 된다. 하부의
    배치 분할은 왕복만 줄이고 이 초기 목록은 줄이지 않으므로, 아주 큰 문서에서는 최대 메모리가
    올라간다(그때는 스트리밍 배치로 나눠야 한다).

    Args:
        texts: 임베딩할 청크 텍스트. **빈 목록이면 아무것도 부르지 않는다**.
        channel: 채널. 주면 백엔드 라우팅을 거치고, ``None`` 이면 로컬 모델로 바로 간다.
        settings: 설정.
        embedding_model_name: 채널이 없을 때 쓸 로컬 모델.
        normalize_embeddings: 벡터 길이를 1로 맞출지.

    Returns:
        입력 순서대로의 벡터 목록.
    """
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
    """파일을 청크로 잘라 **비어 있지 않은 것만** 번호와 함께 흘려보낸다.

    Args:
        path: 대상 파일.
        file_kind: 파일 종류(어떤 방식으로 읽을지 정한다).
        encoding: 텍스트 인코딩.
        chunk_size: 청크 하나의 최대 길이.
        overlap_size: 앞 청크 끝을 다음 청크 앞에 겹쳐 넣는 길이. 문장이 청크 경계에서
            잘려 의미가 끊기는 것을 줄인다. 0이면 겹치지 않는다.

    Yields:
        ``(번호, 청크 텍스트)``. **번호는 빈 청크를 건너뛴 뒤의 연속 순번**이라 원본 문서의
        위치가 아니다 — DB 가 이 번호로 청크를 식별하므로 빈틈이 없어야 한다.
    """
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
    """문서를 청크로 쪼개 임베딩한다 — **본문 텍스트는 DB 에 저장하지 않는다**(벡터만).

    Args:
        file_path: 대상 문서.
        file_kind: 파일 종류. 쪼개는 방식이 여기서 갈린다.
        encoding: 읽을 인코딩.
        chunk_size: 청크 크기. ⚠️ **모델 최대 입력의 2배를 넘으면 조용히 잘린다** — 로컬
            채널에서는 경고를 한 번 남긴다(원격은 모델 한계를 알 수 없어 생략).
        embedding_model_name: 채널이 없을 때 쓸 로컬 모델. 기본값은 **테스트 폴백**이며 운영
            경로는 항상 채널과 모델을 함께 주입한다.
        normalize_embeddings: 벡터 길이를 1로 맞출지.
        channel: 임베딩 채널. 주면 백엔드 라우팅을 거친다.
        settings: 설정. 청크 겹침 크기를 여기서 읽는다 — ``None`` 이면 겹치지 않는다.

    Returns:
        청크 순번과 벡터 목록. 순번은 **빈 청크를 걸러 낸 뒤의 연속 번호**다.

    Raises:
        FileNotFoundError: 파일이 없을 때.
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
    """문자열 하나를 청크로 쪼개 임베딩한다(전사 텍스트 등 — 파일이 아닌 입력).

    Args:
        text: 임베딩할 원문.
        chunk_size: 청크 크기.
        embedding_model_name: 채널이 없을 때 쓸 로컬 모델(기본값은 테스트 폴백).
        normalize_embeddings: 벡터 길이를 1로 맞출지.
        overlap_size: 인접 청크를 겹칠 글자 수. 겹치면 경계에서 잘린 문맥 손실이 줄어든다.
        max_input_chars: 읽을 최대 글자 수(아주 긴 입력의 상한).
        channel: 임베딩 채널.
        settings: 설정.

    Returns:
        청크 순번과 벡터 목록. **쓸 청크가 하나도 없으면 0번 자리에 전부 0인 벡터 한 개**를
        돌려준다 — 행이 아예 없으면 그 자산은 검색에서 통째로 사라진다.
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
