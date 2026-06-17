"""검색 질의 텍스트의 임베딩 — 적재·검색·OS 경로가 공유하는 중립 모듈.

질의를 인덱스(문서) 임베딩과 **같은 벡터 공간**으로 만들기 위한 단일 출처다. PG/OS 어느
검색 백엔드든, 또 관계 후보 경로든 이 함수를 거쳐 질의를 임베딩한다(임베딩 로직 복제 금지).

이 모듈은 임베더·전처리·설정(embedders/preprocess/settings)만 import 한다 — ``media_search``·
``search_service`` 를 import 하지 않는다(단방향 의존, import 순환 방지).
"""

from __future__ import annotations

from src.config.settings import get_current_settings
from src.embedders.text_embedder import embed_texts, pad_embedding_to_storage_dim
from src.preprocess.text_embedding_normalize import normalize_text_for_embedding


def embed_query_for_media_search(
    query: str, *, model_name: str | None = None
) -> list[float]:
    """질의 텍스트를 임베딩한다(017 A/B). ``model_name`` 미지정 시 기존대로 ``cfg.text_embedding_model``
    (KoSimCSE)을 쓴다 — 기본 경로 완전 동치. ``model_name`` 을 주면 그 모델로 질의를 임베딩해
    해당 채널의 문서 임베딩과 같은 벡터 공간에서 비교한다(FR-004 질의-문서 모델 일치)."""
    cfg = get_current_settings()
    mn = model_name if model_name is not None else cfg.text_embedding_model
    raw = query.strip() if query.strip() else " "
    q = normalize_text_for_embedding(raw)
    if not q.strip():
        q = " "
    row = embed_texts(
        [q],
        model_name=mn,
        normalize_embeddings=cfg.text_embedding_normalize,
    )[0]
    return pad_embedding_to_storage_dim(row)
