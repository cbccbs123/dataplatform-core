"""검색 질의 텍스트의 임베딩 — 적재·검색·OS 경로가 공유하는 중립 모듈.

질의를 인덱스(문서) 임베딩과 **같은 벡터 공간**으로 만들기 위한 단일 출처다. PG/OS 어느
검색 백엔드든, 또 관계 후보 경로든 이 함수를 거쳐 질의를 임베딩한다(임베딩 로직 복제 금지).

이 모듈은 임베더·전처리·설정(embedders/preprocess/settings)만 import 한다 — ``media_search``·
``search_service`` 를 import 하지 않는다(단방향 의존, import 순환 방지).
"""

from __future__ import annotations

from src.config.settings import backend_for_channel, get_current_settings
from src.embedders.text_embedder import embed_texts, embed_texts_for, pad_embedding_to_storage_dim
from src.embedders.text_embedding_normalize import normalize_text_for_embedding


def embed_query_for_media_search(
    query: str, *, model_name: str | None = None, channel: str | None = None
) -> list[float]:
    """질의 텍스트를 임베딩한다(017 A/B·062). ``model_name`` 미지정 시 기존대로 ``cfg.embed.model``
    (KoSimCSE)을 쓴다 — 기본 경로 완전 동치. ``model_name`` 을 주면 그 모델로 질의를 임베딩해 해당 채널의
    문서 임베딩과 같은 벡터 공간에서 비교한다(FR-004 질의-문서 모델 일치).

    062: ``channel`` 을 주고 그 채널 백엔드가 API 면 ``embed_texts_for`` 로 API 임베딩한다(적재=질의 백엔드
    일치). 로컬 채널·``channel`` 미지정이면 기존 로컬 경로 그대로. 적재와 동일하게 raw→패딩.

    Args:
        query: 질의 텍스트. **빈 문자열이면 공백 한 칸으로 대체**한다 — 빈 입력은 임베더가
            0-노름 벡터를 내놓아 코사인 비교가 무의미해지기 때문이다.
        model_name: 쓸 임베딩 모델. ``None`` 이면 설정의 기본 모델.
        channel: 임베딩 채널. 이 채널의 백엔드가 API 면 원격 임베딩을 타고, ``None`` 이거나
            로컬이면 로컬 모델로 임베딩한다. **문서와 같은 채널을 줘야** 같은 공간에서 비교된다.

    Returns:
        저장 차원(1536D)까지 패딩된 임베딩 벡터.
    """
    cfg = get_current_settings()
    raw = query.strip() if query.strip() else " "
    q = normalize_text_for_embedding(raw)
    if not q.strip():
        q = " "
    if channel is not None and backend_for_channel(channel, cfg) == "api":
        row = embed_texts_for(
            [q], channel=channel, settings=cfg, normalize_embeddings=cfg.embed.normalize
        )[0]
    else:
        mn = model_name if model_name is not None else cfg.embed.model
        row = embed_texts([q], model_name=mn, normalize_embeddings=cfg.embed.normalize)[0]
    return pad_embedding_to_storage_dim(row)
