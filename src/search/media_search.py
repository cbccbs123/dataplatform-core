"""``media_items`` / ``media_chunks`` 기반 텍스트(ST)·CLIP 검색."""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row

from src.config.embedding_constants import DEFAULT_CLIP_MODEL_NAME, FIX_EMBEDDING_DIMENSION
from src.config.settings import get_current_settings
from src.database.postgres_util import PostgresUtil
from src.embedders.image_embedder import embed_clip_text_query_for_image_search
from src.embedders.text_embedder import embed_texts, pad_embedding_to_storage_dim
from src.file.file_type_defs import (
    MEDIA_TYPES_CLIP_CHUNK_SEARCH,
    MEDIA_TYPES_ST_CHUNK_SEARCH,
)
from src.preprocess.text_embedding_normalize import normalize_text_for_embedding

EMBEDDING_KIND_ST = "st"
EMBEDDING_KIND_CLIP = "clip"


def embed_query_for_media_search(query: str) -> list[float]:
    cfg = get_current_settings()
    raw = query.strip() if query.strip() else " "
    q = normalize_text_for_embedding(raw)
    if not q.strip():
        q = " "
    row = embed_texts(
        [q],
        model_name=cfg.text_embedding_model,
        normalize_embeddings=cfg.text_embedding_normalize,
    )[0]
    return pad_embedding_to_storage_dim(row)


def search_media_text_items(query: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """
    pgvector 코사인 연산(`<=>`, 인덱스 `vector_cosine_ops`)으로 근접 순위를 매긴다.

    ``media_items`` 행은 **중복 없이** 한 줄씩이며, ``similarity``는 그 문서에 속한 청크들 중
    쿼리와 가장 가까운 청크 기준(``MAX(1 - 거리)``)이다.

    쿼리 벡터는 SentenceTransformer 기반이므로 CLIP 이미지 청크와 섞이지 않게
    ``MEDIA_TYPES_ST_CHUNK_SEARCH``(문서·오디오 STT·영상 키프레임 ST 등)만 조회한다.
    """
    db = PostgresUtil()
    query_vector = embed_query_for_media_search(query)
    vdim = FIX_EMBEDDING_DIMENSION
    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT
                        mi.id,
                        mi.file_uri,
                        mi.media_type,
                        MAX(1 - (mc.embedding <=> %s::vector({vdim}))) AS similarity
                    FROM media_items mi
                    JOIN media_chunks mc ON mi.id = mc.media_item_id
                    WHERE mc.embedding IS NOT NULL
                      AND mi.media_type = ANY(%s)
                      AND mc.embedding_kind = %s
                    GROUP BY mi.id, mi.file_uri, mi.media_type
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (
                        query_vector,
                        list(MEDIA_TYPES_ST_CHUNK_SEARCH),
                        EMBEDDING_KIND_ST,
                        limit,
                    ),
                )
                return list(cur.fetchall())


def search_media_images_by_text(
    query: str,
    *,
    limit: int = 20,
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME,
) -> list[dict[str, Any]]:
    """
    CLIP 텍스트 인코더 쿼리와 ``media_chunks``의 CLIP 이미지 임베딩을 코사인으로 비교한다.

    ``search_media_text_items``(SentenceTransformer)와 달리 ``media_type`` 은
    이미지·영상(키프레임 CLIP)만 대상이다.
    """
    db = PostgresUtil()
    query_vector = embed_clip_text_query_for_image_search(query, model_name=clip_model_name)
    vdim = FIX_EMBEDDING_DIMENSION
    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT
                        mi.id,
                        mi.file_uri,
                        mi.media_type,
                        MAX(1 - (mc.embedding <=> %s::vector({vdim}))) AS similarity
                    FROM media_items mi
                    JOIN media_chunks mc ON mi.id = mc.media_item_id
                    WHERE mc.embedding IS NOT NULL
                      AND mi.media_type = ANY(%s)
                      AND mc.embedding_kind = %s
                    GROUP BY mi.id, mi.file_uri, mi.media_type
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (
                        query_vector,
                        list(MEDIA_TYPES_CLIP_CHUNK_SEARCH),
                        EMBEDDING_KIND_CLIP,
                        limit,
                    ),
                )
                return list(cur.fetchall())


def search_media_images_two_stage(
    query_ko: str,
    query_en: str,
    *,
    stage1_limit: int = 80,
    final_limit: int = 20,
    alpha: float = 0.65,
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME,
) -> list[dict[str, Any]]:
    """
    1차: 이미지의 VLM 텍스트(ST) 청크로 넓게 후보를 고른 뒤,
    2차: 동일 후보에 CLIP 텍스트↔CLIP 이미지 벡터 점수를 섞어 재정렬한다.

    ``alpha``·``(1-alpha)`` 가중합으로 ``similarity``를 만든다. CLIP 전용 레거시 행만
    있으면 ``search_media_images_by_text``와 동일하게 동작한다.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha는 0~1 사이여야 합니다.")

    db = PostgresUtil()
    text_q = embed_query_for_media_search(query_ko)
    clip_q = embed_clip_text_query_for_image_search(query_en, model_name=clip_model_name)
    vdim = FIX_EMBEDDING_DIMENSION

    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT
                        mi.id,
                        mi.file_uri,
                        mi.media_type,
                        MAX(1 - (mc.embedding <=> %s::vector({vdim}))) AS s_text
                    FROM media_items mi
                    JOIN media_chunks mc ON mi.id = mc.media_item_id
                    WHERE mc.embedding IS NOT NULL
                      AND mi.media_type = ANY(%s)
                      AND mc.embedding_kind = %s
                    GROUP BY mi.id, mi.file_uri, mi.media_type
                    ORDER BY s_text DESC
                    LIMIT %s
                    """,
                    (
                        text_q,
                        list(MEDIA_TYPES_CLIP_CHUNK_SEARCH),
                        EMBEDDING_KIND_ST,
                        stage1_limit,
                    ),
                )
                stage1 = list(cur.fetchall())

    if not stage1:
        clip_only = search_media_images_by_text(
            query_en, limit=final_limit, clip_model_name=clip_model_name
        )
        return [
            {
                **row,
                "s_text": 0.0,
                "s_clip": float(row["similarity"]),
                "similarity": float(row["similarity"]),
            }
            for row in clip_only
        ]

    ids = [int(r["id"]) for r in stage1]
    clip_by_id: dict[int, float] = {}
    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT
                        mi.id,
                        MAX(1 - (mc.embedding <=> %s::vector({vdim}))) AS s_clip
                    FROM media_items mi
                    JOIN media_chunks mc ON mi.id = mc.media_item_id
                    WHERE mc.embedding IS NOT NULL
                      AND mi.media_type = ANY(%s)
                      AND mc.embedding_kind = %s
                      AND mi.id = ANY(%s)
                    GROUP BY mi.id
                    """,
                    (
                        clip_q,
                        list(MEDIA_TYPES_CLIP_CHUNK_SEARCH),
                        EMBEDDING_KIND_CLIP,
                        ids,
                    ),
                )
                for row in cur.fetchall():
                    clip_by_id[int(row["id"])] = float(row["s_clip"])

    merged: dict[int, dict[str, Any]] = {}
    for row in stage1:
        iid = int(row["id"])
        s_text = float(row["s_text"])
        s_clip = float(clip_by_id.get(iid, 0.0))
        sim = alpha * s_text + (1.0 - alpha) * s_clip
        merged[iid] = {
            "id": iid,
            "file_uri": row["file_uri"],
            "media_type": row["media_type"],
            "s_text": s_text,
            "s_clip": s_clip,
            "similarity": sim,
        }

    clip_extra = search_media_images_by_text(
        query_en, limit=max(stage1_limit, final_limit * 4), clip_model_name=clip_model_name
    )
    for row in clip_extra:
        iid = int(row["id"])
        if iid in merged:
            continue
        s_clip = float(row["similarity"])
        merged[iid] = {
            "id": iid,
            "file_uri": row["file_uri"],
            "media_type": row["media_type"],
            "s_text": 0.0,
            "s_clip": s_clip,
            "similarity": (1.0 - alpha) * s_clip,
        }

    ranked = sorted(merged.values(), key=lambda r: r["similarity"], reverse=True)
    return ranked[:final_limit]
