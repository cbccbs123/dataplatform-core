"""``asset_embedding`` 임베딩(pgvector)과 ``asset_metadata.search_vector`` FTS 하이브리드 검색.

자산별 임베딩 점수(채널·청크 코사인의 ``MAX``)와 ``ts_rank_cd`` 원시값을
``bm25 / (bm25 + k)`` 포화 정규화 후 가중합해 랭킹한다. 결과 행의 ``id`` 는 자산 UUID,
``modality`` 는 ``asset.modality``, 요약은 ``asset_metadata.ext_meta->>'summary'`` 다."""

from __future__ import annotations

import math
from typing import Any

from psycopg.rows import dict_row

from src.config.embedding_constants import DEFAULT_CLIP_MODEL_NAME, FIX_EMBEDDING_DIMENSION
from src.config.settings import get_current_settings
from src.database.postgres_util import PostgresUtil
from src.embedders.image_embedder import embed_clip_text_query_for_image_search
from src.embedders.text_embedder import embed_texts, pad_embedding_to_storage_dim
from src.file.file_type_defs import (
    ALLOWED_TEXT_META_FILE_KINDS,
    MEDIA_TYPES_CLIP_CHUNK_SEARCH,
    MEDIA_TYPES_ST_CHUNK_SEARCH,
    MediaKind,
)
from src.preprocess.media_item_search_text import MEDIA_ITEM_FTS_CONFIG
from src.preprocess.text_embedding_normalize import normalize_text_for_embedding
from src.search.fusion import RRF_DEFAULT_K, fuse_rrf
from src.search.query_preprocess import structure_user_query

EMBEDDING_KIND_ST = "st"
EMBEDDING_KIND_CLIP = "clip"
# FTS tsquery 는 사용자 질의를 ``to_tsvector`` 로 토큰화한 뒤 ``tok | tok | ...``(OR) 로 조립한다
# (원문을 to_tsquery 에 직접 넣으면 ``& | : ( )`` 등에서 구문 오류). 아래 토큰은 너무 일반적이라
# OR 매칭에 들어가면 과매칭을 유발하므로 tsquery 조립에서 제외한다(SQL 의 ``tok <> ALL(%s)``).
BM25_OR_EXCLUDE_TOKENS = (
    "과정",
    "방법",
    "관련",
    "정보",
    "자료",
    "설명",
)
# BM25 원시값 영향력을 완만하게 만드는 포화 정규화 상수.
# bm25_scaled = bm25 / (bm25 + k), bm25==k일 때 0.5.
BM25_SATURATION_K = 0.2

_TWO_STAGE_BM25_FOR_IDS_SQL = """
SELECT
    am.asset_id AS id,
    ts_rank_cd(
        coalesce(am.search_vector, ''::tsvector),
        coalesce(
            to_tsquery(
                %s,
                coalesce(
                    (
                        SELECT string_agg(tok, ' | ')
                        FROM (
                            SELECT DISTINCT tok
                            FROM unnest(
                                tsvector_to_array(to_tsvector(%s, %s))
                            ) AS tok
                            WHERE length(tok) >= 2
                              AND tok !~ '^[0-9]+$'
                              AND tok <> ALL(%s)
                        ) t
                    ),
                    ''
                )
            ),
            ''::tsquery
        )
    ) AS bm25_score
FROM asset_metadata am
WHERE am.asset_id = ANY(%s)
"""

# 자산 UUID 목록 → 요약(asset_metadata.ext_meta->>'summary') 조회.
_SUMMARIES_FOR_ASSET_IDS_SQL = """
SELECT am.asset_id AS id, COALESCE(am.ext_meta->>'summary', '') AS summary
FROM asset_metadata am
WHERE am.asset_id = ANY(%s)
"""


def _hybrid_embedding_bm25_sql(vdim: int) -> str:
    """단일 트랜잭션에서 임베딩 후보 + FTS를 섞는 하이브리드 SQL."""
    return f"""
                    WITH emb_chunks AS (
                        SELECT
                            ae.asset_id AS id,
                            (1 - (ae.embedding <=> %s::vector({vdim}))) AS chunk_sim
                        FROM asset_embedding ae
                        WHERE ae.embedding IS NOT NULL
                          AND ae.channel = %s
                    ),
                    emb AS (
                        SELECT
                            id,
                            MAX(CASE WHEN chunk_sim = chunk_sim THEN chunk_sim END) AS emb_score
                        FROM emb_chunks
                        GROUP BY id
                    ),
                    joined_inner AS (
                        SELECT
                            e.id,
                            a.fs_path AS file_uri,
                            a.modality,
                            e.emb_score,
                            COALESCE(am.ext_meta->>'summary', '') AS summary,
                            COALESCE(
                                ts_rank_cd(
                                    coalesce(am.search_vector, ''::tsvector),
                                    coalesce(
                                        to_tsquery(
                                            %s,
                                            coalesce(
                                                (
                                                    SELECT string_agg(tok, ' | ')
                                                    FROM (
                                                        SELECT DISTINCT tok
                                                        FROM unnest(
                                                            tsvector_to_array(to_tsvector(%s, %s))
                                                        ) AS tok
                                                        WHERE length(tok) >= 2
                                                          AND tok !~ '^[0-9]+$'
                                                          AND tok <> ALL(%s)
                                                    ) t
                                                ),
                                                ''
                                            )
                                        ),
                                        ''::tsquery
                                    )
                                ),
                                0.0::double precision
                            ) AS bm25_raw
                        FROM emb e
                        JOIN asset a ON a.asset_id = e.id
                        LEFT JOIN asset_metadata am ON am.asset_id = a.asset_id
                        WHERE a.modality = ANY(%s)
                    ),
                    joined AS (
                        SELECT
                            ji.id,
                            ji.file_uri,
                            ji.modality,
                            ji.emb_score,
                            ji.summary,
                            CASE
                                WHEN ji.bm25_raw = ji.bm25_raw THEN ji.bm25_raw
                                ELSE 0.0::double precision
                            END AS bm25_score
                        FROM joined_inner ji
                    ),
                    scored_base AS (
                        SELECT
                            j.id,
                            j.file_uri,
                            j.modality,
                            j.emb_score,
                            j.bm25_score,
                            (COALESCE(j.bm25_score, 0.0::double precision)
                                / (COALESCE(j.bm25_score, 0.0::double precision) + %s::double precision)
                            ) AS bm25_scaled,
                            j.summary,
                            COUNT(*) OVER () AS candidate_count
                        FROM joined j
                    ),
                    scored AS (
                        SELECT
                            sb.id,
                            sb.file_uri,
                            sb.modality,
                            sb.emb_score,
                            sb.bm25_score,
                            sb.bm25_scaled,
                            sb.summary,
                            sb.candidate_count,
                            COALESCE(
                                %s::double precision * COALESCE(sb.emb_score, 0.0::double precision)
                                + (1::double precision - %s::double precision)
                                * COALESCE(sb.bm25_scaled, 0.0::double precision),
                                0.0::double precision
                            ) AS similarity
                        FROM scored_base sb
                    )
                    SELECT
                        id,
                        file_uri,
                        modality,
                        emb_score,
                        bm25_score,
                        bm25_scaled,
                        summary,
                        candidate_count,
                        similarity
                    FROM scored
                    ORDER BY similarity DESC NULLS LAST, id ASC
                    LIMIT %s
                    """


def _two_stage_stage1_sql(vdim: int) -> str:
    return f"""
                    SELECT
                        a.asset_id AS id,
                        a.fs_path AS file_uri,
                        a.modality,
                        MAX(1 - (ae.embedding <=> %s::vector({vdim}))) AS s_text
                    FROM asset a
                    JOIN asset_embedding ae ON a.asset_id = ae.asset_id
                    WHERE ae.embedding IS NOT NULL
                      AND a.modality = ANY(%s)
                      AND ae.channel = %s
                    GROUP BY a.asset_id, a.fs_path, a.modality
                    ORDER BY s_text DESC, id ASC
                    LIMIT %s
                    """


def _two_stage_clip_for_ids_sql(vdim: int) -> str:
    return f"""
                    SELECT
                        a.asset_id AS id,
                        MAX(1 - (ae.embedding <=> %s::vector({vdim}))) AS s_clip
                    FROM asset a
                    JOIN asset_embedding ae ON a.asset_id = ae.asset_id
                    WHERE ae.embedding IS NOT NULL
                      AND a.modality = ANY(%s)
                      AND ae.channel = %s
                      AND a.asset_id = ANY(%s)
                    GROUP BY a.asset_id
                    """


def _finite_float(value: object, default: float = 0.0) -> float:
    """JSON/드라이버에서 오는 NaN·inf·NULL을 안전한 유한 실수로 만든다."""
    if value is None:
        return default
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _saturating_bm25(raw_bm25: object, *, k: float = BM25_SATURATION_K) -> float:
    x = max(_finite_float(raw_bm25, 0.0), 0.0)
    kk = max(_finite_float(k, BM25_SATURATION_K), 1e-9)
    return x / (x + kk)


def _sanitize_hybrid_search_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for r in rows:
        # psycopg 가 asset_id 를 uuid.UUID 로 반환하므로 결과 계약(시각 2단계 경로와 동일)상
        # 자산 id 를 항상 str(UUID) 로 통일한다(JSON 직렬화·키 비교 일관).
        if r.get("id") is not None:
            r["id"] = str(r["id"])
        if "similarity" in r:
            r["similarity"] = _finite_float(r["similarity"], 0.0)
        if "emb_score" in r:
            r["emb_score"] = _finite_float(r["emb_score"], 0.0)
        if "bm25_score" in r:
            r["bm25_score"] = _finite_float(r["bm25_score"], 0.0)
        if "bm25_scaled" in r:
            r["bm25_scaled"] = _finite_float(r["bm25_scaled"], 0.0)
        if "candidate_count" in r and r["candidate_count"] is not None:
            try:
                r["candidate_count"] = int(r["candidate_count"])
            except (TypeError, ValueError):
                pass
    return rows


def _merge_clip_only_candidates(
    merged: dict[str, dict[str, Any]], clip_extra: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """1차(ST) 후보 ``merged`` 에 CLIP-단독 후보를 합친다.

    이미 ST 후보로 들어온 자산 id 는 건너뛰어(중복 제거) ST 점수를 보존하고, 1차에 없던
    CLIP-단독 후보만 ``s_text=0`` 으로 추가한다(같은 영상이 ST·CLIP 양쪽에 떠도 결과 1건). (#7)
    """
    for row in clip_extra:
        iid = str(row["id"])
        if iid in merged:
            continue
        merged[iid] = {
            "id": iid,
            "file_uri": row["file_uri"],
            "modality": row["modality"],
            "s_text": 0.0,
            "s_clip": _finite_float(row["similarity"], 0.0),
        }
    return merged


def _dedupe_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """자산 id 당 similarity 가 가장 높은 행 1건만 남긴다.

    video 버킷처럼 ST 하이브리드 결과와 시각 2단계 결과를 합칠 때 동일 영상이 양쪽에서
    나오면 중복으로 보이므로, 더 높은 점수의 행만 보존한다(낮은 쪽 source 는 버린다).
    """
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        iid = str(r.get("id"))
        cur = best.get(iid)
        if cur is None or _finite_float(r.get("similarity"), 0.0) > _finite_float(
            cur.get("similarity"), 0.0
        ):
            best[iid] = r
    return list(best.values())


def _sort_by_similarity_cap(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    # 결정 재현성(헌법 3조): similarity 동점 시 자산 id 오름차순으로 tiebreak 해
    # 입력 순서에 의존하지 않는 고정된 순서를 보장한다.
    return sorted(
        rows,
        key=lambda x: (-_finite_float(x.get("similarity"), 0.0), str(x.get("id", ""))),
    )[:n]


def _apply_fusion(
    rows: list[dict[str, Any]], *, fusion: str, k: int
) -> list[dict[str, Any]]:
    """행을 융합 방식에 따라 재정렬한다(순수, DB·LLM 무관).

    - ``alpha``: 입력 순서를 그대로 유지한다(이미 SQL이 similarity DESC, id ASC로 정렬).
      프로덕션 기본값이라 이 경로는 동작 불변이 핵심 — 행을 만지지 않고 그대로 돌려준다.
    - ``rrf``: emb_score·bm25_score 각각의 독립 랭킹을 RRF로 합쳐 재정렬한다.
      동점·결측은 ``fuse_rrf`` 의 결정적 규칙(점수 DESC, id ASC)을 따른다(헌법 3조).
      결측 점수는 기존 관용대로 ``_finite_float`` 로 0.0 처리한다(NaN/inf/None 방어).
    컷오프는 호출부에서 원 코사인(emb_score) 기준으로 별도 적용한다.
    """
    if fusion == "alpha":
        return rows
    if fusion != "rrf":
        raise ValueError(f"알 수 없는 fusion: {fusion!r} (alpha|rrf)")
    by_id = {str(r["id"]): r for r in rows}
    emb_ranked = [
        str(r["id"])
        for r in sorted(
            rows, key=lambda r: (-_finite_float(r.get("emb_score"), 0.0), str(r["id"]))
        )
    ]
    bm25_ranked = [
        str(r["id"])
        for r in sorted(
            rows, key=lambda r: (-_finite_float(r.get("bm25_score"), 0.0), str(r["id"]))
        )
    ]
    fused = fuse_rrf([emb_ranked, bm25_ranked], k=k)
    return [by_id[i] for i, _ in fused if i in by_id]


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


def _run_hybrid_search(
    *,
    query_vector: list[float],
    bm25_query: str,
    media_types: list[str],
    embedding_kind: str,
    limit: int,
    alpha: float,
    fusion: str = "alpha",
    rrf_k: int = RRF_DEFAULT_K,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """임베딩 상위 후보에 ``asset_metadata.search_vector`` 기준 ``ts_rank_cd`` 를 섞어 반환한다.

    ``similarity = alpha * emb_score + (1-alpha) * bm25_scaled``.
    여기서 ``bm25_scaled = bm25_score / (bm25_score + BM25_SATURATION_K)``.

    ``fusion`` 은 최종 행 정렬 방식이다(기본 ``alpha``=무변경, 프로덕션 동작 불변).
    ``rrf`` 이면 SQL 의 가중합 순서를 버리고 emb·bm25 독립 랭킹을 RRF 로 합쳐 재정렬한다
    (``_apply_fusion``). alpha 경로는 SQL 정렬(similarity DESC, id ASC)을 그대로 둔다.

    ``debug=True`` 이면 ``candidate_count``(임베딩 조건 통과 후보 전체 행 수)를 유지하고,
    ``debug=False`` 이면 응답에서 제거한다.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha는 0~1 사이여야 합니다.")
    db = PostgresUtil()
    vdim = FIX_EMBEDDING_DIMENSION
    sql = _hybrid_embedding_bm25_sql(vdim)
    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    sql,
                    (
                        query_vector,                         # 1
                        embedding_kind,                       # 2
                        MEDIA_ITEM_FTS_CONFIG,                # 3
                        MEDIA_ITEM_FTS_CONFIG,                # 4
                        bm25_query.strip() if bm25_query.strip() else " ",  # 5
                        list(BM25_OR_EXCLUDE_TOKENS),         # 6
                        media_types,                          # 7
                        BM25_SATURATION_K,                    # 8
                        alpha,                                # 9
                        alpha,                                # 10
                        limit,                                # 11
                    ),
                )
                rows = _sanitize_hybrid_search_rows(list(cur.fetchall()))
                # 기본 alpha 면 입력 행을 그대로 반환(동작 불변), rrf 면 순위 융합 재정렬.
                rows = _apply_fusion(rows, fusion=fusion, k=rrf_k)
                if not debug:
                    for r in rows:
                        r.pop("bm25_scaled", None)
                        r.pop("candidate_count", None)
                return rows


def search_media_text_items(
    query: str,
    *,
    limit: int = 20,
    alpha: float = 0.75,
    media_types: list[str] | None = None,
    fusion: str = "alpha",
    debug: bool = False,
) -> list[dict[str, Any]]:
    """
    SentenceTransformer 청크 임베딩(``asset_embedding.channel='st'``)과 ``asset_metadata.search_vector`` FTS를
    한 쿼리에서 하이브리드한다. 자산당 청크 유사도는 ``MAX(1 - (ae.embedding <=> 쿼리))`` 이고,
    ``similarity = alpha * emb_score + (1-alpha) * bm25_scaled``.
    ``bm25_scaled`` 는 ``bm25 / (bm25 + BM25_SATURATION_K)``.

    ``debug=True`` 이면 같은 임베딩 후보 정의 안의 전체 행 수 ``candidate_count`` 를 같이 돌려준다.

    기본 ``media_types`` 는 ``MEDIA_TYPES_ST_CHUNK_SEARCH``(문서·오디오 STT·영상 VLM 텍스트 청크 등)로
    CLIP 청크와 섞이지 않게 한다. 결과에는 ``summary`` 가 포함된다.
    """
    query_vector = embed_query_for_media_search(query)
    mt = media_types if media_types is not None else list(MEDIA_TYPES_ST_CHUNK_SEARCH)
    return _run_hybrid_search(
        query_vector=query_vector,
        bm25_query=query,
        media_types=mt,
        embedding_kind=EMBEDDING_KIND_ST,
        limit=limit,
        alpha=alpha,
        fusion=fusion,
        debug=debug,
    )


def search_media_images_by_text(
    query: str,
    *,
    limit: int = 20,
    alpha: float = 0.85,
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """
    CLIP 텍스트 쿼리와 ``asset_embedding``(``channel='clip'``)의 CLIP 이미지·키프레임 임베딩을
    코사인으로 비교하고, 동일 후보에 ``asset_metadata.search_vector`` FTS(``ts_rank_cd``)를 원시 가중합으로 섞는다.

    ``modality`` 는 ``MEDIA_TYPES_CLIP_CHUNK_SEARCH``(이미지·영상)만 대상이다.
    """
    query_vector = embed_clip_text_query_for_image_search(query, model_name=clip_model_name)
    return _run_hybrid_search(
        query_vector=query_vector,
        bm25_query=query,
        media_types=list(MEDIA_TYPES_CLIP_CHUNK_SEARCH),
        embedding_kind=EMBEDDING_KIND_CLIP,
        limit=limit,
        alpha=alpha,
        debug=debug,
    )


def _two_stage_load_bm25_for_ids(
    conn: Any,
    query_ko: str,
    ids: list[str],
) -> dict[str, float]:
    if not ids:
        return {}
    bm25_by_id: dict[str, float] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            _TWO_STAGE_BM25_FOR_IDS_SQL,
            (
                MEDIA_ITEM_FTS_CONFIG,
                MEDIA_ITEM_FTS_CONFIG,
                query_ko.strip() if query_ko.strip() else " ",
                list(BM25_OR_EXCLUDE_TOKENS),
                ids,
            ),
        )
        for row in cur.fetchall():
            bm25_by_id[str(row["id"])] = _finite_float(row["bm25_score"], 0.0)
    return bm25_by_id


def _summaries_for_media_item_ids(ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    db = PostgresUtil()
    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(_SUMMARIES_FOR_ASSET_IDS_SQL, (ids,))
                return {str(r["id"]): str(r["summary"] or "") for r in cur.fetchall()}


def search_media_images_two_stage(
    query_ko: str,
    query_en: str,
    *,
    stage1_limit: int = 80,
    final_limit: int = 20,
    alpha: float = 0.65,
    bm25_weight: float = 0.2,
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME,
) -> list[dict[str, Any]]:
    """
    1차: 이미지·영상의 SentenceTransformer(VLM 텍스트) 청크로 후보 id 를 고른다.
    2차: 각 id 에 CLIP 텍스트↔CLIP 이미지 최대 유사도와 ``query_ko`` 기준 ``search_vector`` FTS 점수를 붙인다.
        베이스 점수 ``alpha * s_text + (1-alpha) * s_clip`` 과 ``bm25_score`` 를
        원시 가중합 ``(1 - bm25_weight) * base + bm25_weight * bm25_score`` 로 합산해 랭킹한다.

    1차 후보에 없지만 CLIP만 강한 행은 ``search_media_images_by_text(..., alpha=1.0)`` 로 넓게 가져와
    합친 뒤, 병합된 전체 후보에 대해 FTS 점수를 다시 채워 적용한다.
    ST 후보가 하나도 없으면 ``search_media_images_by_text`` (임베딩+FTS 하이브리드)로 폴백한다.

    결과 행에는 ``asset_metadata.ext_meta->>'summary'`` 가 ``summary`` 키로 포함된다.
    """
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= bm25_weight <= 1.0:
        raise ValueError("alpha와 bm25_weight는 0~1 사이여야 합니다.")

    db = PostgresUtil()
    text_q = embed_query_for_media_search(query_ko)
    clip_q = embed_clip_text_query_for_image_search(query_en, model_name=clip_model_name)
    vdim = FIX_EMBEDDING_DIMENSION
    stage1_sql = _two_stage_stage1_sql(vdim)
    clip_sql = _two_stage_clip_for_ids_sql(vdim)

    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    stage1_sql,
                    (
                        text_q,
                        list(MEDIA_TYPES_CLIP_CHUNK_SEARCH),
                        EMBEDDING_KIND_ST,
                        stage1_limit,
                    ),
                )
                stage1 = list(cur.fetchall())

    if not stage1:
        return search_media_images_by_text(
            query_en,
            limit=final_limit,
            alpha=1.0 - bm25_weight,
            clip_model_name=clip_model_name,
        )

    ids = [str(r["id"]) for r in stage1]
    clip_by_id: dict[str, float] = {}
    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    clip_sql,
                    (
                        clip_q,
                        list(MEDIA_TYPES_CLIP_CHUNK_SEARCH),
                        EMBEDDING_KIND_CLIP,
                        ids,
                    ),
                )
                for row in cur.fetchall():
                    clip_by_id[str(row["id"])] = _finite_float(row["s_clip"], 0.0)

    merged: dict[str, dict[str, Any]] = {}
    for row in stage1:
        iid = str(row["id"])
        s_text = _finite_float(row["s_text"], 0.0)
        s_clip = _finite_float(clip_by_id.get(iid, 0.0), 0.0)
        merged[iid] = {
            "id": iid,
            "file_uri": row["file_uri"],
            "modality": row["modality"],
            "s_text": s_text,
            "s_clip": s_clip,
        }

    clip_extra = search_media_images_by_text(
        query_en,
        limit=max(stage1_limit, final_limit * 4),
        alpha=1.0,
        clip_model_name=clip_model_name,
    )
    _merge_clip_only_candidates(merged, clip_extra)

    all_ids = list(merged.keys())
    bm25_full: dict[str, float] = {}
    if all_ids:
        with db:
            with db.transaction() as conn:
                bm25_full = _two_stage_load_bm25_for_ids(conn, query_ko, all_ids)

    for iid, row in merged.items():
        row["bm25_score"] = _finite_float(bm25_full.get(iid, 0.0), 0.0)
        row["bm25_scaled"] = _saturating_bm25(row["bm25_score"])

    for row in merged.values():
        base = (
            alpha * _finite_float(row["s_text"], 0.0)
            + (1.0 - alpha) * _finite_float(row["s_clip"], 0.0)
        )
        row["base_similarity"] = base
        row["similarity"] = _finite_float(
            (1.0 - bm25_weight) * base + bm25_weight * _finite_float(row["bm25_scaled"], 0.0),
            0.0,
        )

    ranked = sorted(
        merged.values(),
        key=lambda r: (-_finite_float(r.get("similarity"), 0.0), str(r.get("id", ""))),
    )
    top = ranked[:final_limit]
    sid = [str(r["id"]) for r in top]
    summaries = _summaries_for_media_item_ids(sid)
    for r in top:
        r["summary"] = summaries.get(str(r["id"]), "")
    return top


def search_media_all_grouped(
    query: str,
    *,
    structured: dict[str, Any] | None = None,
    limit_per_bucket: int = 20,
    text_hybrid_alpha: float = 0.75,
    image_search_alpha: float = 0.65,
    bm25_weight: float = 0.2,
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME,
    fusion: str = "alpha",
    debug: bool = False,
) -> dict[str, Any]:
    """ST 하이브리드(문서·음성·영상 텍스트)와 시각 2단계(이미지·영상)를 한 번에 돌리고
    ``modality`` 기준으로 나눈 결과를 반환한다.

    ``video`` 버킷에는 ST 하이브리드 영상과 시각 검색 영상을 ``similarity`` 기준으로 합친 뒤
    상위 ``limit_per_bucket`` 건만 담는다. 각 row에 ``source`` 가 ``st_hybrid`` 또는
    ``visual_two_stage`` 로 붙는다.

    ``fusion`` 은 ST 하이브리드 경로의 emb·bm25 융합 방식이다(기본 ``alpha``=동작 불변).
    ``rrf`` 는 ST 하이브리드(텍스트/오디오/영상 텍스트) 후보 재정렬에만 적용하고, 시각 2단계
    (이미지·영상)는 별도 가중합 경로라 영향받지 않는다(프로토타입 스코프, 설계 §5).
    """
    if structured is None:
        structured = structure_user_query(query)
    st_q = (structured.get("semantic_query") or query).strip() or query
    en_q = (structured.get("semantic_query_en") or "").strip()

    st_fetch_limit = max(limit_per_bucket * 12, 80)
    st_rows = search_media_text_items(
        st_q,
        limit=st_fetch_limit,
        alpha=text_hybrid_alpha,
        media_types=list(MEDIA_TYPES_ST_CHUNK_SEARCH),
        fusion=fusion,
        debug=debug,
    )

    text_documents: list[dict[str, Any]] = []
    audio_rows: list[dict[str, Any]] = []
    video_st: list[dict[str, Any]] = []
    for r in st_rows:
        mt = r.get("modality")
        if mt is None:
            continue
        row = {**r, "source": "st_hybrid"}
        if mt in ALLOWED_TEXT_META_FILE_KINDS:
            text_documents.append(row)
        elif mt == MediaKind.AUDIO.value:
            audio_rows.append(row)
        elif mt == MediaKind.VIDEO.value:
            video_st.append(row)

    text_documents = _sort_by_similarity_cap(text_documents, limit_per_bucket)
    audio_rows = _sort_by_similarity_cap(audio_rows, limit_per_bucket)

    visual_final_limit = max(limit_per_bucket * 4, 40)
    visual_rows = search_media_images_two_stage(
        st_q,
        en_q or st_q,
        final_limit=visual_final_limit,
        alpha=image_search_alpha,
        bm25_weight=bm25_weight,
        clip_model_name=clip_model_name,
    )

    image_rows: list[dict[str, Any]] = []
    video_vis: list[dict[str, Any]] = []
    for r in visual_rows:
        mt = r.get("modality")
        row = {**r, "source": "visual_two_stage"}
        if mt == MediaKind.IMAGE.value:
            image_rows.append(row)
        elif mt == MediaKind.VIDEO.value:
            video_vis.append(row)

    image_rows = _sort_by_similarity_cap(image_rows, limit_per_bucket)
    # ST 하이브리드 영상 + 시각 2단계 영상을 합치되, 동일 영상이 양쪽에 나오면 id 기준 dedup.
    video_merged = _sort_by_similarity_cap(_dedupe_by_id(video_st + video_vis), limit_per_bucket)

    return {
        "text_documents": text_documents[:limit_per_bucket],
        "audio": audio_rows[:limit_per_bucket],
        "image": image_rows[:limit_per_bucket],
        "video": video_merged[:limit_per_bucket],
        "meta": {"structured": structured},
    }
