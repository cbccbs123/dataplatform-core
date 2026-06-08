"""포탈 자산 상세 조회 — 단일 자산의 메타·임베딩 채널 요약·관계 이웃(spec 010 D-3).

조회 구성(읽기 전용)
    1. ``asset`` + ``LEFT JOIN asset_metadata`` 1행 — modality/도메인/상태 + core/ext_meta/tags.
    2. ``asset_embedding`` 채널별 청크 **개수만** 집계(``COUNT(*) GROUP BY channel``).
       원시 벡터(VECTOR 1536)는 **반환하지 않는다**(FR-005·헌법 6조). 차원 노출 방지.
    3. 관계 이웃은 ``graph_query`` read seam(양방향·정규화, FR-006) 결과 그대로.

노출 게이트(FR-014)
    행 없음 / ``status != 'registered'`` (failed/deferred 등) / ``domain_label = 'medical'`` 이면
    ``None`` 을 반환한다(API 가 404 로 매핑). MVP 엔 medical 자산이 없어 단순 배제로 충분하다.

읽기 전용
    스키마·쓰기 경로 변경 0(헌법 6조). 결정성(헌법 3조): 채널 ORDER BY + graph_query 결정적 정렬.
"""
from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

# 관계 이웃은 반드시 graph_query seam 경유(대칭 엣지 양방향·status 필터). 직접 graph_edge 쿼리 금지.
from src.relations.graph_query import fetch_active_relations_for_asset

# asset + metadata 1행. LEFT JOIN 으로 메타데이터가 없어도 자산 행은 살린다(core/ext 는 NULL 가능).
_FETCH_ASSET_SQL = """
SELECT a.asset_id, a.modality, a.domain_label, a.status,
       m.core_meta, m.ext_meta, m.tags
FROM asset a
LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id
WHERE a.asset_id = %s
LIMIT 1
"""

# 임베딩 채널별 청크 개수만(원시 벡터 SELECT 금지, FR-005). ORDER BY channel 로 결정적.
_FETCH_EMBEDDING_CHANNELS_SQL = """
SELECT channel, COUNT(*) AS chunk_count
FROM asset_embedding
WHERE asset_id = %s
GROUP BY channel
ORDER BY channel
"""


def fetch_asset_detail(conn: Connection[Any], *, asset_id: str) -> dict[str, Any] | None:
    """``asset_id`` 의 상세(메타·임베딩 채널 요약·관계)를 조립해 반환한다.

    Returns:
        노출 가능한 자산이면 dict
        ``{"asset_id","modality","domain_label","status","core_meta","ext_meta","tags",
        "embedding_channels":[{"channel","chunk_count"}],"relations":[...]}``.
        행 없음/비registered/의료(FR-014)면 ``None``.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_FETCH_ASSET_SQL, (asset_id,))
        row = cur.fetchone()

    # 노출 게이트(FR-014): 행 없음 / 비registered / 의료 → 노출 안 함(API 404).
    if row is None:
        return None
    if row["status"] != "registered":
        return None
    if row["domain_label"] == "medical":
        return None

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_FETCH_EMBEDDING_CHANNELS_SQL, (asset_id,))
        channel_rows = cur.fetchall()
    embedding_channels = [
        {"channel": r["channel"], "chunk_count": int(r["chunk_count"])} for r in channel_rows
    ]

    relations = fetch_active_relations_for_asset(conn, asset_id=asset_id)

    return {
        "asset_id": str(row["asset_id"]),
        "modality": row["modality"],
        "domain_label": row["domain_label"],
        "status": row["status"],
        "core_meta": row["core_meta"],
        "ext_meta": row["ext_meta"],
        "tags": row["tags"],
        "embedding_channels": embedding_channels,
        "relations": relations,
    }
