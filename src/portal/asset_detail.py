"""포탈 자산 상세 조회 — 단일 자산의 메타·임베딩 채널 요약·관계 이웃 (spec 010 D-3 · 042).

조회 구성(읽기 전용)
    1. ``asset`` + ``LEFT JOIN asset_metadata`` 1행 — modality/도메인/상태 + core/ext_meta/tags.
    2. ``asset_embedding`` 채널별 청크 **개수만** 집계 — 원시 벡터(1536D) 미반환(FR-005).
    3. 관계 이웃 — ``graph_query`` read seam(양방향·active, FR-006).

노출 게이트(FR-014)
    행 없음 / ``status != 'registered'`` / ``domain_label = 'medical'`` → ``None`` (API 404).

ext_meta read 집행(042 · 040 tier · 041 레지스트리)
    ``clearance`` 지정 시 ``fetch_access_tiers`` + ``project_ext_meta`` — tier 미달 키 **제거(omit)**.
    null·마스킹 문자열 치환 없음(plan D2). DB 원본은 전량 유지(ingest 039).
"""
from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.registry.access_tier import project_ext_meta
from src.registry.ext_meta_field_registry import fetch_access_tiers
from src.relations.graph_query import fetch_active_relations_for_asset

# asset + metadata 1행. LEFT JOIN — 메타 없어도 자산 행 유지(core/ext NULL 가능).
_FETCH_ASSET_SQL = """
SELECT a.asset_id, a.modality, a.domain_label, a.status,
       m.core_meta, m.ext_meta, m.tags
FROM asset a
LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id
WHERE a.asset_id = %s
LIMIT 1
"""

# 임베딩 채널별 청크 개수만(FR-005). ORDER BY channel — 결정적.
_FETCH_EMBEDDING_CHANNELS_SQL = """
SELECT channel, COUNT(*) AS chunk_count
FROM asset_embedding
WHERE asset_id = %s
GROUP BY channel
ORDER BY channel
"""


def fetch_asset_detail(
    conn: Connection[Any],
    *,
    asset_id: str,
    clearance: str | None = None,
) -> dict[str, Any] | None:
    """자산 상세 조립. ``clearance`` 지정 시 ext_meta tier 미달 키 omit(042)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_FETCH_ASSET_SQL, (asset_id,))
        row = cur.fetchone()

    # 노출 게이트(FR-014)
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

    ext_meta = row["ext_meta"]
    if clearance is not None:
        # 포탈 API(042) — principal.clearance 로 read projection. None 이면 DB 원본 그대로(내부·테스트).
        domain = str(row["domain_label"])
        tiers = fetch_access_tiers(conn, domain)  # 040/041 레지스트리
        ext_meta = project_ext_meta(
            ext_meta if isinstance(ext_meta, dict) else {},
            tiers,
            domain=domain,
            clearance=clearance,
        )

    return {
        "asset_id": str(row["asset_id"]),
        "modality": row["modality"],
        "domain_label": row["domain_label"],
        "status": row["status"],
        "core_meta": row["core_meta"],
        "ext_meta": ext_meta,
        "tags": row["tags"],
        "embedding_channels": embedding_channels,
        "relations": relations,
    }
