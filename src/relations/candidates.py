"""임베딩 기반 **관계 후보 검색** (pgvector).

역할
    소스 ``media_item`` 의 ``media_chunks`` 임베딩과, **동일 ``embedding_kind``** 를 가진 다른 아이템들의
    청크 임베딩 사이 **코사인 유사도**(``1 - (a <=> b)``)를 계산해, 아이템당 최고 유사도로 집계한 뒤 상위 ``top_k`` 를 반환한다.

설정
    ``embedding_kind``: ``st`` / ``clip`` / ``both`` — ``src.search.media_search`` 의 상수와 매핑된다.
    차원은 ``FIX_EMBEDDING_DIMENSION`` 과 일치해야 한다(마이그레이션·청크 생성 파이프라인과 동일).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from psycopg import Connection
from psycopg.rows import dict_row

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.search.media_search import EMBEDDING_KIND_CLIP, EMBEDDING_KIND_ST

EmbeddingKindFilter = Literal["st", "clip", "both"]


class EmbeddingCandidate(TypedDict):
    """LLM 프롬프트에 실릴 후보 한 건(미디어 메타 + 임베딩 유사도)."""

    id: int
    file_uri: str
    media_type: str
    emb_score: float
    summary: str


def _kinds_param(kind: EmbeddingKindFilter) -> list[str]:
    """필터 문자열 → ``media_chunks.embedding_kind`` 에 넣을 값 목록."""
    if kind == "st":
        return [EMBEDDING_KIND_ST]
    if kind == "clip":
        return [EMBEDDING_KIND_CLIP]
    return [EMBEDDING_KIND_ST, EMBEDDING_KIND_CLIP]


def find_embedding_candidates(
    conn: Connection[Any],
    *,
    source_media_item_id: int,
    top_k: int,
    embedding_kind: EmbeddingKindFilter = "both",
) -> list[EmbeddingCandidate]:
    """
    소스와 **같은 embedding_kind** 청크끼리만 비교하여, 아이템별 최대 코사인 유사도로 정렬한 상위 ``top_k`` 후보.

    SQL 구조
        ``src_vecs``: 소스 아이템의 (kind, vector) 목록.
        ``cand``: 타 아이템 청크와 소스 벡터를 kind 로 조인한 (id, sim) 행.
        ``per_item``: 아이템 id 별 ``MAX(sim)`` — 한 아이템에 청크가 여러 개일 때 가장 가까운 쌍만 반영.
        최종: ``media_items`` 와 조인해 URI·타입·요약을 붙인다.
    """
    vdim = FIX_EMBEDDING_DIMENSION
    kinds = _kinds_param(embedding_kind)
    sql = f"""
        WITH src_vecs AS (
            SELECT embedding_kind, embedding
            FROM media_chunks
            WHERE media_item_id = %s
              AND embedding IS NOT NULL
              AND embedding_kind = ANY(%s)
        ),
        cand AS (
            SELECT mc.media_item_id AS id,
                   (1 - (mc.embedding <=> sv.embedding)) AS sim
            FROM media_chunks mc
            INNER JOIN src_vecs sv ON sv.embedding_kind = mc.embedding_kind
            WHERE mc.media_item_id <> %s
              AND mc.embedding IS NOT NULL
        ),
        per_item AS (
            SELECT id, MAX(sim) AS best_sim
            FROM cand
            GROUP BY id
        )
        SELECT mi.id,
               mi.file_uri,
               mi.media_type,
               p.best_sim::float8 AS emb_score,
               COALESCE(mi.metadata->>'summary', '') AS summary
        FROM per_item p
        INNER JOIN media_items mi ON mi.id = p.id
        ORDER BY p.best_sim DESC
        LIMIT %s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (source_media_item_id, kinds, source_media_item_id, top_k))
        rows = cur.fetchall()
    out: list[EmbeddingCandidate] = []
    for r in rows:
        out.append(
            {
                "id": int(r["id"]),
                "file_uri": str(r["file_uri"]),
                "media_type": str(r["media_type"]),
                "emb_score": float(r["emb_score"] or 0.0),
                "summary": str(r["summary"] or ""),
            }
        )
    return out
