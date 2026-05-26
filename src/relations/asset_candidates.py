"""임베딩 기반 **관계 후보 검색** (pgvector) — asset_* 재배선판.

역할
    소스 ``asset`` 의 ``asset_embedding`` 벡터와, **동일 채널(channel)** 을 가진 다른 자산들의
    임베딩 사이 **코사인 유사도**(``1 - (a <=> b)``)를 계산해, 자산당 최고 유사도로 집계한 뒤 상위 ``top_k`` 를 반환한다.

설정
    ``embedding_kind``: ``st`` / ``clip`` / ``both`` — ``asset_embedding.channel`` 값과 매핑된다.
    차원은 ``FIX_EMBEDDING_DIMENSION`` 과 일치해야 한다(추출·적재 파이프라인과 동일).

OLD ``candidates.py`` 와의 차이
    media_chunks→asset_embedding, media_items→asset(+asset_metadata), id 가 UUID(str) 다.
    요약은 ``asset_metadata.ext_meta->>'summary'`` 에서 읽는다.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from psycopg import Connection
from psycopg.rows import dict_row

from src.search.media_search import EMBEDDING_KIND_CLIP, EMBEDDING_KIND_ST

EmbeddingKindFilter = Literal["st", "clip", "both"]


class EmbeddingCandidate(TypedDict):
    """LLM 프롬프트에 실릴 후보 한 건(자산 메타 + 임베딩 유사도). id 는 asset_id(UUID str)."""

    id: str
    file_uri: str
    media_type: str
    emb_score: float
    summary: str


def _channels_param(kind: EmbeddingKindFilter) -> list[str]:
    """필터 문자열 → ``asset_embedding.channel`` 에 넣을 값 목록."""
    if kind == "st":
        return [EMBEDDING_KIND_ST]
    if kind == "clip":
        return [EMBEDDING_KIND_CLIP]
    return [EMBEDDING_KIND_ST, EMBEDDING_KIND_CLIP]


def find_embedding_candidates(
    conn: Connection[Any],
    *,
    source_asset_id: str,
    top_k: int,
    embedding_kind: EmbeddingKindFilter = "both",
) -> list[EmbeddingCandidate]:
    """
    소스와 **같은 채널** 임베딩끼리만 비교하여, 자산별 최대 코사인 유사도로 정렬한 상위 ``top_k`` 후보.

    SQL 구조
        ``src_vecs``: 소스 자산의 (channel, embedding) 목록.
        ``cand``: 타 자산 임베딩과 소스 벡터를 channel 로 조인한 (id, sim) 행.
        ``per_item``: 자산 id 별 ``MAX(sim)`` — 한 자산에 청크/키프레임이 여러 개일 때 가장 가까운 쌍만 반영.
        최종: ``asset`` + ``asset_metadata`` 와 조인해 경로·modality·요약을 붙인다.
    """
    channels = _channels_param(embedding_kind)
    sql = """
        WITH src_vecs AS (
            SELECT channel, embedding
            FROM asset_embedding
            WHERE asset_id = %s
              AND embedding IS NOT NULL
              AND channel = ANY(%s)
        ),
        cand AS (
            SELECT ae.asset_id AS id,
                   (1 - (ae.embedding <=> sv.embedding)) AS sim
            FROM asset_embedding ae
            INNER JOIN src_vecs sv ON sv.channel = ae.channel
            WHERE ae.asset_id <> %s
              AND ae.embedding IS NOT NULL
        ),
        per_item AS (
            SELECT id, MAX(sim) AS best_sim
            FROM cand
            GROUP BY id
        )
        SELECT a.asset_id AS id,
               a.fs_path  AS file_uri,
               a.modality AS media_type,
               p.best_sim::float8 AS emb_score,
               COALESCE(m.ext_meta->>'summary', '') AS summary
        FROM per_item p
        INNER JOIN asset a ON a.asset_id = p.id
        LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id
        WHERE a.status = 'registered'
        ORDER BY p.best_sim DESC
        LIMIT %s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (source_asset_id, channels, source_asset_id, top_k))
        rows = cur.fetchall()
    out: list[EmbeddingCandidate] = []
    for r in rows:
        out.append(
            {
                "id": str(r["id"]),
                "file_uri": str(r["file_uri"]),
                "media_type": str(r["media_type"]),
                "emb_score": float(r["emb_score"] or 0.0),
                "summary": str(r["summary"] or ""),
            }
        )
    return out
