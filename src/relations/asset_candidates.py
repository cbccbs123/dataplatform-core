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

import logging
from typing import Any, Literal, TypedDict

from psycopg import Connection
from psycopg.rows import dict_row

from src.config.embedding_constants import EMBEDDING_KIND_CLIP, EMBEDDING_KIND_ST
from src.config.settings import active_embed_channel

_LOG = logging.getLogger(__name__)

EmbeddingKindFilter = Literal["st", "clip", "both"]


class EmbeddingCandidate(TypedDict):
    """LLM 프롬프트에 실릴 후보 한 건(자산 메타 + 임베딩 유사도). id 는 asset_id(UUID str)."""

    id: str
    file_uri: str
    media_type: str
    emb_score: float
    summary: str


def _channels_param(kind: EmbeddingKindFilter) -> list[str]:
    """필터 문자열 → ``asset_embedding.channel`` 에 넣을 값 목록.

    채널 간 벡터는 공간이 달라 코사인 비교가 무의미하므로, SQL에서 channel로 조인해
    같은 채널끼리만 유사도를 계산한다. "both"는 채널별 독립 비교 후 자산 단위 MAX로 합산된다.

    텍스트 채널은 운영 활성 임베딩 채널(018, 적재·검색·관계 단일 출처)을 따른다 — 기본 active='st'
    이면 기존과 동치. settings 미초기화(순수 단위 등)에서는 활성 해소가 ``RuntimeError`` 이므로
    기존 기본 'st' 로 보수적 폴백한다(회귀 0). CLIP 채널은 active 와 무관하게 무변경(텍스트 한정).
    """
    if kind == "clip":
        return [EMBEDDING_KIND_CLIP]
    try:
        text_channel = active_embed_channel()
    except RuntimeError:
        # settings 미초기화(테스트 등): 'st' 보수 폴백. 운영(run_relations)은 init_settings 필수.
        _LOG.warning("settings 미초기화 — 관계 후보 텍스트 채널 'st' 보수 폴백")
        text_channel = EMBEDDING_KIND_ST
    if kind == "st":
        return [text_channel]
    return [text_channel, EMBEDDING_KIND_CLIP]


def find_embedding_candidates(
    conn: Connection[Any],
    *,
    source_asset_id: str,
    top_k: int,
    embedding_kind: EmbeddingKindFilter = "both",
    min_sim: float = 0.0,
) -> list[EmbeddingCandidate]:
    """
    소스와 **같은 채널** 임베딩끼리만 비교하여, 자산별 최대 코사인 유사도로 정렬한 상위 ``top_k`` 후보.

    SQL 구조
        ``src_vecs``: 소스 자산의 (channel, embedding) 목록.
        ``cand``: 타 자산 임베딩과 소스 벡터를 channel 로 조인한 (id, sim) 행.
        ``per_item``: 자산 id 별 ``MAX(sim)`` — 한 자산에 청크/키프레임이 여러 개일 때 가장 가까운 쌍만 반영.
                      ``HAVING MAX(sim) >= min_sim`` 로 유사도 하한 미만 후보 제거.
        최종: ``asset`` + ``asset_metadata`` 와 조인해 경로·modality·요약을 붙인다.
    """
    channels = _channels_param(embedding_kind)
    sql = """
        WITH src_vecs AS (
            -- 소스 자산의 채널별 임베딩 벡터. 청크/키프레임이 여러 개일 수 있다.
            SELECT channel, embedding
            FROM asset_embedding
            WHERE asset_id = %s
              AND embedding IS NOT NULL
              -- 034: 영노름(빈/실패 콘텐츠) 벡터 제외. 영벡터 코사인은 NaN 이고 PG 는 NaN 을
              --      HAVING(NaN>=min_sim TRUE)·ORDER BY DESC(최댓값)로 통과시켜 후보를 오염시킨다.
              AND vector_norm(embedding) > 0
              AND channel = ANY(%s)
        ),
        cand AS (
            -- 타 자산과 소스 벡터를 channel로 inner join → 같은 공간끼리만 비교.
            -- <=> 는 pgvector 코사인 거리(0=동일, 2=반대). 1에서 빼 유사도로 변환.
            SELECT ae.asset_id AS id,
                   (1 - (ae.embedding <=> sv.embedding)) AS sim
            FROM asset_embedding ae
            INNER JOIN src_vecs sv ON sv.channel = ae.channel
            WHERE ae.asset_id <> %s
              AND ae.embedding IS NOT NULL
              -- 034: 영노름 타깃 제외(위와 동일 — NaN 코사인이 top_k 를 점령하는 근원 차단).
              AND vector_norm(ae.embedding) > 0
        ),
        per_item AS (
            -- 한 자산에 청크/키프레임이 여러 개일 때 가장 가까운 쌍 1개만 대표값으로 사용.
            -- HAVING으로 min_sim 하한 적용 — 노이즈 후보를 LLM에 넘기지 않아 토큰 낭비 방지.
            SELECT id, MAX(sim) AS best_sim
            FROM cand
            GROUP BY id
            HAVING MAX(sim) >= %s
        )
        SELECT a.asset_id AS id,
               a.fs_path  AS file_uri,
               a.modality AS media_type,
               p.best_sim::float8 AS emb_score,
               COALESCE(m.ext_meta->>'summary', '') AS summary
        FROM per_item p
        INNER JOIN asset a ON a.asset_id = p.id
        LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id
        -- registered 상태만 포함 — received/deferred 자산은 관계 대상에서 제외.
        WHERE a.status = 'registered'
        -- best_sim 동률 시 후보 id(asset_id) ASC 를 보조 정렬로 둬 순서를 결정적으로 고정.
        -- tiebreaker 가 없으면 동률 후보 순서가 PG 실행 계획에 따라 흔들려 헌법 3조(재현성)를 깬다.
        ORDER BY p.best_sim DESC, p.id ASC
        LIMIT %s
    """
    # 파라미터 순서: src_vecs의 asset_id, channels, cand의 asset_id(self 제외), min_sim, top_k.
    # source_asset_id가 두 번 등장하므로 순서 오류 주의.
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (source_asset_id, channels, source_asset_id, min_sim, top_k))
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
