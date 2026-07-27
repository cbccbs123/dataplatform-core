"""임베딩 기반 **관계 후보 검색**(pgvector).

역할
    소스 자산의 ``asset_embedding`` 벡터와 **같은 채널(channel)** 을 가진 다른 자산의 임베딩 사이
    **코사인 유사도**(``1 - (a <=> b)``)를 계산해, 자산당 최고 유사도로 집계한 뒤 상위 ``top_k`` 를
    돌려준다. 여기서 걸러진 후보만 관계 제안 LLM 프롬프트에 실린다.

채널(channel)이란
    임베딩을 만든 모델별 벡터 공간의 이름이다(``st``=텍스트 모델·``clip``=이미지 모델). 공간이 다르면
    코사인 비교가 무의미하므로 **같은 채널끼리만** 비교한다. 차원은 ``FIX_EMBEDDING_DIMENSION``
    (1536D)과 일치해야 한다 — 추출·적재 파이프라인과 같은 값이다.

이 모듈은 **조회 전용**이다(DB 쓰기 없음).
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
    """LLM 프롬프트에 실릴 후보 한 건(자산 메타 + 임베딩 유사도). id 는 asset_id(UUID str).

    066 FR-201: 후보의 자기주제(``asset_topic``)를 동반한다 — ``topic_ko``/``subtopic_ko``.
    FR-101 의 EXISTS 배제로 미부여 후보는 이미 빠지므로 ``topic_ko`` 는 사실상 항상 존재하나,
    LEFT JOIN 특성상 방어적으로 ``None`` 을 허용한다(호출부가 None 을 견딜 것).
    """

    id: str
    file_uri: str
    media_type: str
    emb_score: float
    summary: str
    topic_ko: str | None
    subtopic_ko: str | None


def _channels_param(kind: EmbeddingKindFilter) -> list[str]:
    """필터 문자열을 ``asset_embedding.channel`` 에 넣을 값 목록으로 바꾼다.

    ``both`` 는 채널별로 따로 비교한 뒤 자산 단위 ``MAX`` 로 합산된다(호출부 SQL).
    텍스트 채널의 실제 이름은 **운영 활성 임베딩 채널**을 따른다 — 적재·검색·관계가 같은 값을 써야
    같은 공간에서 비교되기 때문이다. CLIP(이미지) 채널은 활성 설정과 무관하게 고정이다.

    Args:
        kind: ``st``(텍스트만) · ``clip``(이미지만) · ``both``(둘 다).

    Returns:
        채널 이름 목록. ``both`` 면 [텍스트, CLIP] 2개.

    Raises:
        없음. settings 미초기화(순수 단위 테스트 등)로 활성 채널 해소가 실패하면 예외를 올리지 않고
        ``st`` 로 보수적 폴백한다(운영 경로인 관계 배치는 설정 초기화가 선행되므로 미발생).
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
    """소스와 **같은 채널** 임베딩끼리만 비교해, 자산별 최대 유사도 상위 ``top_k`` 후보를 반환한다.

    **조회 전용**(DB 쓰기 없음). ``registered`` 상태이면서 자기주제가 부여된 자산만 후보가 된다 —
    내용이 없어 엉터리 임베딩을 가진 자산이 남의 후보를 오염시키는 것을 SQL 단계에서 막는다.

    SQL 구조
        ``src_vecs``: 소스 자산의 (channel, embedding) 목록.
        ``cand``: 타 자산 임베딩과 소스 벡터를 channel 로 조인한 (id, sim) 행.
        ``per_item``: 자산 id 별 ``MAX(sim)`` — 한 자산에 청크/키프레임이 여러 개일 때 가장 가까운 쌍만 반영.
                      ``HAVING MAX(sim) >= min_sim`` 로 유사도 하한 미만 후보 제거.
        최종: ``asset`` + ``asset_metadata`` 와 조인해 경로·modality·요약을 붙인다.

    Args:
        source_asset_id: 관계를 찾을 기준 자산. 자기 자신은 후보에서 제외된다.
        top_k: 반환할 최대 후보 **자산 수**(청크 수가 아니다).
        embedding_kind: 비교할 채널(``st``/``clip``/``both``).
        min_sim: 코사인 유사도 하한(0~1). 이 값 **미만**이면 후보에서 버려 LLM 토큰 낭비를 막는다.
            ``0.0`` 이면 하한 없음(0-노름 제외는 별개로 항상 적용).

    Returns:
        ``EmbeddingCandidate`` 리스트. ``emb_score`` 내림차순, 동점이면 ``id`` 오름차순으로
        **결정적** 정렬된다(헌법 3조 재현성). 조건에 맞는 후보가 없으면 빈 리스트.
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
               COALESCE(m.ext_meta->>'summary', '') AS summary,
               -- 066 FR-201: 후보의 자기주제(asset_topic) 정본을 동반 → 관계 LLM 이 주제 정합을
               --   맥락(soft)으로 보게 한다. asset_id PK 라 1:1(행 중복 없음).
               t.topic_ko    AS topic_ko,
               t.subtopic_ko AS subtopic_ko
        FROM per_item p
        INNER JOIN asset a ON a.asset_id = p.id
        LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id
        -- 066 FR-201: 후보 주제 동반(값 로드용 LEFT JOIN). 미부여 배제는 아래 EXISTS 가 담당.
        LEFT JOIN asset_topic t ON t.asset_id = a.asset_id
        -- registered 상태만 포함 — received/deferred 자산은 관계 대상에서 제외.
        WHERE a.status = 'registered'
          -- 066 FR-101: 미부여(asset_topic 행 없음·무내용) 후보를 관계 대상에서 배제한다.
          --   내용 없는 자산의 엉터리 임베딩이 남의 후보를 오염(헛매칭)시키는 것을 원천 차단.
          --   결정적 필터(존재 여부)라 정렬·파라미터·재현성 불변.
          AND EXISTS (SELECT 1 FROM asset_topic at WHERE at.asset_id = a.asset_id)
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
        # 066 FR-201: 주제는 방어적으로 None 허용(LEFT JOIN — EXISTS 배제로 사실상 항상 존재).
        topic_ko = r.get("topic_ko")
        subtopic_ko = r.get("subtopic_ko")
        out.append(
            {
                "id": str(r["id"]),
                "file_uri": str(r["file_uri"]),
                "media_type": str(r["media_type"]),
                "emb_score": float(r["emb_score"] or 0.0),
                "summary": str(r["summary"] or ""),
                "topic_ko": str(topic_ko) if topic_ko is not None else None,
                "subtopic_ko": str(subtopic_ko) if subtopic_ko is not None else None,
            }
        )
    return out
