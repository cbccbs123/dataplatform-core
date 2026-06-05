"""샘플 도메인 팩 cross_asset 전략(결정적·비학습·무LLM).

목적(spec 016)
    008 이 만든 cross_asset 슬롯 resolve seam 의 **성공 경로**를, ``contracts.py`` 에
    정의돼 있으나 한 번도 실행된 적 없는 cross_asset 계약 4종으로 처음 실행해 증명한다.
    이 모듈은 그 계약을 만족하는 **트리비얼·결정적 데모 전략**이다.

헌법 준수
    - 1조(비학습): 학습/파인튜닝/.fit 없음 — 사전계산된 임베딩 유사도와 고정 규칙만 사용.
    - 2조(LLM 단일 seam): LLM 을 전혀 호출하지 않는다(샘플은 결정적 규칙 점수).
    - 3조(결정성 100%): 정렬 tiebreak 는 target asset_id 오름차순으로 고정한다.
    - 6조(스키마 불변): persist 는 기존 ``graph_edge`` + 기존 ``relation_kind`` 어휘만
      재사용한다. 신규 테이블·컬럼·마이그레이션·relation_kind 카탈로그 등록 없음.

계약(contracts.py Protocol)
    - ``sample_candidates(conn, source_asset_id) -> list[Candidate]``      # CandidateStage
    - ``sample_score(conn, pairs) -> list[ScoredPair]``                     # ScoreStage
    - ``sample_decide(scored) -> list[Decision]``                          # DecideStage
    - ``sample_persist_edges(conn, decisions) -> None``                    # EdgePersistStage
"""
from __future__ import annotations

from typing import Any

from psycopg import Connection

from src.pipeline.cross_types import Candidate, Decision, Evidence, ScoredPair
from src.relations.asset_candidates import find_embedding_candidates
from src.relations.graph_persist import sync_graph_edges

# ── 상수 ────────────────────────────────────────────────────────────────────
# 샘플 후보 top-k. 데모이므로 작게 고정한다.
SAMPLE_TOP_K = 10
# 결정 임계 τ — score 가 이 값 이상이면 'match'. 경계는 >= 로 match 쪽에 포함(결정적).
SAMPLE_DECIDE_TAU = 0.5
# 샘플 데모가 쓰는 기존 relation_kind 어휘(시드 active). 의미 약결합 데모용.
SAMPLE_RELATION_KIND = "same_series"
# Evidence/검색 식별용 블로킹 키·비교자(추적 메타).
_BLOCK_KEY = "embedding"
_COMPARATOR = "embedding_cosine"
_METHOD = "sample"


def sample_candidates(conn: Connection[Any], source_asset_id: str) -> list[Candidate]:
    """임베딩 top-k 이웃을 ``Candidate`` 로 매핑(결정적).

    ``find_embedding_candidates`` 결과(같은 채널 코사인 top-k)를 재사용한다.
    각 행을 ``Candidate(source_id, target_id, block_key='embedding', method='sample')`` 로
    매핑하고, **target asset_id 오름차순**으로 정렬해 동일 입력 2회 동일 출력을 보장한다
    (헌법 3조; find_embedding_candidates 자체도 (best_sim DESC, id ASC) tiebreak 를 갖지만
    candidates 슬롯의 계약 순서는 target_id 오름차순으로 명시적으로 재고정한다).
    """
    rows = find_embedding_candidates(
        conn, source_asset_id=source_asset_id, top_k=SAMPLE_TOP_K
    )
    cands = [
        Candidate(
            source_id=source_asset_id,
            target_id=str(r["id"]),
            block_key=_BLOCK_KEY,
            method=_METHOD,
        )
        for r in rows
    ]
    # 결정성: target asset_id 오름차순 고정.
    cands.sort(key=lambda c: c.target_id)
    return cands


def sample_score(conn: Connection[Any], pairs: list[Candidate]) -> list[ScoredPair]:
    """후보를 결정적 코사인 점수로 채점(LLM 미호출).

    헌법 2조: LLM seam 을 전혀 호출하지 않는다. 점수는 후보의 임베딩 코사인 유사도라는
    **사전계산 신호**(find_embedding_candidates)에서 가져온 고정 규칙이다.

    설계
        모든 pair 의 source_id 가 동일하다는 전제 하에, 그 소스에 대한 임베딩 이웃을 한 번
        조회해 ``{target_id: 코사인유사도}`` 맵을 만든다. 후보가 그 맵에 없으면(이웃 컷오프
        밖) 점수 0.0 으로 보수 처리한다. ``Evidence(field='embedding',
        comparator='embedding_cosine', similarity=score)`` 를 부착해 추적성을 남긴다.
        조회 1회를 모든 pair 가 공유하므로 동일 입력 2회 동일 출력(결정성)이다.
    """
    if not pairs:
        return []
    # 모든 pair 가 같은 소스라는 계약 전제. 첫 pair 의 소스로 이웃 유사도 맵을 만든다.
    source_id = pairs[0].source_id
    rows = find_embedding_candidates(
        conn, source_asset_id=source_id, top_k=SAMPLE_TOP_K
    )
    sim_by_target: dict[str, float] = {
        str(r["id"]): float(r["emb_score"] or 0.0) for r in rows
    }
    out: list[ScoredPair] = []
    for c in pairs:
        score = sim_by_target.get(c.target_id, 0.0)
        ev = Evidence(
            field=_BLOCK_KEY,
            comparator=_COMPARATOR,
            similarity=score,
            weight=score,
        )
        out.append(ScoredPair(candidate=c, score=score, evidence=[ev]))
    return out


def sample_decide(scored: list[ScoredPair]) -> list[Decision]:
    """결정적 임계로 verdict 결정(순수 함수, conn 불필요).

    규칙: ``score >= SAMPLE_DECIDE_TAU`` 이면 'match', 아니면 'non_match'.
    경계값(score == τ)은 ``>=`` 로 match 쪽에 결정적으로 포함한다(헌법 3조).
    입력 순서를 보존하므로 동점·경계 입력이라도 동일 입력 2회 동일 출력이다.
    """
    out: list[Decision] = []
    for sp in scored:
        verdict = "match" if sp.score >= SAMPLE_DECIDE_TAU else "non_match"
        out.append(
            Decision(candidate=sp.candidate, verdict=verdict, score=sp.score)
        )
    return out


def sample_persist_edges(conn: Connection[Any], decisions: list[Decision]) -> None:
    """verdict=='match' 결정만 기존 ``graph_edge`` 로 upsert(헌법 6조: 스키마 불변).

    재사용
        기존 ``graph_persist.sync_graph_edges`` 헬퍼 한 곳으로만 적재한다. 신규 테이블·
        컬럼·마이그레이션·relation_kind 카탈로그 등록은 일절 하지 않는다. relation_type_code
        는 시드 active 어휘 ``SAMPLE_RELATION_KIND``('same_series')만 사용한다(데모 의미
        약결합). ``allowed_target_ids`` 는 match 타깃 집합으로 두어 sync_graph_edges 의
        환각 방지 게이트(후보 집합 밖 타깃 거부)와 정합시킨다.

    설계
        Decision('match') 한 건 → sync_graph_edges 가 받는 edge dict 한 건으로 매핑한다
        (target_media_item_id / relation_type_code / confidence=score / reason). 모든
        결정의 source_id 가 동일하다는 계약 전제 하에 첫 match 의 source 를 쓴다.
        'non_match' 는 건너뛰며, match 가 하나도 없으면 적재 호출 자체를 생략한다.
    """
    matches = [d for d in decisions if d.verdict == "match"]
    if not matches:
        # 적재할 엣지가 없으면 sync_graph_edges 를 부르지 않는다(불필요한 노드 보장 회피).
        return None
    source_id = matches[0].candidate.source_id
    allowed_target_ids = frozenset(d.candidate.target_id for d in matches)
    edges: list[dict[str, Any]] = [
        {
            "target_media_item_id": d.candidate.target_id,
            "relation_type_code": SAMPLE_RELATION_KIND,
            "confidence": d.score,
            "reason": "샘플 팩 결정적 cross_asset 전략(데모)",
        }
        for d in matches
    ]
    sync_graph_edges(
        conn,
        source_asset_id=source_id,
        edges=edges,
        allowed_target_ids=allowed_target_ids,
    )
    return None
