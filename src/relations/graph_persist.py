"""공용 그래프 영속화 — node(asset) 보장 + graph_edge upsert.

엣지 종류는 relation_kind(통제 어휘)를 직접 참조하고, 주제는 graph_edge.topic jsonb 에 저장한다.
대칭 kind 는 (src,dst) 캐논 순서로 1행만 저장하고, 신뢰도는 충돌 시 GREATEST 로 화해한다.
후보 집합 안의 타깃만, active kind 만 적재(환각·미검토 kind 방지).
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from psycopg import Connection

from src.database.ids import uuid7_str
from src.relations.relation_type_catalog import fetch_relation_kind
from src.relations.schema import coerce_topic_fields_mvp
from src.relations.topic_canonicalize import canonicalize_subtopic, canonicalize_topic


def _topic_canonicalize_enabled() -> bool:
    """``TOPIC_CANONICALIZE_ENABLED`` 설정 조회(058 FR-401). 미초기화 시 보수적 **False** 폴백.

    settings 미초기화(순수 단위 테스트 등)에서는 ``get_current_settings()`` 가 ``RuntimeError`` 이므로
    False 로 폴백해 현행 경로(canonicalize 미배선·coerce 결과 그대로)를 보존한다 — 다른 선택 설정
    조회 헬퍼의 미초기화 보수 폴백과 동형. 운영 진입점은 항상 ``init_settings`` 하므로 이 폴백은 비운영 경로다.
    기본값 False 라 시드 전에는 관계 저장이 바이트 동일하다(동작 불변·회귀 0)."""
    from src.config.settings import get_current_settings

    try:
        return bool(get_current_settings().topic_canonicalize_enabled)
    except RuntimeError:
        return False


def _as_uuid_str(value: Any) -> str | None:
    # LLM이 반환한 target_media_item_id가 문자열이 아닌 타입일 수 있으므로 방어적 변환.
    # 유효하지 않은 UUID면 None을 돌려 호출자가 엣지를 skip할 수 있게 한다.
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def _canonical_pair(src: str, dst: str, *, symmetric: bool) -> tuple[str, str]:
    """대칭 kind 면 (min,max) 로 정렬해 무방향 쌍을 1행으로 모은다. 비대칭은 방향 유지.

    설계 의도: relation_kind.is_symmetric=True 인 kind(예: '유사', '관련')는
    A→B 와 B→A 가 의미상 동일하다. 두 자산이 각각 소스로 처리될 때 같은 엣지가
    (src=A,dst=B) 와 (src=B,dst=A) 로 이중 삽입되는 것을 막기 위해 node_id 의
    문자열 대소 비교로 항상 작은 쪽을 src 로 고정한다.
    이 순서가 uq_graph_edge_kind(src_node, dst_node, relation_kind_id) 유니크 제약과
    맞물려 ON CONFLICT 1행 수렴을 보장한다.
    """
    if symmetric and dst < src:
        return dst, src
    return src, dst


def _decide_status(
    conf_f: float | None,
    emb_score: float | None,
    auto_approve_min: float,
    auto_approve_emb_min: float,
) -> str:
    """신규 엣지 status 결정 — LLM conf AND emb_score 두 게이트(033 FR-001).

    - conf 가 None 이거나 auto_approve_min 미만이면 'proposed'(현행과 동일).
    - auto_approve_emb_min<=0.0 이면 emb 변이가 **무력**화돼 conf 단독 결정(동작 보존·SC-001).
    - emb_min>0 이고 emb_score 가 그 하한 미달이면(또는 None) conf 충분해도 'proposed'(SC-002).
    - 둘 다 통과해야 'active'.
    """
    if conf_f is None or conf_f < auto_approve_min:
        return "proposed"
    if auto_approve_emb_min > 0.0 and (emb_score is None or emb_score < auto_approve_emb_min):
        return "proposed"
    return "active"


def ensure_asset_node(conn: Connection[Any], asset_id: str) -> str:
    """asset_id 의 asset-노드를 보장하고 node_id(UUID str) 반환.

    node 테이블은 asset 과 entity(단계 D 의료 ER)를 함께 수용하는 통합 노드 테이블이다.
    자산당 1개 asset-노드는 uq_node_asset 부분 유니크(node_kind='asset')로 강제된다.

    READ-then-INSERT 패턴 이유: 대부분의 호출은 이미 노드가 존재하는 경우이므로
    SELECT로 먼저 확인해 불필요한 INSERT 왕복을 줄인다. 동시 삽입 경쟁은
    ON CONFLICT DO NOTHING + 재조회로 처리해 중복 없이 항상 확정된 node_id를 반환한다.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT node_id FROM node WHERE node_kind = 'asset' AND asset_id = %s LIMIT 1",
            (asset_id,),
        )
        row = cur.fetchone()
        if row is not None:
            return str(row[0])
        nid = uuid7_str()
        cur.execute(
            """
            INSERT INTO node (node_id, node_kind, asset_id) VALUES (%s, 'asset', %s)
            ON CONFLICT (asset_id) WHERE node_kind = 'asset' DO NOTHING
            RETURNING node_id
            """,
            (nid, asset_id),
        )
        inserted = cur.fetchone()
        if inserted is not None:
            return str(inserted[0])
        # RETURNING 이 빈 경우 = 동시 삽입 경쟁에서 다른 트랜잭션이 먼저 커밋.
        # 재조회로 그 행을 가져온다.
        cur.execute(
            "SELECT node_id FROM node WHERE node_kind = 'asset' AND asset_id = %s LIMIT 1",
            (asset_id,),
        )
        return str(cur.fetchone()[0])


def sync_graph_edges(
    conn: Connection[Any],
    *,
    source_asset_id: str,
    edges: list[dict[str, Any]],
    allowed_target_ids: frozenset[str],
    auto_approve_min: float = 1.01,
    target_emb_scores: dict[str, float] | None = None,
    auto_approve_emb_min: float = 0.0,
    collect: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """후보 집합 안 타깃·active kind 만 graph_edge 에 upsert. Returns (upserted, skipped).

    ``collect`` 리스트를 주면 upsert 된 엣지마다 ``{target_asset_id, kind_code, confidence, status}``
    를 append 한다(skip 제외). 계보(asset_lineage)에 "어떤 자산과 어떤 관계로 생성됐는지" 쌍을
    남기기 위한 단일 출처(013) — 반환 튜플 계약은 불변(미전달 시 기존 동작 그대로).

    신규 엣지 status: LLM conf AND emb_score 두 게이트를 모두 통과하면 'active'(자동승인),
    아니면 'proposed'(검토 대기). conf>=auto_approve_min 이고 (emb_min>0 일 때) 타깃 emb_score>=emb_min.
    기본 auto_approve_min=1.01·auto_approve_emb_min=0.0(또는 맵 미전달)이면 emb 변이 무력 →
    현행과 비트-동일(동작 보존·033 SC-001). 충돌 시 status 는 보존(사람 결정 유지), confidence 만 GREATEST.
    target_emb_scores: {타깃 id: 후보 코사인 유사도}. 미전달 타깃은 0.0(emb_min>0 이면 proposed).
    """
    allowed = frozenset(str(t) for t in allowed_target_ids)
    upserted = 0
    skipped = 0
    # 058 FR-401: topic 정본화 배선 여부를 루프 밖에서 1회 조회(엣지마다 동일·설정 불변).
    # 기본 False → 아래 배선 블록을 통째로 건너뛰어 canonicalize·registry 쓰기·LLM 0(동작 불변).
    canonicalize_on = _topic_canonicalize_enabled()
    src_node = ensure_asset_node(conn, source_asset_id)
    for edge in edges:
        tid = _as_uuid_str(edge.get("target_media_item_id"))
        # 후보 집합 밖 타깃은 LLM 환각으로 간주해 엣지 생성 거부.
        # 자기 자신 참조도 루프 엣지가 되므로 차단한다.
        if tid is None or tid not in allowed or tid == source_asset_id:
            skipped += 1
            continue
        code = edge.get("relation_type_code")
        if not code or not str(code).strip():
            skipped += 1
            continue
        kind_code = str(code).strip().lower()
        # active 상태인 kind만 허용 — 미검토(inactive) kind가 그래프에 섞이는 것을 방지.
        # register_new_relation_kinds 가 신규 kind를 inactive로 등록하므로,
        # 같은 사이클 내에서 방금 등록된 kind는 여기서 걸러져 다음 검토 사이클 이후에야 엣지화된다.
        kind = fetch_relation_kind(conn, kind_code=kind_code, status="active")
        if kind is None:  # 비활성/미등록 kind 는 엣지로 만들지 않음(검토 대기)
            skipped += 1
            continue
        kind_id = str(kind["relation_kind_id"])
        symmetric = bool(kind["is_symmetric"])

        # topic은 C+ 슬림화 설계에 따라 graph_edge.topic jsonb에 비정규화 저장.
        # relation_type / relation_subtopic / relation_topic_parent 3테이블은 v210에서 드랍됨.
        # 065 FR-405: graph_edge.topic 은 **관계 맥락 라벨(자산 주제 아님)** — 이 쌍(pairwise) 관계를
        #   설명하는 메타일 뿐, 자산의 주제는 asset_topic 정본이 결정한다. 기록은 존치(관계 검토 UI용)·
        #   소비만 끊음(065). 컬럼 제거는 후속.
        topic_ko, subtopic_ko, topic_en, subtopic_en, _ = coerce_topic_fields_mvp(edge)
        # 058 FR-401: persist 직전 정본화(플래그 게이트). off(기본)면 이 블록 통째로 skip →
        # coerce 결과를 그대로 저장(canonicalize·registry·LLM 0·바이트 동일·동작 불변·T403).
        # on 이면 topic 은 canonicalize_topic(정본 ko/en), subtopic 은 canonicalize_subtopic 통과.
        # ⚠ LLM-in-transaction 한계: sync_graph_edges 는 execute_in_transaction 안에서 돌고,
        #   flag-on 경로의 canonicalize_topic 은 캐시 미스 시 judge_topic(LLM)을 부를 수 있다.
        #   그러나 (1) 이 트랜잭션에는 관계-제안 LLM(propose_edges_json)이 이미 동거하는 선례가 있고
        #   (asset_entry._run), (2) 시드(G5)된 레지스트리에선 alias 정확일치가 대부분이라 LLM 은
        #   희소하다(캐시 히트). 트랜잭션 밖 사전 정본화는 다중 호출부(asset_entry·sample) 배선 확장이
        #   필요해 1차 배제하고, 여기 배선 + 본 주석으로 한계를 명시한다(ADR 2026-07-06·헌법 ⑤).
        if canonicalize_on:
            res = canonicalize_topic(conn, topic_ko, topic_en)
            topic_ko = res["canonical_ko"]
            # canonical_en 이 None(registry topic_en NULL)이면 기존 topic_en 보존(빈 라벨 방지).
            topic_en = res["canonical_en"] or topic_en
            sub = canonicalize_subtopic(conn, topic_ko, subtopic_ko)
            if sub is None:
                # 모달리티어/계층 규칙으로 비운 경우 en 도 함께 비운다(계층 일관·FR-301/302).
                subtopic_ko, subtopic_en = "", ""
            else:
                subtopic_ko = sub  # subtopic_en 정본화(영문)는 후속 여지(1차는 ko 라벨만)
        topic_json = json.dumps(
            {"topic_ko": topic_ko, "subtopic_ko": subtopic_ko,
             "topic_en": topic_en, "subtopic_en": subtopic_en},
            ensure_ascii=False,
        )
        conf = edge.get("confidence")
        try:
            conf_f = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf_f = None
        reason = str(edge.get("reason") or "").strip() or None
        # auto_approve_min 기본값 1.01 = 신뢰도가 1.0을 초과할 수 없으므로 사실상 자동승인 없음.
        # 운영에서 임계를 낮추면(예: 0.85) 해당 구간 이상 엣지는 HITL 없이 바로 'active'.
        # 033 FR-001: emb_min>0 이면 타깃 emb_score 도 통과해야 active(AND 게이트). 맵 미전달·emb_min=0 이면 무력.
        emb_s = (target_emb_scores or {}).get(tid, 0.0)
        status_val = _decide_status(conf_f, emb_s, auto_approve_min, auto_approve_emb_min)

        dst_node = ensure_asset_node(conn, tid)
        a_node, b_node = _canonical_pair(src_node, dst_node, symmetric=symmetric)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO graph_edge
                    (edge_id, src_node, dst_node, relation_kind_id, confidence, reason, topic, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (src_node, dst_node, relation_kind_id)
                DO UPDATE SET
                    confidence = GREATEST(
                        COALESCE(graph_edge.confidence, 0), COALESCE(EXCLUDED.confidence, 0)),
                    topic = CASE
                        WHEN COALESCE(EXCLUDED.confidence, 0) > COALESCE(graph_edge.confidence, 0)
                        THEN EXCLUDED.topic ELSE graph_edge.topic END,
                    reason = CASE
                        WHEN COALESCE(EXCLUDED.confidence, 0) > COALESCE(graph_edge.confidence, 0)
                        THEN EXCLUDED.reason ELSE graph_edge.reason END,
                    updated_at = now()
                RETURNING status
                """,
                # ON CONFLICT(032·#5): confidence 더 큰 재제안의 topic·reason 만 갱신(작거나 같으면 기존 유지)
                # — 첫 제안 stale topic 고정을 해소한다. confidence 는 GREATEST 화해.
                # ★ status 는 **미갱신** — 사람이 한 번 내린 검토 결정(특히 rejected)을 LLM 재제안이 덮어쓰면 안 된다.
                # 069 B4(P2-4): RETURNING status 로 DB 확정 status(신규=status_val, 충돌=기존 보존값)를 회수한다.
                (uuid7_str(), a_node, b_node, kind_id, conf_f, reason, topic_json, status_val),
            )
            # ON CONFLICT DO UPDATE 는 신규·충돌 모두 행을 돌려주므로 RETURNING 은 항상 1행이다.
            returned = cur.fetchone()
        # 계보에는 계산 status_val 이 아니라 DB 실제 status 를 기록한다 — 반려 재제안이 계보에
        # active 로 남는 오염을 차단(B4). 방어적으로 RETURNING 부재 시에만 status_val 로 폴백.
        db_status = returned[0] if returned is not None else status_val
        upserted += 1
        if collect is not None:  # 계보 관계쌍 기록용(013) — upsert 된 것만, skip 제외
            collect.append({"target_asset_id": tid, "kind_code": kind_code,
                            "confidence": conf_f, "status": db_status})
    return upserted, skipped
