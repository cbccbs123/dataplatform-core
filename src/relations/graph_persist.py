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
        return bool(get_current_settings().topic.canonicalize_enabled)
    except RuntimeError:
        return False


def _as_uuid_str(value: Any) -> str | None:
    """LLM이 돌려준 타깃 id 를 UUID 문자열로 방어적 변환한다.

    LLM 출력은 타입이 보장되지 않아(숫자·None·잘린 문자열) 그대로 쓰면 SQL 단계에서 터진다.

    Args:
        value: 어떤 타입이든 받는다(LLM JSON 값).

    Returns:
        정규화된 UUID 문자열. 파싱 실패면 ``None`` — 호출자는 그 엣지를 skip 한다(예외 아님).
    """
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

    Args:
        src: 소스 자산의 node_id.
        dst: 타깃 자산의 node_id.
        symmetric: 관계 종류가 대칭인지(`relation_kind.is_symmetric`). **True 면 순서를 재배치**해
            같은 쌍이 두 행으로 갈라지지 않게 한다. False 면 방향을 그대로 둔다.

    Returns:
        ``(src_node, dst_node)`` 저장 순서.
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
    """신규 엣지의 status 를 정한다 — LLM 신뢰도 **AND** 임베딩 유사도 두 게이트(033 FR-001).

    둘 다 통과해야 ``active``(자동 승인), 하나라도 못 넘으면 ``proposed``(사람 검토 대기)다.

    Args:
        conf_f: LLM이 매긴 신뢰도(0~1). ``None`` 이면 판정 불가로 보고 무조건 ``proposed``.
        emb_score: 타깃 후보의 코사인 유사도(0~1). ``None`` 은 미달과 같게 취급한다.
        auto_approve_min: 자동 승인 신뢰도 하한. 호출부 기본 ``1.01`` 은 신뢰도가 1을 넘을 수 없어
            **사실상 자동 승인을 끈 값**이다(전건 사람 검토).
        auto_approve_emb_min: 자동 승인 임베딩 유사도 하한. **``0.0`` 이하면 이 게이트 자체를 끄고**
            신뢰도 단독으로 결정한다.

    Returns:
        ``"active"`` 또는 ``"proposed"``.
    """
    if conf_f is None or conf_f < auto_approve_min:
        return "proposed"
    if auto_approve_emb_min > 0.0 and (emb_score is None or emb_score < auto_approve_emb_min):
        return "proposed"
    return "active"


def ensure_asset_node(conn: Connection[Any], asset_id: str) -> str:
    """asset_id 의 asset-노드가 있는지 확인하고, **없으면 만들어서** node_id 를 돌려준다.

    **DB에 쓴다**(노드가 없을 때만 INSERT). 호출자의 트랜잭션 안에서 실행된다.

    node 테이블은 asset 과 entity(단계 D 의료 ER)를 함께 수용하는 통합 노드 테이블이다.
    자산당 1개 asset-노드는 uq_node_asset 부분 유니크(node_kind='asset')로 강제된다.

    READ-then-INSERT 패턴 이유: 대부분의 호출은 이미 노드가 존재하는 경우이므로
    SELECT로 먼저 확인해 불필요한 INSERT 왕복을 줄인다. 동시 삽입 경쟁은
    ON CONFLICT DO NOTHING + 재조회로 처리해 중복 없이 항상 확정된 node_id를 반환한다.

    Returns:
        확정된 ``node_id``(UUID 문자열). 기존 노드면 그 값, 신규면 방금 만든 값.
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
    """LLM이 제안한 엣지들을 검증해 ``graph_edge`` 에 upsert 한다.

    **DB에 쓴다**: 필요한 ``node`` 생성 + ``graph_edge`` INSERT/UPDATE. 호출자의 트랜잭션 안에서
    돈다(``asset_entry._run``). 세 가지를 걸러낸다 — 후보 집합 밖 타깃(LLM 환각) · 자기 자신 참조 ·
    아직 ``active`` 가 아닌 관계 종류(미검토 kind).

    같은 쌍이 다시 제안되면(ON CONFLICT) **신뢰도는 더 큰 값으로**, topic·reason 은 더 높은 신뢰도
    쪽으로 갱신하되 **status 는 건드리지 않는다** — 사람이 내린 검토 결정(특히 ``rejected``)을 LLM
    재제안이 덮어쓰면 안 되기 때문이다.

    Args:
        source_asset_id: 관계의 출발 자산.
        edges: LLM 제안 엣지 목록(``parse_and_normalize_edges`` 결과).
        allowed_target_ids: **환각 차단 화이트리스트** — 후보 검색이 실제로 내놓은 자산 id 집합.
            여기 없는 타깃은 저장하지 않고 skip 으로 센다.
        auto_approve_min: 자동 승인 신뢰도 하한. 기본 ``1.01`` 은 사실상 자동 승인 끔(전건 검토).
        target_emb_scores: ``{타깃 id: 코사인 유사도}``. ``None`` 이거나 맵에 없는 타깃은 ``0.0``
            으로 본다 — ``auto_approve_emb_min`` 이 0보다 크면 그 타깃은 자동 승인되지 않는다.
        auto_approve_emb_min: 자동 승인 임베딩 유사도 하한. ``0.0`` 이면 이 게이트를 끈다.
        collect: 주면 upsert 된 엣지마다 ``{target_asset_id, kind_code, confidence, status}`` 를
            append 한다(skip 은 제외). 계보(``asset_lineage``)에 관계 쌍을 남기려는 호출부가 쓴다.
            ``None`` 이면 수집하지 않는다 — 반환값 계약은 그대로다.

    Returns:
        ``(upserted, skipped)`` 개수.

    Raises:
        RuntimeError: upsert 가 행을 돌려주지 않을 때(현재 SQL로는 도달 불가한 방어적 가드).
            조용히 넘기면 계보에 잘못된 status 가 남으므로 즉시 터뜨린다.
    """
    allowed = frozenset(str(t) for t in allowed_target_ids)
    upserted = 0
    skipped = 0
    # 토글은 **루프 밖에서 한 번만** 읽는다 — 엣지마다 읽으면 같은 배치 안에서 설정이 바뀔 경우
    # 앞뒤 엣지가 다른 규칙으로 저장된다.
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

        # 여기서 다루는 topic 은 **이 쌍(관계)의 맥락 라벨**이지 자산의 주제가 아니다 —
        # 자산 주제는 asset_topic 이 따로 정한다. 검토 화면 표시용으로만 남긴다.
        # 5번째 반환값(자동 보정 여부)을 버리는 것은 의도적이다 — 지금은 아무 동작도 걸려 있지
        # 않은 예비 표식이라 여기서 쓰지 않는다(산출 자체는 테스트가 봉인한다).
        topic_ko, subtopic_ko, topic_en, subtopic_en, _ = coerce_topic_fields_mvp(edge)
        # ⚠️ 이 블록은 **트랜잭션 안에서 LLM 을 부를 수 있다**(캐시가 비어 있을 때).
        #   같은 트랜잭션에 관계 제안 LLM 이 이미 들어 있어 새로운 위험은 아니고, 레지스트리가
        #   시드된 상태에서는 대부분 캐시로 끝난다. 트랜잭션 밖으로 빼려면 호출부 여럿을 함께
        #   고쳐야 해서 미뤄 둔 한계다 — 배치가 길어지면 여기를 먼저 의심할 것.
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
                # 재제안 시 갱신 규칙(위 SQL):
                #   신뢰도 — 더 큰 값으로 화해한다.
                #   topic·reason — **더 높은 신뢰도일 때만** 덮는다(첫 제안의 낡은 값이 굳지 않게).
                #   status — **절대 덮지 않는다**. 사람이 내린 검토 결정, 특히 반려를 LLM 재제안이
                #            되돌리면 안 되기 때문이다.
                # RETURNING 으로 DB 가 확정한 status 를 받아 온다(신규면 계산값, 충돌이면 기존 값).
                (uuid7_str(), a_node, b_node, kind_id, conf_f, reason, topic_json, status_val),
            )
            # ON CONFLICT DO UPDATE 는 신규·충돌 모두 행을 돌려주므로 RETURNING 은 항상 1행이다.
            returned = cur.fetchone()
        # 계보에는 **DB 가 확정한 status** 를 남긴다(방금 계산한 값이 아니라).
        # 사람이 반려한 엣지를 LLM 이 다시 제안하면 계산값은 active 지만 DB 는 rejected 를 유지하는데,
        # 계산값을 기록하면 계보에 "승인됨"으로 남아 사실과 어긋난다.
        # 아래 가드를 예외로 둔 이유: 조용히 계산값으로 폴백하면 그 오염이 소리 없이 되살아난다.
        if returned is None:  # 현재 SQL 로는 도달 불가 — SQL 이 바뀌면 여기서 즉시 드러나게 한다
            raise RuntimeError(
                "graph_edge upsert RETURNING 이 행을 돌려주지 않음 — ON CONFLICT 가 "
                "DO NOTHING 으로 바뀌었는지 확인(B4: 계보 status 는 DB 확정값이어야 함)."
            )
        db_status = returned[0]
        upserted += 1
        if collect is not None:  # 계보 관계쌍 기록용(013) — upsert 된 것만, skip 제외
            collect.append({"target_asset_id": tid, "kind_code": kind_code,
                            "confidence": conf_f, "status": db_status})
    return upserted, skipped
