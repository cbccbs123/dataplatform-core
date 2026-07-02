"""주제 재색인 오케스트레이션 — 관계 변화를 OpenSearch topics 필드에 반영(056 G5·FR-301~304).

왜 이 모듈인가 (접근 C′·증분 재색인)
    관계 단계(``run_relations`` 배치·검토 승인/반려/정정)가 ``graph_edge`` 를 바꾸면, 그 엣지에
    걸린 자산의 **active-only 주제 투영**(``project_asset_topics``)이 달라진다. 전체 재색인
    (``sync_all``) 없이, **변경 자산 + 그 이웃**의 OS topics 2필드(``topics``/``subtopics``)만
    부분 업데이트(``update_asset_topics``)해 검색·패싯을 최신으로 유지한다.

왜 이웃까지 재색인하나 (핵심)
    한 엣지는 **양끝 자산 모두**의 active-only 주제 집계를 바꾼다. 그래서 변경 자산만 재색인하면
    반대편(이웃)의 색인된 topics 가 stale 로 남는다. 따라서 target = 입력 자산 ∪ (각 자산의 active
    이웃). 이웃 수집은 ``graph_query.fetch_active_relations_for_asset`` 를 재사용한다(1홉만 —
    이웃의 이웃까지 확장하지 않는다: 이 배치/승인이 건드린 엣지의 양끝만 영향받으므로).

best-effort·격리 (헌법 8조 결·SC-003 결)
    이 함수는 **어떤 예외도 전파하지 않는다**. PG 읽기 실패·OS 미도달·개별 자산 갱신 실패를
    모두 삼키고 로그만 남기며 ``{updated, failed}`` 를 반환한다. 호출자(run_relations 배치 꼬리·
    검토 승인 커밋 후 훅)는 관계 write 트랜잭션 **밖**(커밋 후)에서 부르므로, 재색인 실패가
    관계 적재·승인·HTTP 응답을 되돌리지 않는다(FR-304). **트랜잭션 밖 호출 전제.**

seam·순수성
    ``fetch_active_relations_for_asset``/``project_asset_topics``/``update_asset_topics``/
    ``get_client``/``get_current_settings`` 를 **모듈 상단에서 import** 한다 — 단위 테스트가
    ``src.search.topic_reindex.<name>`` 위치를 patch 해 실 DB·OS 없이 동작을 통제한다.
    opensearch-py 는 ``get_client`` 내부에서만 지연 import 되므로(색인 IO 시점), 이 모듈 import
    만으로는 opensearch-py 를 당기지 않는다(순수 게이트 보존).

신규 LLM 0 — 주제는 관계 단계의 확정 산출(재사용). 결정성 100%(투영이 결정적, 헌법 3조).
"""
from __future__ import annotations

import logging
from typing import Any

from src.config.settings import get_current_settings
from src.relations.graph_query import fetch_active_relations_for_asset
from src.relations.topic_query import project_asset_topics
from src.search.opensearch_sync import get_client, update_asset_topics

_LOG = logging.getLogger("meta_extract.topic_reindex")


def _collect_targets_and_topics(db: Any, inputs: list[str]) -> list[tuple[str, list[dict]]]:
    """입력 자산 ∪ 각 자산의 active 이웃(1홉)을 모아 자산별 현재 주제를 투영한다(PG 읽기 전용).

    한 **짧은 읽기 트랜잭션**에서: ① 각 입력 자산의 active 이웃 asset_id 를 수집해 target 집합을
    입력 순서 보존·중복 제거로 확장하고, ② 각 target 의 ``project_asset_topics`` 를 계산한다.
    반환: ``[(asset_id, topics), ...]``(target 순). 트랜잭션 경계는 커밋된 관계를 읽으므로(호출은
    관계 write 커밋 후 전제) 최신 상태를 본다.
    """
    seen: set[str] = set(inputs)
    targets: list[str] = list(inputs)  # 입력 순서 보존(결정적)
    with db.transaction() as conn:
        # 1) 각 입력 자산의 active 이웃을 1홉 수집(이웃의 이웃은 확장하지 않음).
        for aid in inputs:
            for nb in fetch_active_relations_for_asset(conn, asset_id=aid, status="active"):
                nid = str(nb["asset_id"])
                if nid not in seen:
                    seen.add(nid)
                    targets.append(nid)
        # 2) target 별 현재 active 주제 투영(같은 트랜잭션 내 읽기).
        return [(aid, project_asset_topics(conn, asset_id=aid)) for aid in targets]


def reindex_asset_topics(db: Any, *, asset_ids: list[str]) -> dict:
    """변경 자산 ∪ 그 active 이웃의 OS topics 2필드(topics·subtopics)를 부분 재색인한다(best-effort·격리).

    Args:
        db: ``PostgresUtil`` — PG 읽기(이웃 수집·주제 투영)에 ``db.transaction()`` 을 쓴다.
        asset_ids: 재색인 기점이 되는 변경 자산 목록(관계 배치/검토가 건드린 자산).

    Returns:
        ``{"updated": int, "failed": int}`` — OS 부분 업데이트에 성공/실패한 자산 수.

    동작:
        1. 입력 자산 dedup(빈 입력 → 즉시 ``{0,0}``, 조회·클라이언트 생성 없음).
        2. PG 읽기: target = 입력 ∪ active 이웃(1홉), 각 target 의 현재 주제 투영. 이 단계 실패는
           삼키고 ``failed=len(입력)`` 로 반환(OS 갱신 시도 없음).
        3. OS 클라이언트·인덱스 획득(``get_client`` + ``settings.opensearch_index`` — index_asset
           /resync 와 동형). 실패 시 삼키고 ``failed=len(target)`` 로 반환.
        4. target 별 ``update_asset_topics`` 부분 문서 갱신. **자산별 try/except** 로 한 자산의
           OS 오류가 나머지를 막지 않게 격리(성공 ``updated++``·실패 ``failed++``·로그).

    **트랜잭션 밖(관계 write 커밋 후) 호출 전제** — 이 함수는 어떤 예외도 전파하지 않는다(FR-304).
    """
    # 입력 정규화·dedup(순서 보존). 빈 입력은 즉시 no-op(OS 접촉·조회 없음).
    inputs: list[str] = []
    seen: set[str] = set()
    for a in asset_ids:
        s = str(a)
        if s not in seen:
            seen.add(s)
            inputs.append(s)
    if not inputs:
        return {"updated": 0, "failed": 0}

    # 2) PG 읽기(이웃 수집·주제 투영). 실패는 격리 — OS 갱신 시도 없이 실패 집계만.
    try:
        projected = _collect_targets_and_topics(db, inputs)
    except Exception as exc:  # noqa: BLE001 — PG 읽기 실패가 배치/승인을 깨지 않는다(best-effort)
        _LOG.warning("topic reindex: 대상 수집/투영 실패(무시): %s", exc)
        return {"updated": 0, "failed": len(inputs)}

    # 3) OS 클라이언트·인덱스 획득(index_asset/resync 와 동형: get_client + opensearch_index).
    try:
        settings = get_current_settings()
        client = get_client(settings.opensearch_url)
        index = settings.opensearch_index
    except Exception as exc:  # noqa: BLE001 — OS 미도달/설정 미초기화도 삼킨다(best-effort)
        _LOG.warning("topic reindex: OS 클라이언트/인덱스 획득 실패(무시): %s", exc)
        return {"updated": 0, "failed": len(projected)}

    # 4) target 별 부분 업데이트 — 자산별 격리(한 자산 실패가 나머지를 막지 않음).
    updated = failed = 0
    for aid, topics in projected:
        try:
            update_asset_topics(client, index, aid, topics)
            updated += 1
        except Exception as exc:  # noqa: BLE001 — 개별 자산 OS 갱신 실패 격리
            failed += 1
            _LOG.warning("topic reindex: 자산 %s OS 부분 업데이트 실패(무시): %s", aid, exc)
    return {"updated": updated, "failed": failed}
