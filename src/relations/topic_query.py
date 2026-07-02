"""주제 투영·탐색 seam — 자산의 active 관계 이웃 ``topic`` 을 주제 집합으로 투영.

왜 이 seam 인가 (056 접근 C′)
    관계 생성이 확정한 ``graph_edge.topic``(주제/하위주제 jsonb: topic_ko·subtopic_ko·
    topic_en·subtopic_en)을 **새 PG 테이블 없이** 자산 관점으로 투영한다. 검색(OS 색인)과
    탐색(포털 same-topic) 두 소비자가 이 단일 투영을 공유한다.
    **새 LLM 호출 0** — 주제는 관계 단계의 기존 산출을 재사용한다(헌법 2조).

seam 재사용
    이웃 수집은 ``graph_query.fetch_active_relations_for_asset`` 을 그대로 쓴다. 이 read
    seam 은 대칭 엣지 누락 방지(양방향 매칭)·status 필터가 이미 검증돼 있어, 여기서는
    이웃들의 ``topic`` 만 집계하면 된다(순진한 단방향 쿼리 재발명 금지).
    ``fetch_active_relations_for_asset`` 는 **모듈 상단에서 import** 한다 — 테스트가
    ``src.relations.topic_query.fetch_active_relations_for_asset`` 위치를 patch 해 실 DB
    없이 이웃을 통제할 수 있게 하기 위함이다.

결정성 (헌법 3조)
    투영은 **active-only** 이며 정렬 타이브레이커를
    ``weight desc → topic_ko asc → subtopic_ko asc`` 로 고정한다. 같은 입력에 같은 출력.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from src.relations.graph_query import fetch_active_relations_for_asset


def project_asset_topics(conn, *, asset_id: str, top_n: int = 10) -> list[dict]:
    """``asset_id`` 의 active 이웃 엣지 ``topic`` 을 ``(topic_ko, subtopic_ko)`` 로 집계.

    Args:
        conn: DB 연결(이웃 수집 seam 에 그대로 전달).
        asset_id: 투영 대상 자산.
        top_n: 반환 상한(기본 10). weight 상위 top_n 개만.

    Returns:
        ``[{topic_ko, subtopic_ko, topic_en, subtopic_en, weight}]``.
        - ``weight`` = 그 ``(topic_ko, subtopic_ko)`` 를 가진 active 이웃 수.
        - ``topic_en``/``subtopic_en`` 은 해당 그룹에서 **처음 만난** 이웃 값을 보존(안정적).
        - 빈/None ``topic_ko`` 또는 dict 아닌 ``topic`` 이웃은 스킵(주제 미부여 엣지).
        - 정렬 ``weight desc → topic_ko asc → subtopic_ko asc``(결정적) 후 앞에서 top_n 절단.
    """
    # active-only 이웃 수집(status 고정 — 투영은 확정 관계만 반영, proposed/rejected 제외)
    neighbors = fetch_active_relations_for_asset(conn, asset_id=asset_id, status="active")

    counts: Counter[tuple[str, Any]] = Counter()
    # 그룹별 en 표기는 처음 본 이웃 값으로 고정(결정적·안정적)
    en_by_key: dict[tuple[str, Any], tuple[Any, Any]] = {}

    for nb in neighbors:
        topic = nb.get("topic")
        if not isinstance(topic, dict):
            continue  # topic 미부여(None) 또는 비정상 형상 → 스킵
        topic_ko = topic.get("topic_ko")
        if not topic_ko:  # 빈 문자열·None → 주제 없음으로 간주, 스킵
            continue
        subtopic_ko = topic.get("subtopic_ko")
        key = (topic_ko, subtopic_ko)
        counts[key] += 1
        if key not in en_by_key:  # 첫 등장 이웃의 en 표기 보존
            en_by_key[key] = (topic.get("topic_en"), topic.get("subtopic_en"))

    def _sort_key(item: tuple[tuple[str, Any], int]):
        (topic_ko, subtopic_ko), weight = item
        # weight 내림차순은 음수로, topic_ko/subtopic_ko 오름차순. None subtopic 은 "" 로 안정 비교.
        return (-weight, topic_ko, subtopic_ko or "")

    ranked = sorted(counts.items(), key=_sort_key)

    out: list[dict] = []
    for (topic_ko, subtopic_ko), weight in ranked[:top_n]:
        topic_en, subtopic_en = en_by_key[(topic_ko, subtopic_ko)]
        out.append(
            {
                "topic_ko": topic_ko,
                "subtopic_ko": subtopic_ko,
                "topic_en": topic_en,
                "subtopic_en": subtopic_en,
                "weight": weight,
            }
        )
    return out
