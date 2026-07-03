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

import os
from collections import Counter
from typing import Any

from psycopg.rows import dict_row

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


# ---------------------------------------------------------------------------
# 주제 탐색 seam (056 G2 · FR-401~403)
#
# graph_query 조인 스타일 미러링: graph_edge → node(src/dst; node_kind='asset') →
# asset(양끝) 로 자산을 해소한다. 대상 주제 실은 active 엣지의 양끝을 후보/집계 대상으로 쓴다.
#
# 의료(PHI) 제외 (헌법 10조) — ``review._build_review_where`` 와 **동일 조건 재사용**:
#   양끝 asset(sa=src, da=dst)에 ``domain_label IS DISTINCT FROM 'medical'`` 을 건다.
#   NULL 도메인 노출 방지 위해 ``= 'medical'`` 이 아니라 ``IS DISTINCT FROM`` 을 쓴다
#   (``= 'medical'`` 은 NULL 을 놓친다). 어느 한 끝이라도 의료면 엣지 전체를 제외한다.
#
# 투영 = active-only (plan Global Constraints) — status 는 'active' 리터럴로 고정(사용자 입력 아님).
# topic 술어는 표현식 인덱스(v294 ix_graph_edge_topic_ko/subtopic_ko) 친화형
#   ``topic->>'topic_ko' = %s`` / ``= ANY(%s)`` — 값은 전부 %s 바인딩(f-string 값 주입 금지·인젝션 0).
# 조회행 계약(graph_query 관례): 반환 asset_id 는 항상 ``str()``.
# ---------------------------------------------------------------------------

# 양끝 자산 해소 조인(graph_query _FETCH_RELATIONS_SQL 스타일) + review 의료 제외 2조건.
_TOPIC_JOIN = """
FROM graph_edge ge
JOIN node sn ON sn.node_id = ge.src_node AND sn.node_kind = 'asset'
JOIN node dn ON dn.node_id = ge.dst_node AND dn.node_kind = 'asset'
JOIN asset sa ON sa.asset_id = sn.asset_id
JOIN asset da ON da.asset_id = dn.asset_id
"""

# active-only + 의료 제외(양끝) — 세 함수 공통 전제.
_ACTIVE_MEDICAL_WHERE = """
WHERE ge.status = 'active'
  AND sa.domain_label IS DISTINCT FROM 'medical'
  AND da.domain_label IS DISTINCT FROM 'medical'
"""

# 대상 주제(topic_ko 집합)를 실은 active 엣지의 양끝 자산 수집.
# 057 FR-103: 후보 표시필드(modality·file_name) 하향 — _TOPIC_JOIN 이 이미 건 asset sa/da 에서
#   양끝 modality·fs_path 를 함께 SELECT(조인 재사용·재구현 없음). basename 은 파이썬에서 파생.
# 057-후속: subtopic_ko 도 SELECT — 같은주제 그룹의 하위주제 중첩용(평면 find_topic_neighbors 는 미사용·무영향).
_NEIGHBOR_SQL = f"""
SELECT sn.asset_id AS src_asset, dn.asset_id AS dst_asset,
       sa.modality AS src_modality, sa.fs_path AS src_fs_path,
       da.modality AS dst_modality, da.fs_path AS dst_fs_path,
       ge.topic->>'topic_ko' AS topic_ko,
       ge.topic->>'subtopic_ko' AS subtopic_ko
{_TOPIC_JOIN}{_ACTIVE_MEDICAL_WHERE}
  AND ge.topic->>'topic_ko' = ANY(%s)
"""

# 주제 목록 집계용 — 양끝 자산 + (topic_ko, subtopic_ko). 빈 topic_ko 는 조회 단계에서 배제.
_LIST_TOPICS_SQL = f"""
SELECT sn.asset_id AS src_asset, dn.asset_id AS dst_asset,
       ge.topic->>'topic_ko' AS topic_ko,
       ge.topic->>'subtopic_ko' AS subtopic_ko
{_TOPIC_JOIN}{_ACTIVE_MEDICAL_WHERE}
  AND COALESCE(ge.topic->>'topic_ko', '') <> ''
"""

# 주제별 자산 페이징용 — 양끝 자산 + 식별(fs_uri·fs_path). subtopic 필터는 호출 시 append.
_ASSETS_IN_TOPIC_SQL = f"""
SELECT sn.asset_id AS src_asset, sa.fs_uri AS src_fs_uri, sa.fs_path AS src_fs_path,
       dn.asset_id AS dst_asset, da.fs_uri AS dst_fs_uri, da.fs_path AS dst_fs_path
{_TOPIC_JOIN}{_ACTIVE_MEDICAL_WHERE}
  AND ge.topic->>'topic_ko' = %s
"""


def find_topic_neighbors(conn, *, asset_id: str, top_k: int = 20) -> list[dict]:
    """``asset_id`` 와 주제를 1개 이상 공유하는 다른 자산을 찾는다(같은-주제 탐색).

    Args:
        conn: DB 연결.
        asset_id: 탐색 관점 자산.
        top_k: 반환 상한(기본 20).

    Returns:
        ``[{asset_id(str), shared_topics:[topic_ko...], overlap_weight:int, already_linked:bool,
        file_name(str·fs_path basename·FR-103), modality(str·FR-103)}]``.
        - ``overlap_weight`` = 그 자산이 (대상 주제를 실은) active 엣지에 참여한 횟수(엣지당 1).
        - ``shared_topics`` = 대상과 공유하는 topic_ko 집합(정렬).
        - ``already_linked`` = 그 자산이 대상의 **직접 관계 이웃**인지
          (``fetch_active_relations_for_asset(asset_id)`` 이웃 집합 포함 여부).
        - 정렬 ``overlap_weight desc → asset_id asc``(결정적) 후 top_k 절단. 대상 자산 자신 제외.

    대상 주제가 없으면(직접 이웃 topic 미부여) 빈 리스트(불필요한 DB 조회도 하지 않음).
    """
    # 1) 대상 주제 = 대상의 active 이웃 엣지 topic 투영(project_asset_topics seam 재사용).
    target_topics = project_asset_topics(conn, asset_id=asset_id)
    topic_kos = sorted({t["topic_ko"] for t in target_topics if t.get("topic_ko")})
    if not topic_kos:
        return []

    # 2) already_linked 판정용 대상 직접 이웃 집합(같은 active read seam 재사용; asset_id 는 str).
    linked = {nb["asset_id"] for nb in fetch_active_relations_for_asset(conn, asset_id=asset_id)}
    target_str = str(asset_id)

    # 3) 대상 주제를 실은 active·의료제외 엣지의 양끝 자산 수집.
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_NEIGHBOR_SQL, (topic_kos,))
        rows = cur.fetchall()

    # 4) 후보 자산별 overlap_weight(엣지 참여 수)·shared_topics(공유 topic_ko) 집계. 대상 자신 스킵.
    #    FR-103: 각 끝점의 modality·fs_path 를 함께 나른다(같은 자산은 어느 엣지에서 보든 같은 값이라
    #    처음 만난 값 보존 — assets_in_topic 의 first-wins 관례와 동일·결정적).
    agg: dict[str, dict] = {}
    for r in rows:
        topic_ko = r["topic_ko"]
        for endpoint, modality, fs_path in (
            (r["src_asset"], r["src_modality"], r["src_fs_path"]),
            (r["dst_asset"], r["dst_modality"], r["dst_fs_path"]),
        ):
            aid = str(endpoint)
            if aid == target_str:
                continue
            entry = agg.setdefault(
                aid,
                {"overlap_weight": 0, "shared": set(),
                 "modality": modality, "file_name": os.path.basename(fs_path or "")},
            )
            entry["overlap_weight"] += 1
            if topic_ko:
                entry["shared"].add(topic_ko)

    out = [
        {
            "asset_id": aid,
            "shared_topics": sorted(e["shared"]),
            "overlap_weight": e["overlap_weight"],
            "already_linked": aid in linked,
            # FR-103(057): 후보 표시필드 하향(하위호환 필드 추가·상세 진입 폴백 제거).
            "file_name": e["file_name"],
            "modality": e["modality"],
        }
        for aid, e in agg.items()
    ]
    # 결정성(헌법 3조): overlap_weight desc → asset_id asc.
    out.sort(key=lambda o: (-o["overlap_weight"], o["asset_id"]))
    return out[:top_k]


def find_topic_neighbor_groups(
    conn,
    *,
    asset_id: str,
    max_topics: int = 12,
    max_subtopics_per_topic: int = 12,
    max_assets_per_subtopic: int = 8,
) -> list[dict]:
    """``asset_id`` 와 공유하는 같은-주제 이웃을 **주제→하위주제 2단 중첩**으로 묶는다(같은주제 탐색 UX·057-후속).

    ``find_topic_neighbors`` 가 자산 단위 평면 목록이라면, 이 함수는 **무슨 주제·무슨 하위주제로 같은지**가
    바로 보이도록 ``topic_ko → subtopic_ko`` 로 중첩한다(검색 패싯 ``list_topics`` 와 동일한 2단 구조).
    평면 목록의 ``overlap_weight``(엣지 참여수)를 "공유 주제 N개"로 오라벨하던 혼선을 구조로 제거하고,
    스포츠처럼 광범위한 상위주제(예: 113건)를 마라톤·스키·축구… 하위주제로 드릴다운되게 한다.
    한 자산이 여러 (하위)주제를 공유하면 각 그룹에 모두 등장한다(주제 브라우즈 렌즈).

    Returns:
        ``[{topic_ko, asset_count, subtopics:[{subtopic_ko, asset_count,
        assets:[{asset_id(str), file_name, modality, already_linked}]}]}]``.
        - 상위 ``asset_count`` = 그 주제를 대상과 공유하는(대상 제외) **distinct** 자산 수(하위 합과 다를 수 있음
          — 한 자산이 여러 하위주제에 걸치면 상위는 1회, 하위는 각각 계수).
        - 하위 ``asset_count`` = 그 하위주제의 distinct 자산 수(``assets`` 절단 전 실수). ``subtopic_ko`` 는
          하위주제 미부여 엣지면 ``None``(프론트에서 "기타/하위주제 없음"으로 표기).
        - ``assets`` = 하위주제 내 자산(엣지 참여수 desc → asset_id asc·결정적), ``max_assets_per_subtopic`` 절단.
        - 정렬(결정적): 주제 ``asset_count desc → topic_ko asc``(``max_topics`` 절단) · 하위주제는 이름있는
          것 먼저(``asset_count desc → subtopic_ko asc``)·None(기타)은 항상 마지막(``max_subtopics_per_topic`` 절단).
        - ``already_linked`` = 그 자산이 대상의 직접 관계 이웃인지(``fetch_active_relations_for_asset`` 집합).
    대상 주제가 없으면 빈 리스트(DB 조회도 안 함). active·의료제외는 ``find_topic_neighbors`` 와 동일 SQL 재사용.
    """
    # 1) 대상 주제 + 직접 이웃 집합 — find_topic_neighbors 와 동일 seam(중복 재발명 없음).
    target_topics = project_asset_topics(conn, asset_id=asset_id)
    topic_kos = sorted({t["topic_ko"] for t in target_topics if t.get("topic_ko")})
    if not topic_kos:
        return []
    linked = {nb["asset_id"] for nb in fetch_active_relations_for_asset(conn, asset_id=asset_id)}
    target_str = str(asset_id)

    # 2) 대상 주제를 실은 active·의료제외 엣지 양끝 자산 수집(_NEIGHBOR_SQL 공유·subtopic_ko 포함).
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_NEIGHBOR_SQL, (topic_kos,))
        rows = cur.fetchall()

    # 3) topic_ko → {assets(상위 distinct 집합), subs: subtopic_ko → {asset_id → {표시필드, weight}}}.
    #    같은 자산은 어느 엣지에서 보든 같은 표시값이라 first-wins(assets_in_topic 관례·결정적).
    groups: dict[str, dict] = {}
    for r in rows:
        topic_ko = r["topic_ko"]
        if not topic_ko:
            continue
        subtopic_ko = r["subtopic_ko"] or None
        for endpoint, modality, fs_path in (
            (r["src_asset"], r["src_modality"], r["src_fs_path"]),
            (r["dst_asset"], r["dst_modality"], r["dst_fs_path"]),
        ):
            aid = str(endpoint)
            if aid == target_str:
                continue
            g = groups.setdefault(topic_ko, {"assets": set(), "subs": {}})
            g["assets"].add(aid)  # 상위 distinct(하위 걸침 무관 1회)
            bucket = g["subs"].setdefault(subtopic_ko, {})
            entry = bucket.setdefault(
                aid,
                {"file_name": os.path.basename(fs_path or ""), "modality": modality,
                 "already_linked": aid in linked, "weight": 0},
            )
            entry["weight"] += 1

    out = []
    for topic_ko, g in groups.items():
        subtopics = []
        for subtopic_ko, bucket in g["subs"].items():
            # 하위주제 내 자산: 엣지 참여수 desc → asset_id asc(결정적) 후 절단.
            ordered = sorted(bucket.items(), key=lambda kv: (-kv[1]["weight"], kv[0]))
            assets = [
                {"asset_id": aid, "file_name": e["file_name"], "modality": e["modality"],
                 "already_linked": e["already_linked"]}
                for aid, e in ordered[:max_assets_per_subtopic]
            ]
            subtopics.append(
                {"subtopic_ko": subtopic_ko, "asset_count": len(bucket), "assets": assets}
            )
        # 하위주제 정렬: 이름있는 하위주제 먼저(asset_count desc → subtopic_ko asc), None(기타)은 항상 마지막·절단.
        subtopics.sort(key=lambda s: (s["subtopic_ko"] is None, -s["asset_count"], s["subtopic_ko"] or ""))
        out.append({
            "topic_ko": topic_ko,
            "asset_count": len(g["assets"]),
            "subtopics": subtopics[:max_subtopics_per_topic],
        })
    # 결정성(헌법 3조): 주제 asset_count desc → topic_ko asc.
    out.sort(key=lambda o: (-o["asset_count"], o["topic_ko"]))
    return out[:max_topics]


def list_topics(conn) -> list[dict]:
    """active·의료제외 엣지의 주제 패싯 목록.

    Returns:
        ``[{topic_ko, subtopic_ko|None, asset_count, topic_asset_count}]`` — ``(topic_ko,
        subtopic_ko)`` 조합별 **distinct 양끝 자산 수**(``asset_count``). 빈 topic_ko 는 제외,
        빈 subtopic_ko 는 ``None`` 으로 정규화. 정렬 ``topic_ko asc → subtopic_ko asc``
        (None 은 "" 로 최상단·결정적).

        057 FR-105 — ``topic_asset_count``(하위호환 필드 추가): ``topic_ko`` **주제 전체**의
        distinct 양끝 자산 수. 한 자산이 여러 하위주제에 걸치면 하위주제 ``asset_count`` 합은
        중복카운트가 되므로(web B2 실버그), 프론트가 합산하지 않고 이 주제 레벨 distinct 를 그대로
        쓰게 한다. 같은 ``topic_ko`` 의 모든 행은 동일한 ``topic_asset_count`` 를 갖는다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_LIST_TOPICS_SQL)
        rows = cur.fetchall()

    # (topic_ko, subtopic_ko) → distinct 자산 집합. 양끝 자산 모두 그 주제에 참여한 것으로 본다.
    # topic_assets: topic_ko → 주제 전체 distinct 자산 집합(하위주제 합산 아님·FR-105 중복카운트 방지).
    groups: dict[tuple[str, Any], set[str]] = {}
    topic_assets: dict[str, set[str]] = {}
    for r in rows:
        topic_ko = r["topic_ko"]
        if not topic_ko:  # COALESCE 로 조회 단계에서 걸러지지만 방어적 스킵.
            continue
        sub = r["subtopic_ko"] or None  # "" 또는 None → None 정규화
        src, dst = str(r["src_asset"]), str(r["dst_asset"])
        assets = groups.setdefault((topic_ko, sub), set())
        assets.add(src)
        assets.add(dst)
        topic_set = topic_assets.setdefault(topic_ko, set())
        topic_set.add(src)
        topic_set.add(dst)

    out = [
        {"topic_ko": ko, "subtopic_ko": sub, "asset_count": len(assets),
         "topic_asset_count": len(topic_assets[ko])}
        for (ko, sub), assets in groups.items()
    ]
    out.sort(key=lambda o: (o["topic_ko"], o["subtopic_ko"] or ""))
    return out


def assets_in_topic(
    conn, *, topic_ko: str, subtopic_ko: str | None = None, limit: int = 50, offset: int = 0
) -> dict:
    """특정 주제(active·의료제외 엣지)에 참여하는 자산을 페이징 조회.

    Args:
        topic_ko: 대주제(정확 일치).
        subtopic_ko: 세부주제(주면 추가 필터, None 이면 topic_ko 하위 전체).
        limit/offset: 페이징.

    Returns:
        ``{rows:[{asset_id(str), fs_uri, file_name}], total}``. ``total`` 은 페이징 전 distinct
        자산 수. ``rows`` 는 ``asset_id asc`` 결정적 정렬 후 ``[offset:offset+limit]``.
        ``file_name`` 은 ``fs_path`` basename(review.py 관례 일치).
    """
    sql = _ASSETS_IN_TOPIC_SQL
    params: list[Any] = [topic_ko]
    if subtopic_ko is not None:
        # 세부주제 필터도 표현식 인덱스(ix_graph_edge_subtopic_ko) 친화 술어 + %s 바인딩.
        sql = sql + "  AND ge.topic->>'subtopic_ko' = %s\n"
        params.append(subtopic_ko)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()

    # 양끝 자산을 distinct 수집(asset_id → 식별 필드). 중복 엣지·양방향 참여를 dedupe.
    assets: dict[str, dict] = {}
    for r in rows:
        for aid_key, uri_key, path_key in (
            ("src_asset", "src_fs_uri", "src_fs_path"),
            ("dst_asset", "dst_fs_uri", "dst_fs_path"),
        ):
            aid = str(r[aid_key])
            if aid not in assets:
                assets[aid] = {
                    "asset_id": aid,
                    "fs_uri": r[uri_key],
                    "file_name": os.path.basename(r[path_key] or ""),
                }

    ordered = sorted(assets.values(), key=lambda a: a["asset_id"])  # asset_id asc 결정적
    total = len(ordered)
    page = ordered[offset : offset + limit]
    return {"rows": page, "total": total}
