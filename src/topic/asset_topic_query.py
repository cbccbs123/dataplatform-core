"""자산 자기주제 정본 **조회(read)** — 주제 패싯·같은주제 그룹·미분류 목록 (spec 065 · 077 코어 분리).

분류(``classify_asset_topic``·write·LLM)는 파이프라인 레포(``processing.classify.asset_topic``)에 두고, 여기는
백엔드·검색이 공유하는 **read seam** 만 담는다 — 077 레포 분리에서 "파이프라인=분류 / 백엔드=read" 런타임
소유가 갈리므로 read 를 코어(config·database·file 만 의존)로 승격했다. ``asset_topic`` 테이블(v299) 조인으로
주제 트리·같은주제·미분류를 파생한다(관계 파이프라인 불변). ``find_same_topic_groups`` 의 already_linked
EXISTS 는 ``graph_query`` 대칭 엣지 규칙(양방향)을 따른다(순진한 단방향 ``WHERE src_node=X`` 금지).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from psycopg.rows import dict_row

# 표시용 파일명(아카이브 asset_id 프리픽스 제거) 단일 출처 — archiver 프리픽스 생성부와 대칭(065 T605).
from src.config.filename_util import display_file_name


def fetch_asset_topic(conn, asset_id) -> list[dict]:
    """자기주제 정본 읽기(T204) — 구 ``project_asset_topics`` 출력 형상과 동일(소비처 무변경 스왑).

    행 있으면 ``[{topic_ko, subtopic_ko, topic_en, subtopic_en, weight:1}]``, 없으면 ``[]``.
    (행 부재 = 주제 미부여 → 소비처는 현행 "topics 없음"과 동일 경로.) 조회행 id 는 str 관례를
    따르나 이 함수는 라벨만 반환한다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT topic_ko, subtopic_ko, topic_en, subtopic_en
            FROM asset_topic
            WHERE asset_id = %s
            """,
            (asset_id,),
        )
        row = cur.fetchone()
    if row is None:
        return []
    return [
        {
            "topic_ko": str(row["topic_ko"]) if row["topic_ko"] is not None else None,
            "subtopic_ko": row["subtopic_ko"],
            "topic_en": row["topic_en"],
            "subtopic_en": row["subtopic_en"],
            "weight": 1,
        }
    ]


# 같은-주제 그룹 후보 조회 — asset_topic 조인(자기주제 정본)으로 같은 (topic, subtopic) 자산 회수.
# already_linked = 대상 자산과 후보 자산 사이 active 엣지 존재 여부. graph_query 대칭 엣지 규칙을 따라
#   양방향(EXISTS: src=대상∧dst=후보 OR src=후보∧dst=대상)으로 판정한다 — 대칭 엣지는 캐논 순서
#   단일 행으로 저장되므로 순진한 단방향 WHERE 는 접힌 엣지를 누락한다(CLAUDE.md 규칙).
# 도메인 제외 없음(2026-07-23 전면 제거) — 의료 특수 트랙 미운용. 후보 자산은 도메인 무관 균일 취급.
_ALREADY_LINKED_EXISTS = """
       EXISTS (
           SELECT 1 FROM graph_edge ge
           JOIN node sn ON sn.node_id = ge.src_node AND sn.node_kind = 'asset'
           JOIN node dn ON dn.node_id = ge.dst_node AND dn.node_kind = 'asset'
           WHERE ge.status = 'active'
             AND ((sn.asset_id = %s AND dn.asset_id = at.asset_id)
               OR (sn.asset_id = at.asset_id AND dn.asset_id = %s))
       ) AS already_linked"""


def find_same_topic_groups(
    conn,
    asset_id,
    *,
    max_topics: int = 12,
    max_subtopics_per_topic: int = 12,
    max_assets_per_subtopic: int = 8,
) -> list[dict]:
    """대상 자산과 같은 주제의 다른 자산을 자기주제 정본 조인으로 묶는다(T204·구 형상 동일).

    매칭 규칙(구 find_topic_neighbor_groups 규칙 계승):
      - 대상의 (topic_ko, subtopic_ko) 쌍에서 **subtopic 이 있으면 같은 쌍**(topic AND subtopic)을
        가진 자산만 매칭 — 굵은 상위주제 희석 방지(정밀 관련).
      - 대상 subtopic 이 None 이면(가진 신호가 topic 뿐) **topic 단독 매칭** — 그 topic 의 자산을
        각자의 subtopic 버킷으로 모은다.

    Args:
        conn: DB 연결.
        asset_id: 기준 자산. **주제가 없으면 후보 조회조차 하지 않고** 빈 목록을 돌려준다.
        max_topics: 담을 주제 수 상한.
        max_subtopics_per_topic: 주제당 하위주제 수 상한.
        max_assets_per_subtopic: 하위주제당 자산 수 상한. **상한은 표시용 절단이고**,
            함께 담는 개수는 절단 전 실제 수다(화면이 "더 있음"을 알 수 있게).

    Returns:
        주제 → 하위주제 → 자산 3단 목록. 정렬은 전부 고정되며(개수 내림차순, 동수는 이름순),
        이름 없는 하위주제는 **항상 마지막**에 온다(화면에서 '기타'로 표시된다).
    """
    # 1) 대상 자산 자기주제 정본(asset_id PK → 0/1행). 없으면 미부여 → 빈 결과.
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT topic_ko, subtopic_ko FROM asset_topic WHERE asset_id = %s",
            (asset_id,),
        )
        target = cur.fetchone()
    if target is None or not target.get("topic_ko"):
        return []

    topic_ko = str(target["topic_ko"])
    target_sub = target.get("subtopic_ko")  # None 이면 topic 단독 매칭

    # 2) 같은 주제(subtopic 있으면 같은 쌍)의 다른 자산 + already_linked(양방향 EXISTS).
    #    바인딩 순서 = SQL 텍스트상 %s 등장 순서: EXISTS 의 대상 asset_id 2개 → 제외 대상 →
    #    topic_ko → (subtopic 있으면) subtopic_ko.
    sql = f"""
        SELECT at.asset_id, at.topic_ko, at.subtopic_ko,
               a.fs_path, a.modality,
{_ALREADY_LINKED_EXISTS}
        FROM asset_topic at
        JOIN asset a ON a.asset_id = at.asset_id
        WHERE at.asset_id <> %s
          AND at.topic_ko = %s
    """
    params: list[Any] = [asset_id, asset_id, asset_id, topic_ko]
    if target_sub is not None:
        sql += "          AND at.subtopic_ko = %s\n"
        params.append(target_sub)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()

    # 3) topic → {assets(상위 distinct 집합), subs: subtopic_ko → {asset_id → 표시필드}}.
    groups: dict[str, dict] = {}
    for r in rows:
        tk = str(r["topic_ko"]) if r["topic_ko"] is not None else None
        if not tk:
            continue
        sub = r.get("subtopic_ko") or None
        aid = str(r["asset_id"])
        g = groups.setdefault(tk, {"assets": set(), "subs": {}})
        g["assets"].add(aid)  # 상위 distinct(하위 걸침 무관 1회)
        bucket = g["subs"].setdefault(sub, {})
        bucket.setdefault(
            aid,
            {
                "file_name": display_file_name(r.get("fs_path")),
                "modality": r.get("modality"),
                "already_linked": bool(r.get("already_linked")),
            },
        )

    out: list[dict] = []
    for tk, g in groups.items():
        subtopics = []
        for sub, bucket in g["subs"].items():
            ordered = sorted(bucket.items(), key=lambda kv: kv[0])  # asset_id asc(결정적)
            assets = [
                {
                    "asset_id": aid,
                    "file_name": e["file_name"],
                    "modality": e["modality"],
                    "already_linked": e["already_linked"],
                }
                for aid, e in ordered[:max_assets_per_subtopic]
            ]
            subtopics.append(
                {"subtopic_ko": sub, "asset_count": len(bucket), "assets": assets}
            )
        # 하위주제 정렬: 이름있는 것 먼저(asset_count desc → subtopic_ko asc), None(기타)은 마지막·절단.
        subtopics.sort(
            key=lambda s: (s["subtopic_ko"] is None, -s["asset_count"], s["subtopic_ko"] or "")
        )
        out.append(
            {
                "topic_ko": tk,
                "asset_count": len(g["assets"]),
                "subtopics": subtopics[:max_subtopics_per_topic],
            }
        )
    # 결정성(헌법 3조): 주제 asset_count desc → topic_ko asc.
    out.sort(key=lambda o: (-o["asset_count"], o["topic_ko"]))
    return out[:max_topics]


# ── 주제 패싯·브라우즈(FR-402) — 자기주제 정본 조인 ──────────────────────────────
# 구 ``topic_query.list_topics``/``assets_in_topic`` 를 정본(``asset_topic``) 기준으로 이식한다.
# 응답 계약(필드명·정렬)은 구 함수와 동일 → 포탈 /topics·/topics/{topic} 무변경 스왑(FR-402).
# 정본은 자산당 1행(asset_id PK)이라 구 이웃-엣지 투영의 "양끝 자산 중복카운트" 문제가 원천 소거된다.
# 도메인 제외 없음(2026-07-23 전면 제거) — 의료 특수 트랙 미운용. 후보 자산은 도메인 무관 균일 취급.

# (topic_ko, subtopic_ko) 별 자산 집계용 — 빈 topic_ko 는 조회 단계에서 배제.
_LIST_TOPICS_SQL = """
SELECT at.topic_ko, at.subtopic_ko, at.asset_id
FROM asset_topic at
JOIN asset a ON a.asset_id = at.asset_id
WHERE COALESCE(at.topic_ko, '') <> ''
"""

# 주제별 자산 페이징용 — subtopic 필터는 호출 시 append.
_ASSETS_IN_TOPIC_SQL = """
SELECT at.asset_id, a.fs_uri, a.fs_path, a.modality,
       m.ext_meta->'keywords' AS keywords, m.ext_meta->'labels' AS labels
FROM asset_topic at
JOIN asset a ON a.asset_id = at.asset_id
LEFT JOIN asset_metadata m ON m.asset_id = at.asset_id
WHERE at.topic_ko = %s
"""


def list_topics(conn) -> list[dict]:
    """자기주제 정본의 주제 패싯 목록(구 ``topic_query.list_topics`` 형상·FR-402).

    Returns:
        ``[{topic_ko, subtopic_ko|None, asset_count, topic_asset_count}]`` — ``(topic_ko,
        subtopic_ko)`` 조합별 distinct 자산 수(``asset_count``). 빈 topic_ko 는 제외, 빈 subtopic_ko 는
        ``None`` 으로 정규화. 정렬 ``topic_ko asc → subtopic_ko asc``(None 은 "" 로 최상단·결정적).
        ``topic_asset_count`` = ``topic_ko`` 주제 전체의 distinct 자산 수(하위주제 합산 아님 —
        프론트 중복카운트 방지, 057 FR-105). 같은 ``topic_ko`` 의 모든 행은 동일 값을 갖는다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_LIST_TOPICS_SQL)
        rows = cur.fetchall()

    # (topic_ko, subtopic_ko) → distinct 자산 집합. topic_assets: topic_ko → 주제 전체 distinct 자산.
    groups: dict[tuple[str, Any], set[str]] = {}
    topic_assets: dict[str, set[str]] = {}
    for r in rows:
        topic_ko = r["topic_ko"]
        if not topic_ko:  # COALESCE 로 조회 단계에서 걸러지지만 방어적 스킵.
            continue
        sub = r["subtopic_ko"] or None  # "" 또는 None → None 정규화
        aid = str(r["asset_id"])
        groups.setdefault((topic_ko, sub), set()).add(aid)
        topic_assets.setdefault(topic_ko, set()).add(aid)

    out = [
        {"topic_ko": ko, "subtopic_ko": sub, "asset_count": len(assets),
         "topic_asset_count": len(topic_assets[ko])}
        for (ko, sub), assets in groups.items()
    ]
    out.sort(key=lambda o: (o["topic_ko"], o["subtopic_ko"] or ""))
    return out


def _asset_list_item(r: dict) -> dict:
    """조회 행을 파일 목록 항목으로 바꾼다(주제별·미분류 목록이 공유).

    키워드·라벨을 **상위 일부만** 담는다 — 목록에는 미리보기로 충분하고, 전부 담으면 자산 수만큼
    응답이 커진다.

    Args:
        r: 조회 행. 라벨은 ``{label, score}`` 형태를 가정하되 **문자열 원소도 받는다**
            (형태가 섞여 들어와도 죽지 않게).

    Returns:
        목록 항목 dict.
    """
    return {
        "asset_id": str(r["asset_id"]),
        "fs_uri": r["fs_uri"],
        "file_name": display_file_name(r["fs_path"]),
        "modality": r["modality"],
        "keywords": [str(k) for k in (r.get("keywords") or []) if k][:8],
        "labels": [
            (it.get("label") if isinstance(it, dict) else it)
            for it in (r.get("labels") or [])
            if it
        ][:5],
    }


def assets_in_topic(
    conn,
    *,
    topic_ko: str,
    subtopic_ko: str | None = None,
    unassigned_only: bool = False,
    modality: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """특정 주제(자기주제 정본)에 속한 자산을 페이징 조회(구 ``topic_query.assets_in_topic`` 형상).

    Args:
        topic_ko: 대주제(정확 일치).
        subtopic_ko: 세부주제(주면 추가 필터, None 이면 topic_ko 하위 전체).
        unassigned_only: True 면 **'기타'(subtopic 미부여)만** — ``subtopic_ko IS NULL`` 필터.
            ``subtopic_ko=None``(=필터 없음·topic 전체)과 구분하는 명시 플래그. True 면 subtopic_ko 지정보다 우선.
        modality: 주면 그 모달리티(text/image/video/audio) 자산만 반환(파일탐색기 모달리티 폴더 진입).
            None 이면 전체. ``modality_counts`` 는 **modality 필터와 무관하게** 이 주제/하위의 전체 모달리티
            분포다(모달리티 폴더 카운트 — 폴더를 그린 뒤 클릭 시 modality 로 좁힌다).
        limit/offset: 페이징.

    Returns:
        ``{rows:[{asset_id(str), fs_uri, file_name, modality}], total, modality_counts:{<modality>:n}}``.
        ``total`` 은 (modality 필터 적용 후) 페이징 전 distinct 자산 수, ``modality_counts`` 는 필터 전
        전체 분포, ``rows`` 는 ``asset_id asc`` 결정적 정렬 후 페이징. ``file_name`` 은 ``fs_path`` basename.
    """
    sql = _ASSETS_IN_TOPIC_SQL
    params: list[Any] = [topic_ko]
    # subtopic_ko=None 은 '필터 없음(topic 전체)'이라 '기타'(미부여)만 좁힐 수 없다 → unassigned_only 로
    # IS NULL 명시 필터. unassigned_only 가 subtopic 지정보다 우선. (modality 는 아래 파이썬에서 필터 —
    # modality_counts 를 필터 전 전체 분포로 먼저 세기 위함.)
    if unassigned_only:
        sql = sql + "  AND at.subtopic_ko IS NULL\n"
    elif subtopic_ko is not None:
        sql = sql + "  AND at.subtopic_ko = %s\n"
        params.append(subtopic_ko)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()

    # 정본은 자산당 1행이지만 관례상 asset_id → 식별 필드로 dedupe(멱등·결정적).
    assets: dict[str, dict] = {}
    for r in rows:
        aid = str(r["asset_id"])
        if aid not in assets:
            assets[aid] = _asset_list_item(r)

    ordered = sorted(assets.values(), key=lambda a: a["asset_id"])  # asset_id asc 결정적
    # modality_counts = 필터 전 전체 분포(모달리티 폴더 카운트). 그 뒤 modality 로 rows 만 좁힌다.
    modality_counts = dict(Counter(a["modality"] for a in ordered))
    if modality is not None:
        ordered = [a for a in ordered if a["modality"] == modality]
    total = len(ordered)
    return {"rows": ordered[offset : offset + limit], "total": total, "modality_counts": modality_counts}


# ── 미분류(주제 미부여) 조회 — 파일탐색기 '미분류' 폴더(전수 포함) ──────────────────────
# list_topics/assets_in_topic 은 asset_topic 조인이라 **주제 정본이 없는** 자산(분류 실패·무내용 등)을
# 누락한다. 자산목록을 파일시스템처럼 '빠짐없이' 보이려면 이들을 별도 회수해야 한다(주제 트리의 최상위
# '미분류' 폴더). registered 만(수집 중/실패 제외). 도메인 제외 없음(2026-07-23 전면 제거).
_ASSETS_UNCLASSIFIED_SQL = """
SELECT a.asset_id, a.fs_uri, a.fs_path, a.modality,
       m.ext_meta->'keywords' AS keywords, m.ext_meta->'labels' AS labels
FROM asset a
LEFT JOIN asset_topic at ON at.asset_id = a.asset_id
LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id
WHERE a.status = 'registered'
  AND at.asset_id IS NULL
"""


def assets_unclassified(conn, *, modality: str | None = None, limit: int = 50, offset: int = 0) -> dict:
    """주제 미부여(자기주제 정본 없음) 자산을 페이징 조회 — 파일탐색기의 '미분류' 폴더용.

    ``asset_topic`` 행이 없는 registered 자산(도메인 무관 — 2026-07-23 도메인 제외 전면 제거·의료 포함 균일).
    주제 트리(``list_topics``)는 asset_topic 조인이라
    이들을 누락하므로, 전수 조회(빠짐없이)를 위해 별도로 회수한다.

    Args:
        modality: 주면 그 모달리티만 반환(미분류 폴더 안 모달리티 폴더 진입). ``modality_counts`` 는 필터
            무관 전체 분포(모달리티 폴더 카운트). None 이면 전체.
        limit/offset: 페이징.

    Returns:
        ``{rows:[{asset_id(str), fs_uri, file_name, modality}], total, modality_counts:{<modality>:n}}``
        — ``total`` 은 (modality 필터 후) 미분류 자산 수, ``modality_counts`` 는 필터 전 전체 분포,
        ``rows`` 는 ``asset_id asc`` 결정적 정렬 후 페이징.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_ASSETS_UNCLASSIFIED_SQL)
        rows = cur.fetchall()
    assets: dict[str, dict] = {}
    for r in rows:
        aid = str(r["asset_id"])
        if aid not in assets:
            assets[aid] = _asset_list_item(r)
    ordered = sorted(assets.values(), key=lambda a: a["asset_id"])  # asset_id asc 결정적
    modality_counts = dict(Counter(a["modality"] for a in ordered))  # 필터 전 전체 분포(모달리티 폴더)
    if modality is not None:
        ordered = [a for a in ordered if a["modality"] == modality]
    total = len(ordered)
    return {"rows": ordered[offset : offset + limit], "total": total, "modality_counts": modality_counts}
