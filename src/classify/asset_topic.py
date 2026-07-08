"""자산 자기주제(aboutness) 분류 seam — 자기 내용에서 (topic, subtopic) 정본을 1회 확정(spec 065).

왜 이 seam 인가 (spec 065 §설계 원칙·ADR 2026-07-08)
    지금까지 자산 주제는 관계 이웃 엣지(``graph_edge.topic``)를 자산으로 투영해 만들었다
    (``topic_query.project_asset_topics``). 엣지 topic 은 관계 LLM 이 쌍(pairwise) 비교 중 붙이는
    라벨이라 상대 자산 쪽으로 치우치고, 어떤 이웃 엣지가 활성 임계를 넘느냐(이웃 운)에 따라 자산
    주제가 흔들려 오염됐다(농구 영상이 축구·배드민턴으로 노출). 065 는 "주제(무엇인가)"와
    "관계(어떻게 연결되나)"를 분리한다 — 주제를 **자산 자기 내용(summary/keywords/labels)에서
    1회 확정한 정본**(``asset_topic`` 테이블·v299)으로 둔다. 같은-주제 묶음·패싯은 전부 이 정본
    조인으로 파생한다(관계 파이프라인 불변).

하이브리드 판정 (FR-202)
    ① 자기 텍스트 구성(``build_self_text``) → ② 활성 채널 임베딩 kNN 으로 topic 층(닫힌 28) 후보
    top-k(``topic_candidates_for_self_text``·058 프리미티브 재사용) → ③ LLM(단일 seam·temp=0)이
    **닫힌 후보 집합에서 topic 1개 확정** + subtopic 생성(후보 밖이면 1회 재질의·재실패 시 미부여) →
    ④ subtopic 은 058 ``canonicalize_subtopic`` 으로 정규화(alias 캐시 동결 재사용) → ⑤ topic_en 은
    registry 정본 우선(``_lookup_topic_en``) → ⑥ ``asset_topic`` upsert(멱등·policy_version 기록).

헌법·불변식
    - **결정성(3조)**: temp=0 + 닫힌 topic 후보 + subtopic canonicalize 캐시 동결 + 멱등 upsert →
      같은 입력 같은 (topic, subtopic)(SC-05).
    - **LLM 단일 seam(6조)**: ``src.llm.client.complete_json``·``client=`` 주입.
    - **닫힌집합 검증(FR-203)**: LLM 이 후보 밖 topic 을 답하면 1회 재질의 후 실패 시 미부여(강제
      매핑 금지·환각 차단). 임베딩 kNN 후보가 비면(레지스트리 미시드) 분류 스킵.
    - **실패 격리(FR-204)**: 예외는 삼키지 않고 올린다 — 호출부(run_ingest)가 registered 를 유지한
      채 주제만 미부여로 격리한다.
    - **학습 0(3조)**: 임베딩 kNN·LLM zero-shot 전부 inference-only.

재사용 (058·graph_query)
    ``knn_topic_candidates``/``canonicalize_subtopic``/``_lookup_topic_en`` 는 058 정본을 **모듈
    상단에서 import** 해 그대로 쓴다(중복 구현 금지). 테스트가 이 위치를 patch 해 실 DB/LLM 없이
    분기만 검증할 수 있게 하기 위함이다. ``find_same_topic_groups`` 의 ``already_linked`` EXISTS 는
    ``graph_query`` 대칭 엣지 규칙(양방향)을 따른다(순진한 단방향 ``WHERE src_node=X`` 금지).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from psycopg.rows import dict_row

# 058 정본 프리미티브 재사용(모듈 상단 import = 테스트 patch 지점). 중복 구현 금지.
from src.relations.topic_canonicalize import (
    _lookup_topic_en,
    canonicalize_subtopic,
    knn_topic_candidates,
)

logger = logging.getLogger(__name__)

# 분류 정책 버전(FR-601) — 프롬프트/후보수 변경 시 증가시켜 재현성을 추적한다.
POLICY_VERSION = "asset_topic.v1"

# kNN topic 후보 기본 개수(닫힌 28 중 상위 k). 프롬프트에 넣을 후보 목록 크기.
_DEFAULT_TOPIC_K = 5


def build_self_text(
    summary: str | None, keywords: list | None, labels: list | None = None
) -> str:
    """자기 텍스트 구성 — 결정적 순서(summary → keywords → 상위 라벨). 전부 비면 ''(FR-201).

    - ``summary``: 요약 문자열(공백만이면 제외).
    - ``keywords``: 문자열 리스트(공백·None 원소 제외 후 공백 join).
    - ``labels``: image/video 제로샷 라벨 ``[{label, score}]`` 가정 — ``label`` 문자열만 순서대로
      사용(score 제외). 문자열 원소도 방어적으로 허용. 순서는 입력 순(제로샷 score desc)을 보존해
      재실행마다 동일(헌법 3조). None/빈 입력은 전부 안전하게 건너뛴다.
    """
    parts: list[str] = []

    if summary and str(summary).strip():
        parts.append(str(summary).strip())

    if keywords:
        kw = [str(k).strip() for k in keywords if k and str(k).strip()]
        if kw:
            parts.append(" ".join(kw))

    if labels:
        labs: list[str] = []
        for item in labels:
            # dict 형([{label, score}])이면 label 만, 문자열이면 그대로(방어적).
            label = item.get("label") if isinstance(item, dict) else item
            if label and str(label).strip():
                labs.append(str(label).strip())
        if labs:
            parts.append(" ".join(labs))

    return " ".join(parts).strip()


def topic_candidates_for_self_text(
    conn, self_text: str | None, *, k: int = _DEFAULT_TOPIC_K
) -> list[str]:
    """자기 텍스트 → topic 층(닫힌 28) kNN 후보 topic_ko 목록(058 프리미티브 재사용·T202).

    빈 텍스트면 임베딩·kNN 자체를 건너뛴다(``[]``·비용 0). 그 외에는 058
    ``knn_topic_candidates(parent_topic=None)`` 에 위임 — 활성 채널 임베딩·결정적 정렬(거리 asc →
    topic_ko asc)·0-노름/NaN 가드가 그 안에 이미 있다. 후보가 비면 ``[]``(레지스트리 미시드 가드).
    """
    if not self_text or not str(self_text).strip():
        return []
    return knn_topic_candidates(conn, self_text, k, parent_topic=None)


# LLM 판정 프롬프트(FR-202·temp=0) — 자기 텍스트 + 후보 topic 목록 제시 → 닫힌 후보 중 하나 확정 +
# 자산의 구체 subtopic 생성. 후보 밖 topic 을 지어내지 못하도록 규칙을 못박는다(FR-203·환각 차단).
_CLASSIFY_PROMPT = """너는 자산의 자기 내용(요약·키워드·라벨)을 읽고 그 자산의 **대표 주제(topic)**를
아래 "후보 주제 목록" 중 **정확히 하나**로 고르고, 자산의 구체적인 **하위주제(subtopic)**를 만드는
분류기다.

규칙:
- topic 은 반드시 "후보 주제 목록"에 있는 라벨 하나만 고른다. 목록에 없는 라벨을 지어내지 않는다.
- subtopic 은 자산이 실제로 다루는 구체 주제어를 짧게(한 어절 위주) 생성한다. 자신이 없으면 null.
- topic_en/subtopic_en 은 대응 영문(없으면 null).
- confidence 는 판정 확신도 0~1 실수.
- JSON 객체 하나만 출력한다. 코드블록·설명 문장 금지.
- 형식: {{"topic_ko":"<후보 중 하나>","topic_en":"...","subtopic_ko":"...","subtopic_en":"...","confidence":0.0}}

자산 자기 내용:
{self_text}

후보 주제 목록:
{candidates}

출력: {{"topic_ko":"...","topic_en":"...","subtopic_ko":"...","subtopic_en":"...","confidence":0.0}}"""

# 재질의 경고(후보 밖 응답 1회 재시도·FR-203) — 원 프롬프트 뒤에 덧붙인다.
_RETRY_SUFFIX = """

경고: 직전 응답의 topic_ko 가 후보 목록에 없었다. 반드시 아래 후보 중 하나의 정확한 라벨만 고르라: {candidates}"""


def _pick_topic_via_llm(self_text: str, candidates: list[str], *, client) -> dict | None:
    """닫힌 후보 중 topic 확정(LLM·temp=0). 후보 밖이면 1회 재질의, 재실패 시 None(FR-203).

    반환은 LLM 응답 dict(``topic_ko`` 가 후보 내임이 검증된 상태) 또는 None. subtopic/en/confidence
    는 호출부가 이 dict 에서 읽는다.
    """
    from src.llm.client import complete_json

    prompt = _CLASSIFY_PROMPT.format(
        self_text=self_text, candidates="\n".join(f"- {c}" for c in candidates)
    )
    out = complete_json(prompt, client=client)
    if _topic_in_candidates(out, candidates):
        return out

    # 후보 밖 → 경고 문구를 덧붙여 1회 재질의(강제 매핑 금지·환각 차단).
    retry_prompt = prompt + _RETRY_SUFFIX.format(candidates=", ".join(candidates))
    out2 = complete_json(retry_prompt, client=client)
    if _topic_in_candidates(out2, candidates):
        return out2
    return None


def _topic_in_candidates(out: dict, candidates: list[str]) -> bool:
    """LLM 응답의 topic_ko 가 닫힌 후보 집합 안에 있는지(닫힌집합 검증·FR-203)."""
    topic_ko = out.get("topic_ko") if isinstance(out, dict) else None
    return isinstance(topic_ko, str) and topic_ko in candidates


def _load_self_meta(conn, asset_id) -> tuple[str | None, list | None, list | None]:
    """``asset_metadata.ext_meta`` 에서 summary/keywords/labels 로드(자기 텍스트 소스).

    keywords 는 문자열 배열(jsonb), labels 는 ``[{label, score}]`` 객체 배열(jsonb·039/v298) →
    psycopg 가 파이썬 list 로 디코드한다. 행이 없으면 (None, None, None).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT m.ext_meta->>'summary' AS summary,
                   m.ext_meta->'keywords' AS keywords,
                   m.ext_meta->'labels'   AS labels
            FROM asset_metadata m
            WHERE m.asset_id = %s
            """,
            (asset_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None, None, None
    return row.get("summary"), row.get("keywords"), row.get("labels")


def _coerce_confidence(value: Any) -> float | None:
    """confidence 를 float 로 강제(파싱 실패·부재 → None)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _upsert_asset_topic(
    conn,
    asset_id,
    topic_ko: str,
    topic_en: str | None,
    subtopic_ko: str | None,
    subtopic_en: str | None,
    confidence: float | None,
) -> None:
    """``asset_topic`` 멱등 upsert(FR-202④) — ON CONFLICT(asset_id) DO UPDATE·updated_at·policy_version.

    최초 insert 는 created_at 기본값(now())·updated_at NULL, 재분류(conflict) 시 updated_at=now() 로
    갱신하고 policy_version 을 기록한다(재현성 추적·FR-601). decided_by 는 하이브리드 고정.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO asset_topic
                (asset_id, topic_ko, topic_en, subtopic_ko, subtopic_en,
                 confidence, decided_by, policy_version)
            VALUES (%s, %s, %s, %s, %s, %s, 'hybrid', %s)
            ON CONFLICT (asset_id) DO UPDATE SET
                topic_ko = EXCLUDED.topic_ko,
                topic_en = EXCLUDED.topic_en,
                subtopic_ko = EXCLUDED.subtopic_ko,
                subtopic_en = EXCLUDED.subtopic_en,
                confidence = EXCLUDED.confidence,
                decided_by = EXCLUDED.decided_by,
                policy_version = EXCLUDED.policy_version,
                updated_at = now()
            """,
            (
                asset_id,
                topic_ko,
                topic_en,
                subtopic_ko,
                subtopic_en,
                confidence,
                POLICY_VERSION,
            ),
        )


def classify_asset_topic(
    conn, asset_id, *, self_text: str | None = None, settings=None, client=None
) -> dict | None:
    """자산 자기주제 하이브리드 판정 + canonicalize + upsert(FR-202). 미부여면 None.

    반환 ``{topic_ko, topic_en, subtopic_ko, subtopic_en, confidence, decided_by:'hybrid'}`` 또는
    None(미부여). 미부여 경로: 자기 텍스트 없음(LLM 미호출) · kNN 후보 없음(레지스트리 미시드) ·
    닫힌집합 검증 2회 실패(FR-203). **예외는 삼키지 않고 올린다** — 호출부(run_ingest)가 registered
    를 유지한 채 주제만 격리한다(FR-204).

    Args:
        self_text: (선택) 이미 구성된 자기 텍스트. None 이면 ``asset_metadata`` 에서 로드해 구성한다.
        settings: (예약) 후보수 등 정책 파라미터 주입용(현재는 기본값 사용).
        client: LLM 클라이언트 주입 seam(미주입=운영 클라이언트·temp=0).
    """
    # ① 자기 텍스트 확보(미주입 시 메타 로드 후 구성). 비면 미부여(LLM 미호출).
    if self_text is None:
        summary, keywords, labels = _load_self_meta(conn, asset_id)
        self_text = build_self_text(summary, keywords, labels)
    if not self_text or not str(self_text).strip():
        return None

    # ② topic 층 kNN 후보(닫힌 28 중 top-k). 비면 미부여(레지스트리 미시드 가드).
    candidates = topic_candidates_for_self_text(conn, self_text, k=_DEFAULT_TOPIC_K)
    if not candidates:
        logger.info("자기주제 분류 스킵 — kNN 후보 없음(레지스트리 미시드): asset_id=%s", asset_id)
        return None

    # ③ LLM 닫힌 확정(후보 밖이면 1회 재질의). 재실패 → 미부여(강제 매핑 금지).
    picked = _pick_topic_via_llm(self_text, candidates, client=client)
    if picked is None:
        logger.info("자기주제 분류 미부여 — 닫힌집합 검증 2회 실패: asset_id=%s", asset_id)
        return None

    topic_ko = picked["topic_ko"]
    raw_sub = picked.get("subtopic_ko")

    # ④ subtopic 정규화(058 부모 스코프 canonicalize·alias 캐시 동결 재사용).
    subtopic_ko = canonicalize_subtopic(conn, topic_ko, raw_sub, client=client)
    # subtopic_en 은 canonicalize 가 추적하지 않으므로(058 은 subtopic en 미보유) LLM 값을 쓴다.
    # subtopic_ko 가 비면(모달리티어·중복 등으로 드롭) en 도 비운다. subtopic_en 정본화는 후속 여지.
    subtopic_en = picked.get("subtopic_en") if subtopic_ko else None

    # ⑤ topic_en 은 registry 정본 우선(FR-102 닫힌 어휘), 없으면 LLM 응답 값.
    topic_en = _lookup_topic_en(conn, topic_ko) or picked.get("topic_en")

    confidence = _coerce_confidence(picked.get("confidence"))

    # ⑥ 멱등 upsert(policy_version 기록).
    _upsert_asset_topic(
        conn, asset_id, topic_ko, topic_en, subtopic_ko, subtopic_en, confidence
    )

    return {
        "topic_ko": topic_ko,
        "topic_en": topic_en,
        "subtopic_ko": subtopic_ko,
        "subtopic_en": subtopic_en,
        "confidence": confidence,
        "decided_by": "hybrid",
    }


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
# 의료(PHI) 제외(헌법 10조): 후보 자산 domain_label IS DISTINCT FROM 'medical'(NULL 도 노출 방지).
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

    Returns:
        ``[{topic_ko, asset_count, subtopics:[{subtopic_ko, asset_count,
        assets:[{asset_id(str), file_name, modality, already_linked}]}]}]`` (구 형상 동일).
        - 상위 ``asset_count`` = 그 주제를 공유하는 distinct 자산 수(대상 제외).
        - 하위 ``asset_count`` = 그 하위주제의 distinct 자산 수(``assets`` 절단 전 실수).
        - 정렬(결정적): 주제 ``asset_count desc → topic_ko asc``(max_topics 절단) · 하위주제는
          이름있는 것 먼저(``asset_count desc → subtopic_ko asc``)·None(기타)은 항상 마지막
          (max_subtopics_per_topic 절단) · 자산은 ``asset_id asc``(max_assets_per_subtopic 절단).
    대상 자산 자기주제 행이 없으면(미부여) 빈 리스트(후보 조회도 안 함).
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
          AND a.domain_label IS DISTINCT FROM 'medical'
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
                "file_name": os.path.basename(r.get("fs_path") or ""),
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
