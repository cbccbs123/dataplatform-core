"""기존 active ``graph_edge.topic`` 을 **닫힌 분류체계(v2)** 로 소급 재작성(spec 058 v2 · G12 · T1201).

목적 (FR-501v2 · C6·C7 · SC-01v2/02v2)
    v2 는 topic 을 닫힌 27+미분류 대범주로 고정하고, 옛 자유기입 topic 은 **대개 subtopic 으로 내려간다**:
      · ``(등산, 입문)`` → ``(스포츠·레저, 등산)`` — 옛 topic(등산)이 subtopic 으로, 옛 subtopic(입문·범용) 폐기.
      · ``(요리, 김밥)`` → ``(음식·요리, 김밥)`` — 옛 topic(요리)은 범주급이라 subtopic 부적격(C7 비움), 옛 subtopic(김밥)이 subtopic.
    규칙:
      · **new_topic = canonicalize_topic(old_topic)** — 닫힌 목록으로 **분류**(off-list→LLM classify·애매하면 미분류).
      · **new_subtopic** = {old_topic, old_subtopic} 중 "그 범주 아래 가장 유용한 구체 주제어" 를 LLM 이 택1
        (범주명/범용어/모달리티 배제) → ``canonicalize_subtopic(new_topic, 택1)`` 로 부모 스코프 정규화(None 가능).

결정성·비용 (헌법 3조·C6)
    distinct **(old_topic, old_subtopic) 쌍**(~수백) 단위로 위 결정을 **1회** 계산해 **쌍 캐시**에 동결한다 —
    엣지수(수천)가 아니라 쌍수만 LLM 을 태운다. 같은 쌍 → 같은 결과(재실행 결정적: topic 분류·subtopic
    선정 seam 은 alias 캐시 + temp=0). topic 분류는 old_topic 단위로 amortize(alias 동결·재분류 0).

세 모드
    ``--dry-run``(기본): 재작성 **계산만**(graph_edge/OS 쓰기 0). 변경 엣지 수·distinct 쌍 수·LLM 호출 수·
        **new topic 분포(27+미분류)·미분류율(SC-02v2)**·매핑 샘플 20·SC 예상(topic∩subtopic=0·모달리티 0·
        distinct new_topic 수)을 리포트. (canonicalize_* 는 alias/subtopic 을 동결하나 트랜잭션 **롤백**으로
        DB 미오염 — LLM/임베딩만 실호출.)
    ``--apply``: ① **백업 먼저**(active 엣지 topic 원본을 ``graph_edge_topic_bak_058_v2`` 로 덤프·v1 백업과 별개
        신규 스냅샷) → ② 변경 엣지 topic jsonb 배치 UPDATE → ③ 커밋. 이미 백업이 있으면 클로버 방지로 중단.
    ``--restore``: 백업에서 topic 원복(골든 회귀 시 되돌리기).

주의
    - **dev 만**. 프로덕션 백필·OS 재색인은 별도 사람/드라이버 게이트(plan G12·🔴 T1202).
    - ``TOPIC_CANONICALIZE_ENABLED`` 플래그와 **무관** — 백필은 게이트를 거치지 않고 직접 재작성한다.
    - OS 재색인(T1202)은 별도: ``python -m src.app.run_opensearch_resync --env dev``. 재작성 → 재색인 순서.

실행
    conda activate AuroraFS
    python scripts/backfill_topic_canonical.py --env dev --dry-run   # 계산만(기본)
    python scripts/backfill_topic_canonical.py --env dev --apply     # 백업 후 재작성·커밋(T1202·사람)
    python scripts/backfill_topic_canonical.py --env dev --restore   # 백업에서 원복
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

_REPO_ROOT = Path(__file__).resolve().parents[1]
# 직접 실행(python scripts/...) 시 repo 루트를 경로에 올려 src 패키지 import 보장(measure_* 러너 동형).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 모달리티 블랙리스트 단일 출처(SC-04 판정·후보 필터) — topic_canonicalize 와 공유(중복 정의 금지).
# 해소 seam(쌍 캐시 결선에서 patch 가능하도록 모듈 네임스페이스로 import).
from src.relations.topic_canonicalize import (  # noqa: E402
    _MODALITY_BLACKLIST,
    canonicalize_subtopic,
    canonicalize_topic,
)

# 백업 테이블(058 v2 전용·신규 스냅샷) — v1 백업(graph_edge_topic_bak_058[_2])과 별개로 둔다.
_BACKUP_TABLE = "graph_edge_topic_bak_058_v2"

# topic jsonb 4필드(graph_persist 저장 계약과 동일 형태).
_TOPIC_KEYS = ("topic_ko", "subtopic_ko", "topic_en", "subtopic_en")

# resolve_pair_fn: (old_topic_ko, old_subtopic_ko) -> 결정 dict
#   {new_topic_ko, new_topic_en, new_subtopic_ko(str|None), selected_source("topic"|"subtopic"|None)}
ResolvePairFn = Callable[[str, str], dict[str, Any]]
# select_fn: (new_topic_ko, candidates) -> 선택 라벨(후보 중 하나) or None
SelectFn = Callable[[str, list[str]], "str | None"]


# ────────────────────────────────────────────────────────────────────────────
# 1) 주제어 선정 (순수 후보 구성 + LLM 택1 seam)
# ────────────────────────────────────────────────────────────────────────────
def _candidate_sources(old_topic_ko: str, old_subtopic_ko: str) -> dict[str, str]:
    """새 subtopic 후보 → 출처 표식 맵(순수). 빈값·모달리티어 제외·중복은 subtopic 출처 우선.

    후보는 옛 (topic, subtopic) 둘 다이다(2→2 매핑: 옛 topic 이 subtopic 으로 내려가는 경우 포함).
    - 빈/공백 라벨 제외. 모달리티어(텍스트/이미지/…)는 subtopic 자격 없음(C7) → 제외.
    - 같은 문자열이면 **subtopic 출처**로 표식(더 구체적 위치라는 신호).
    출처는 재작성 시 subtopic_en 유도에 쓴다(옛 topic 에서 왔으면 topic_en, subtopic 에서 왔으면 subtopic_en).
    """
    out: dict[str, str] = {}
    topic = (old_topic_ko or "").strip()
    sub = (old_subtopic_ko or "").strip()
    if topic and topic.lower() not in _MODALITY_BLACKLIST:
        out[topic] = "topic"
    if sub and sub.lower() not in _MODALITY_BLACKLIST:
        out[sub] = "subtopic"  # 같은 문자열이면 topic 표식을 덮어써 subtopic 우선
    return out


# 주제어 선정 프롬프트(v2·C6) — 옛 (topic, subtopic) 중 그 범주 아래 **구체 주제어** 택1(또는 NONE).
#
# canonicalize 의 classify(범주 분류)와 별개다: 여기서는 이미 정해진 범주 아래에 붙일 **하위주제**로
# 가장 유용한 구체어를 두 후보 중에서 고른다. 범주명 자체(요리·스포츠 같은 대분류어)·범용어(입문·기초·
# 개요·장비·소개 등 내용 없는 일반어)·매체어는 고르지 않는다(쓸만한 게 없으면 NONE). temp=0·zero-shot.
_SELECT_SUBTOPIC_PROMPT = """너는 관계 topic 백필 도우미다. 아래 "범주" 아래에 붙일 **하위주제(subtopic)** 로
가장 유용한 **구체 주제어** 하나를 "후보" 중에서 고른다.

핵심 기준: 그 범주 안에서 **검색·분류에 도움이 되는 구체적인 대상·소재·종목·개념**인가?
- 후보가 **범주명 자체**이거나(예: 요리·스포츠·예술 같은 대분류어), **범용어**(입문·기초·개요·소개·
  장비·정보 등 내용이 비어 일반적인 말)이거나, **매체 형태**(텍스트·이미지·영상·오디오)이면 고르지 않는다.
- 그 범주 아래에서 그 자체로 하나의 주제가 되는 **구체어**를 고른다(예: 김밥·마라톤·태풍·인공위성).
- 두 후보 다 구체 주제어로 쓸만하지 않으면 "NONE".

규칙:
- 반드시 "후보" 목록에 있는 라벨 하나 또는 "NONE" 을 고른다. 목록에 없는 라벨을 지어내지 않는다.
- JSON 객체 하나만 출력한다. 코드블록·설명 문장 금지.
- 형식: {{"subtopic": "<후보 중 하나>"}} 또는 {{"subtopic": "NONE"}}.

범주: {new_topic}
후보: {candidates}

출력: {{"subtopic": "..."}}"""


def _llm_select_subtopic(new_topic_ko: str, candidates: list[str], *, client=None) -> str | None:
    """옛 (topic, subtopic) 후보 중 그 범주 아래 구체 주제어 택1 — 없으면 None(NONE).

    - ``src.llm.client.complete_json`` 단일 seam·temp=0·``client=`` 주입.
    - LLM 이 후보 밖 라벨을 지어내거나 "NONE"/누락이면 안전하게 None(오배정 방지·결정성).
    """
    from src.llm.client import complete_json

    prompt = _SELECT_SUBTOPIC_PROMPT.format(new_topic=new_topic_ko, candidates=candidates)
    out = complete_json(prompt, client=client)
    sub = out.get("subtopic")
    if isinstance(sub, str) and sub in candidates:
        return sub
    return None


def select_subtopic_term(
    new_topic_ko: str, old_topic_ko: str, old_subtopic_ko: str, *, select_fn: SelectFn
) -> tuple[str | None, str | None]:
    """새 subtopic 주제어를 옛 (topic, subtopic) 에서 택1(순수·select seam 주입). 반환 ``(선택, 출처)``.

    - 후보 구성(빈/모달 제외·중복 정리)은 순수 규칙, 최종 택1 은 주입된 ``select_fn`` (운영=LLM).
    - 후보 0 → LLM 미호출·``(None, None)``. 후보가 있으면(단일 후보라도) select_fn 에 제시해
      광의어/범용어를 **거부(None)** 할 수 있게 한다(예: (스포츠,"") 는 광의어라 subtopic 없음).
    - **구체(subtopic) 우선** 순서로 제시(광의 topic 뒤). 선택이 후보에 없으면 ``(None, None)``.
    """
    sources = _candidate_sources(old_topic_ko, old_subtopic_ko)
    if not sources:
        return None, None
    # 구체 우선: subtopic 출처를 앞에 둔다(topic 은 대개 광의).
    candidates = sorted(sources, key=lambda c: 0 if sources[c] == "subtopic" else 1)
    chosen = select_fn(new_topic_ko, candidates)
    if chosen is None or chosen not in sources:
        return None, None
    return chosen, sources[chosen]


# ────────────────────────────────────────────────────────────────────────────
# 2) 쌍→엣지 재작성 계획 (순수·resolve_pair seam 주입)
# ────────────────────────────────────────────────────────────────────────────
def _rewrite_row(old: dict[str, str], decision: dict[str, Any]) -> tuple[dict[str, str], bool, dict[str, bool]]:
    """엣지 하나의 topic jsonb → 재작성(순수). ``decision`` 은 쌍 해소 결과.

    - new_topic_ko/en 은 decision(분류 결과). new_topic_en 이 비면 원본 en 보존(빈 라벨 방지).
    - new_subtopic 이 None → subtopic_ko/en 둘 다 비움(계층·모달리티 정리·graph_persist 동형).
      값이면 subtopic_ko=그 라벨, subtopic_en=출처의 옛 en(topic 발탁→topic_en·subtopic→subtopic_en).
    """
    topic_ko = str(old.get("topic_ko") or "")
    subtopic_ko = str(old.get("subtopic_ko") or "")
    topic_en = str(old.get("topic_en") or "")
    subtopic_en = str(old.get("subtopic_en") or "")

    new_topic_ko = str(decision.get("new_topic_ko") or "") or topic_ko
    new_topic_en = str(decision.get("new_topic_en") or "") or topic_en
    new_sub = decision.get("new_subtopic_ko")
    source = decision.get("selected_source")

    if new_sub is None or not str(new_sub).strip():
        new_subtopic_ko, new_subtopic_en = "", ""
    else:
        new_subtopic_ko = str(new_sub)
        if source == "topic":
            new_subtopic_en = topic_en
        elif source == "subtopic":
            new_subtopic_en = subtopic_en
        else:
            new_subtopic_en = ""

    new = {
        "topic_ko": new_topic_ko,
        "subtopic_ko": new_subtopic_ko,
        "topic_en": new_topic_en,
        "subtopic_en": new_subtopic_en,
    }
    changed = any(new[k] != str(old.get(k) or "") for k in _TOPIC_KEYS)
    flags = {
        "topic_changed": new_topic_ko != topic_ko,
        "topic_en_changed": new_topic_en != topic_en,
        "subtopic_changed": new_subtopic_ko != subtopic_ko,
        "subtopic_cleared": subtopic_ko != "" and new_subtopic_ko == "",
        "etc_topic": new_topic_ko == "미분류",
    }
    return new, changed, flags


def build_plan(rows: list[dict[str, Any]], resolve_pair_fn: ResolvePairFn) -> list[dict[str, Any]]:
    """active 엣지 행 목록 → 재작성 계획(순수·결정적). 각 항목 ``{edge_id, old, new, changed, flags}``.

    각 엣지의 (topic_ko, subtopic_ko) 쌍을 ``resolve_pair_fn`` 으로 해소한다(같은 쌍은 결선 캐시로 1회 계산).
    subtopic_en 은 쌍 결정이 아니라 **그 엣지 자신의 옛 en** 에서 유도하므로 여기서 조립한다(쌍 캐시는 결정만).
    """
    plan: list[dict[str, Any]] = []
    for r in rows:
        old = {k: str(r.get(k) or "") for k in _TOPIC_KEYS}
        decision = resolve_pair_fn(old["topic_ko"], old["subtopic_ko"])
        new, changed, flags = _rewrite_row(old, decision)
        plan.append(
            {"edge_id": r.get("edge_id"), "old": old, "new": new, "changed": changed, "flags": flags}
        )
    return plan


# ────────────────────────────────────────────────────────────────────────────
# 3) SC 판정·분포·리포트 (순수)
# ────────────────────────────────────────────────────────────────────────────
def _distinct_labels(dicts: list[dict[str, str]], key: str) -> set[str]:
    """비어있지 않은 라벨 distinct 집합(순수)."""
    return {str(d.get(key) or "") for d in dicts if str(d.get(key) or "").strip()}


def sc07_distinct_topics(dicts: list[dict[str, str]]) -> int:
    """SC-07v2: distinct topic_ko 수(닫힌 집합으로 수렴)."""
    return len(_distinct_labels(dicts, "topic_ko"))


def sc03_topic_subtopic_overlap(dicts: list[dict[str, str]]) -> list[str]:
    """SC-01v2: topic 이자 subtopic 인 라벨(계층 불일치). 재작성 후 0 이어야 한다."""
    return sorted(_distinct_labels(dicts, "topic_ko") & _distinct_labels(dicts, "subtopic_ko"))


def sc04_modality_subtopics(dicts: list[dict[str, str]]) -> list[str]:
    """SC-01v2: subtopic 이 매체어인 라벨. 재작성 후 0 이어야 한다."""
    subs = _distinct_labels(dicts, "subtopic_ko")
    return sorted(s for s in subs if s.lower() in _MODALITY_BLACKLIST)


def topic_distribution(dicts: list[dict[str, str]]) -> Counter:
    """new topic_ko 분포(순수·빈도 Counter) — 27+미분류 각 몇 건씩."""
    return Counter(str(d.get("topic_ko") or "") for d in dicts if str(d.get("topic_ko") or "").strip())


def etc_rate(dicts: list[dict[str, str]]) -> tuple[int, int]:
    """SC-02v2: (미분류로 분류된 엣지 수, 전체 엣지 수) — 미분류율 근거."""
    total = sum(1 for d in dicts if str(d.get("topic_ko") or "").strip())
    n_etc = sum(1 for d in dicts if str(d.get("topic_ko") or "") == "미분류")
    return n_etc, total


def mapping_sample(plan: list[dict[str, Any]], n: int = 20, *, keywords: list[str] | None = None) -> list[dict[str, Any]]:
    """재작성 계획 → distinct 쌍 매핑 샘플(순수·빈도 desc). 각 항목 old/new (topic,subtopic)+count.

    driver 가 subtopic 선택의 합리성을 검토할 근거다. ``keywords`` 가 주어지면 해당 old_topic 을 가진
    쌍을 우선 포함(등산/요리/김밥/반도체/천문 등 관심 케이스 보장).
    """
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for p in plan:
        key = (p["old"]["topic_ko"], p["old"]["subtopic_ko"])
        entry = by_pair.get(key)
        if entry is None:
            by_pair[key] = {
                "old_topic": key[0], "old_subtopic": key[1],
                "new_topic": p["new"]["topic_ko"], "new_subtopic": p["new"]["subtopic_ko"],
                "count": 1, "changed": p["changed"],
            }
        else:
            entry["count"] += 1
    ordered = sorted(by_pair.values(), key=lambda e: (-e["count"], e["old_topic"], e["old_subtopic"]))
    if not keywords:
        return ordered[:n]
    # keyword 우선 포함 + 나머지 빈도순으로 채워 n 개.
    picked: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for kw in keywords:
        for e in ordered:
            if kw in e["old_topic"] or kw in e["old_subtopic"]:
                k = (e["old_topic"], e["old_subtopic"])
                if k not in seen:
                    picked.append(e)
                    seen.add(k)
                    break
    for e in ordered:
        if len(picked) >= n:
            break
        k = (e["old_topic"], e["old_subtopic"])
        if k not in seen:
            picked.append(e)
            seen.add(k)
    return picked[:n]


def summarize_plan(plan: list[dict[str, Any]]) -> dict[str, Any]:
    """재작성 계획 → 리포트 dict(순수). 전/후 SC 지표·변경 통계·분포·미분류율·매핑 샘플."""
    olds = [p["old"] for p in plan]
    news = [p["new"] for p in plan]
    changed = [p for p in plan if p["changed"]]
    n_etc, n_total = etc_rate(news)
    dist = topic_distribution(news)
    n_pairs = len({(p["old"]["topic_ko"], p["old"]["subtopic_ko"]) for p in plan})
    return {
        "n_edges": len(plan),
        "n_pairs": n_pairs,
        "n_changed": len(changed),
        "n_topic_changed": sum(1 for p in plan if p["flags"]["topic_changed"]),
        "n_subtopic_changed": sum(1 for p in plan if p["flags"]["subtopic_changed"]),
        "n_subtopic_cleared": sum(1 for p in plan if p["flags"]["subtopic_cleared"]),
        "sc07_before": sc07_distinct_topics(olds),
        "sc07_after": sc07_distinct_topics(news),
        "sc03_before": sc03_topic_subtopic_overlap(olds),
        "sc03_after": sc03_topic_subtopic_overlap(news),
        "sc04_before": sc04_modality_subtopics(olds),
        "sc04_after": sc04_modality_subtopics(news),
        "n_etc": n_etc,
        "n_etc_total": n_total,
        "distribution": dict(dist.most_common()),
        "sample": mapping_sample(
            plan, n=20, keywords=["등산", "요리", "김밥", "반도체", "천문"]
        ),
    }


def format_report_lines(report: dict[str, Any], *, mode: str) -> list[str]:
    """리포트 dict → 콘솔 줄(순수·사람 검수용)."""
    n_total = report["n_etc_total"] or 1
    etc_pct = 100.0 * report["n_etc"] / n_total
    lines = [
        f"[백필 topic 정규화 v2 · {mode}] active 엣지 {report['n_edges']}건 · distinct 쌍 {report['n_pairs']}개",
        f"  변경 엣지: {report['n_changed']}건 "
        f"(topic {report['n_topic_changed']} · subtopic 변경 {report['n_subtopic_changed']}"
        f"[비움 {report['n_subtopic_cleared']}])",
    ]
    if "n_llm_calls" in report:
        lines.append(f"  LLM 호출 수: {report['n_llm_calls']}회 (쌍/topic 단위·엣지수 무관)")
    lines += [
        f"  SC-07 distinct topic: {report['sc07_before']} → {report['sc07_after']} "
        f"(닫힌 27+미분류 중 {report['sc07_after']}종 등장)",
        f"  SC-01 topic∩subtopic 라벨: {len(report['sc03_before'])} → {len(report['sc03_after'])}"
        + (f"  ⚠ 잔존: {report['sc03_after']}" if report["sc03_after"] else "  (0·계층 일관)"),
        f"  SC-01 subtopic 모달리티: {len(report['sc04_before'])} → {len(report['sc04_after'])}"
        + (f"  ⚠ 잔존: {report['sc04_after']}" if report["sc04_after"] else "  (0·모달리티 정리)"),
        f"  SC-02 미분류율: {report['n_etc']}/{report['n_etc_total']} = {etc_pct:.1f}%",
        "  new topic 분포:",
    ]
    for topic, cnt in report["distribution"].items():
        lines.append(f"      {cnt:>4}  {topic}")
    lines.append("  매핑 샘플(old (topic,subtopic) → new (topic,subtopic) · count):")
    for s in report["sample"]:
        old = f"({s['old_topic']}, {s['old_subtopic']})"
        new = f"({s['new_topic']}, {s['new_subtopic']})"
        mark = "" if s["changed"] else "  =불변"
        lines.append(f"      {s['count']:>4}  {old:<28} → {new}{mark}")
    return lines


# ────────────────────────────────────────────────────────────────────────────
# 4) DB 경로 — 쌍 캐시 결선(canonicalize_topic·select·canonicalize_subtopic) · 백업 · 재작성 · 복원
# ────────────────────────────────────────────────────────────────────────────
_ACTIVE_EDGES_SQL = """
SELECT edge_id,
       topic->>'topic_ko'    AS topic_ko,
       topic->>'subtopic_ko' AS subtopic_ko,
       topic->>'topic_en'    AS topic_en,
       topic->>'subtopic_en' AS subtopic_en
FROM graph_edge
WHERE status = 'active'
ORDER BY edge_id
"""


def fetch_active_edges(conn) -> list[dict[str, Any]]:
    """active 엣지의 (edge_id + topic 4필드) 평탄화 목록(결정적 정렬·edge_id str 강제)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_ACTIVE_EDGES_SQL)
        rows = cur.fetchall()
    return [
        {
            "edge_id": str(r["edge_id"]),
            "topic_ko": r["topic_ko"],
            "subtopic_ko": r["subtopic_ko"],
            "topic_en": r["topic_en"],
            "subtopic_en": r["subtopic_en"],
        }
        for r in rows
    ]


def make_pair_resolver(conn, *, client=None) -> tuple[ResolvePairFn, dict[str, int]]:
    """쌍 해소 결선(캐시) — canonicalize_topic(분류) + LLM 주제어 선정 + canonicalize_subtopic(부모 스코프).

    거치는 결정(쌍당 1회·캐시 동결·엣지수 무관):
      1) new_topic = ``canonicalize_topic(old_topic)`` — 닫힌 목록 분류(off-list→LLM classify·미분류 폴백).
         old_topic 단위로 alias 동결되어 같은 old_topic 은 재분류 0(amortize).
      2) new_subtopic 후보 택1 = ``_llm_select_subtopic``(범주명/범용/모달 배제·NONE 가능).
      3) ``canonicalize_subtopic(new_topic, 택1)`` — 부모 스코프 정규화(alias→kNN→judge→등록·None 가능).
    ``stats`` 로 계산한 distinct 쌍 수를 노출(리포트 근거). LLM 호출 총수는 run 층에서 seam 카운트.
    """
    pair_cache: dict[tuple[str, str], dict[str, Any]] = {}
    stats = {"n_pairs": 0}

    def resolve_pair(old_topic_ko: str, old_subtopic_ko: str) -> dict[str, Any]:
        key = (old_topic_ko or "", old_subtopic_ko or "")
        if key in pair_cache:
            return pair_cache[key]

        topic_res = canonicalize_topic(conn, old_topic_ko, None, client=client)
        new_topic_ko = str(topic_res.get("canonical_ko") or "")
        new_topic_en = topic_res.get("canonical_en")

        chosen, source = select_subtopic_term(
            new_topic_ko, old_topic_ko or "", old_subtopic_ko or "",
            select_fn=lambda nt, cands: _llm_select_subtopic(nt, cands, client=client),
        )
        if chosen is None:
            new_sub_ko = None
        else:
            # 부모 스코프 정규화(모달/범주명 None·부모 내 alias/kNN/judge/등록). None 가능.
            new_sub_ko = canonicalize_subtopic(conn, new_topic_ko, chosen, client=client)
            if new_sub_ko is None:
                source = None

        decision = {
            "new_topic_ko": new_topic_ko,
            "new_topic_en": new_topic_en,
            "new_subtopic_ko": new_sub_ko,
            "selected_source": source,
        }
        pair_cache[key] = decision
        stats["n_pairs"] += 1
        return decision

    return resolve_pair, stats


def compute_plan(conn, *, client=None) -> list[dict[str, Any]]:
    """DB 에서 active 엣지 읽고 쌍 캐시 결선으로 재작성 계획 산출(seam 결선 + 순수 build_plan)."""
    rows = fetch_active_edges(conn)
    resolve_pair, _ = make_pair_resolver(conn, client=client)
    return build_plan(rows, resolve_pair)


def backup_row_count(conn) -> int | None:
    """백업 테이블 행 수(없으면 None) — --apply 클로버 방지·--restore 대상 확인."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (_BACKUP_TABLE,))
        if cur.fetchone()[0] is None:
            return None
        cur.execute(f"SELECT count(*) FROM {_BACKUP_TABLE}")
        return int(cur.fetchone()[0])


def create_backup(conn) -> int:
    """active 엣지의 (edge_id, topic) 원본을 백업 테이블에 덤프(신규 스냅샷). 백업 행 수 반환."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE {_BACKUP_TABLE} (
                edge_id uuid PRIMARY KEY,
                topic jsonb NOT NULL,
                backed_up_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            INSERT INTO {_BACKUP_TABLE} (edge_id, topic)
            SELECT edge_id, topic FROM graph_edge WHERE status = 'active'
            """
        )
        cur.execute(f"SELECT count(*) FROM {_BACKUP_TABLE}")
        return int(cur.fetchone()[0])


def apply_rewrite(conn, plan: list[dict[str, Any]]) -> int:
    """변경 엣지의 ``topic`` jsonb 를 정본으로 배치 UPDATE. 재작성 행 수 반환(topic 만 갱신·복원 대칭)."""
    changed = [p for p in plan if p["changed"]]
    if not changed:
        return 0
    params = [(json.dumps(p["new"], ensure_ascii=False), p["edge_id"]) for p in changed]
    with conn.cursor() as cur:
        cur.executemany("UPDATE graph_edge SET topic = %s::jsonb WHERE edge_id = %s", params)
    return len(changed)


def restore_from_backup(conn) -> int:
    """백업 테이블에서 ``topic`` 원복(--restore). 복원 행 수 반환."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE graph_edge ge
            SET topic = b.topic
            FROM {_BACKUP_TABLE} b
            WHERE ge.edge_id = b.edge_id
            """
        )
        return cur.rowcount


# ────────────────────────────────────────────────────────────────────────────
# 5) 실행(IO) — 모드별 오케스트레이션
# ────────────────────────────────────────────────────────────────────────────
class _LlmCallCounter:
    """dry-run 중 실제 LLM(``complete_json``) 호출 총수를 센다(리포트 근거).

    classify/select/judge seam 모두 호출 시점에 ``from src.llm.client import complete_json`` 하므로
    모듈 속성을 감싸면 전부 카운트된다. temp=0 이라 카운트는 결정적(쌍/topic 단위 상한).
    """

    def __init__(self):
        self.n = 0

    def __enter__(self):
        import src.llm.client as _llmc

        self._mod = _llmc
        self._orig = _llmc.complete_json

        def _counting(*a, **k):
            self.n += 1
            return self._orig(*a, **k)

        _llmc.complete_json = _counting
        return self

    def __exit__(self, *exc):
        self._mod.complete_json = self._orig
        return False


def run_dry_run(db) -> dict[str, Any]:
    """재작성 계산만(graph_edge/OS 쓰기 0). LLM 호출은 실행되나 트랜잭션 롤백으로 DB 미오염."""
    with _LlmCallCounter() as counter:
        with db, db.connection() as conn:
            plan = compute_plan(conn)
            conn.rollback()  # canonicalize_* 가 동결한 alias/subtopic 을 되돌려 DB 미오염(읽기 전용 보장)
    report = summarize_plan(plan)
    report["n_llm_calls"] = counter.n
    return report


def run_apply(db) -> dict[str, Any]:
    """① 백업(클로버 방지) → ② 재작성 → ③ 커밋. 리포트 + 백업/재작성 수 반환.

    주의: canonicalize_* 가 동결하는 alias/subtopic 은 이 커밋에 함께 남는다(생성시=백필 일치·정본 캐시).
    """
    with _LlmCallCounter() as counter:
        with db, db.connection() as conn:
            existing = backup_row_count(conn)
            if existing is not None:
                raise SystemExit(
                    f"백업 테이블 {_BACKUP_TABLE} 이 이미 존재(행 {existing}). 클로버 방지로 중단.\n"
                    f"  되돌리려면 --restore, 다시 백필하려면 먼저 백업 테이블을 삭제하라\n"
                    f"  (DROP TABLE {_BACKUP_TABLE};)."
                )
            plan = compute_plan(conn)
            report = summarize_plan(plan)
            n_backup = create_backup(conn)
            n_rewrite = apply_rewrite(conn, plan)
            conn.commit()
    report["n_llm_calls"] = counter.n
    report["n_backup"] = n_backup
    report["n_rewrite"] = n_rewrite
    report["backup_table"] = _BACKUP_TABLE
    return report


def run_restore(db) -> int:
    """백업에서 topic 원복·커밋. 복원 수 반환."""
    with db, db.connection() as conn:
        existing = backup_row_count(conn)
        if existing is None:
            raise SystemExit(f"백업 테이블 {_BACKUP_TABLE} 이 없다 — 복원 불가.")
        n = restore_from_backup(conn)
        conn.commit()
    return n


def main() -> int:
    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    p = argparse.ArgumentParser(
        description="graph_edge.topic 백필 v2 — 닫힌 분류체계로 쌍 단위 재작성(spec 058 v2 · G12)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="재작성 계산만(기본·쓰기 0)")
    mode.add_argument("--apply", action="store_true", help="백업 후 재작성·커밋(T1202·사람 게이트)")
    mode.add_argument("--restore", action="store_true", help="백업에서 topic 원복")
    args = p.parse_args()

    # 프로덕션 백필은 사람 게이트(plan G12·🔴) — 스크립트에서 실수 차단.
    if args.env == "prod":
        raise SystemExit("프로덕션 백필은 별도 사람 게이트다. 이 스크립트는 dev 만 지원한다.")

    dotenv_path = _REPO_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    db = PostgresUtil()

    if args.apply:
        report = run_apply(db)
        print("\n".join(format_report_lines(report, mode="apply")))
        print(
            f"  백업: {report['backup_table']} 에 {report['n_backup']}행 · "
            f"재작성: {report['n_rewrite']}행 커밋 완료."
        )
        return 0
    if args.restore:
        n = run_restore(db)
        print(f"[복원] 백업 {_BACKUP_TABLE} 에서 topic {n}행 원복·커밋 완료.")
        return 0

    report = run_dry_run(db)
    print("\n".join(format_report_lines(report, mode="dry-run")))
    print("  (dry-run·graph_edge/OS 쓰기 0. 적용하려면 --apply · 이후 OS 재색인 — T1202 사람 게이트)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
