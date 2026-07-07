"""미분류(catch-all) 감시 리포트 — 닫힌 분류체계의 건강 지표를 관측(spec 058 v2 · G13 · T1301).

목적 (FR-601v2 · 거버넌스 taxonomy_draft.md §4)
    v2 는 topic 을 닫힌 27+미분류로 고정하고, 확장은 **병합이 아니라 가산적 추가**로 한다(§4-3):
    미분류(어디에도 확신 없음)에 일관된 주제가 누적되면 사람이 새 범주를 추가(v3…)하고
    **미분류 파킹분만** 재분류한다. 이 리포트는 그 판단을 위한 **감시 지표**를 주기 관측한다:
      · **미분류율** = active graph_edge 중 topic='미분류' 엣지 / 전체 (§4-2 건강 지표·임계 ≤ 수 %).
      · **미분류에 걸린 자산 수** = 미분류 엣지가 잇는 distinct 자산 수(파킹 규모).
      · **미분류 내 subtopic(제안 라벨) 누적** = 미분류 엣지의 subtopic 빈도(새 범주 후보의 근거).
      · (선택) **classify alias 원본 라벨** = ``topic_alias`` 중 ``decided_by='classify'`` 로 미분류에
        매핑된 raw 라벨(생성시 LLM 이 미분류로 분류한 자유 라벨 → 향후 범주 추가 후보).

읽기전용·결정성 (헌법 3조·G7 대체)
    graph_edge/registry/alias 를 **일절 변경하지 않는다**(병합 배치 아님). 순수 집계라 LLM 0·재실행 동일.
    G7(자동 병합 휴리스틱)은 v2 에서 폐기됐고(진동 원인), 그 자리를 이 감시 리포트 + 사람의 가산적
    범주 추가가 대신한다(spec 058 v2 · ADR 2026-07-07).

실행
    conda activate AuroraFS
    python scripts/report_topic_unclassified.py --env dev
    python scripts/report_topic_unclassified.py --env dev --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

_REPO_ROOT = Path(__file__).resolve().parents[1]
# 직접 실행(python scripts/...) 시 repo 루트를 경로에 올려 src 패키지 import 보장(다른 러너와 동형).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# catch-all 범주명 — 2026-07-07 '기타'→'미분류' 개명(음악>기타[guitar] subtopic 동음이의 해소).
# taxonomy_seed.json·topic_canonicalize classify 폴백과 **동일 문자열**(단일 출처 원칙).
_UNCLASSIFIED_KO = "미분류"

# 건강 임계 기본값(거버넌스 §4-2) — 미분류율이 이 이하면 healthy. §3 커버리지 완전 → 실측 0%.
_DEFAULT_THRESHOLD = 0.05


# ────────────────────────────────────────────────────────────────────────────
# 1) 순수 집계 (실 DB/LLM 없이 단위테스트로 덮는다)
# ────────────────────────────────────────────────────────────────────────────
def unclassified_rate(edge_rows: list[dict[str, Any]]) -> tuple[int, int]:
    """SC-02v2·§4-2: (미분류로 분류된 엣지 수, 전체 active 엣지 수) — 미분류율 근거(분모=전체)."""
    total = len(edge_rows)
    n_unc = sum(1 for r in edge_rows if str(r.get("topic_ko") or "") == _UNCLASSIFIED_KO)
    return n_unc, total


def unclassified_subtopics(edge_rows: list[dict[str, Any]]) -> Counter:
    """미분류 엣지의 subtopic(제안 라벨) 빈도 누적(순수·§4-3 범주 추가 근거). 빈 subtopic 제외."""
    c: Counter = Counter()
    for r in edge_rows:
        if str(r.get("topic_ko") or "") == _UNCLASSIFIED_KO:
            sub = str(r.get("subtopic_ko") or "").strip()
            if sub:
                c[sub] += 1
    return c


def build_report(
    edge_rows: list[dict[str, Any]],
    *,
    n_unclassified_assets: int,
    classify_raw_labels: list[str],
    threshold: float = _DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """감시 지표 리포트 조립(순수). 미분류율·건강 판정·제안 라벨 누적·classify 후보 라벨."""
    n_unc, total = unclassified_rate(edge_rows)
    rate = (n_unc / total) if total else 0.0
    subs = unclassified_subtopics(edge_rows)
    return {
        "n_edges_total": total,
        "n_unclassified": n_unc,
        "unclassified_rate": round(rate, 4),
        "threshold": threshold,
        "healthy": rate <= threshold,
        "n_unclassified_assets": n_unclassified_assets,
        "unclassified_subtopics": dict(subs.most_common()),
        "classify_raw_labels": sorted(classify_raw_labels),
    }


def format_report_lines(report: dict[str, Any]) -> list[str]:
    """리포트 dict → 콘솔 줄(순수·사람 검수용)."""
    pct = 100.0 * report["unclassified_rate"]
    thr_pct = 100.0 * report["threshold"]
    health = "건강(healthy·임계 이하)" if report["healthy"] else "⚠ 임계 초과 — 범주 추가 검토"
    lines = [
        "[미분류 감시 리포트 · 058 G13 · 읽기전용]",
        f"  미분류율: {report['n_unclassified']}/{report['n_edges_total']} = {pct:.1f}%"
        f"  (임계 {thr_pct:.1f}% → {health})",
        f"  미분류에 걸린 자산 수: {report['n_unclassified_assets']}",
    ]
    subs = report["unclassified_subtopics"]
    if subs:
        lines.append("  미분류 내 제안 라벨(subtopic) 누적(빈도·범주 추가 후보):")
        for label, cnt in subs.items():
            lines.append(f"      {cnt:>4}  {label}")
    else:
        lines.append("  미분류 내 제안 라벨: 없음")
    labels = report["classify_raw_labels"]
    if labels:
        lines.append("  classify alias 원본 라벨(생성시 LLM 미분류 분류·범주 추가 후보):")
        for label in labels:
            lines.append(f"      {label}")
    else:
        lines.append("  classify alias 미분류 매핑: 없음")
    return lines


# ────────────────────────────────────────────────────────────────────────────
# 2) DB 경로 (읽기전용 — graph_edge/registry/alias 미변경)
# ────────────────────────────────────────────────────────────────────────────
_ACTIVE_TOPIC_SQL = """
SELECT topic->>'topic_ko'    AS topic_ko,
       topic->>'subtopic_ko' AS subtopic_ko
FROM graph_edge
WHERE status = 'active'
ORDER BY edge_id
"""

# 미분류 엣지가 잇는 distinct 자산 수 — src/dst node 양끝의 asset 을 합집합으로 센다(대칭 엣지 대비).
_UNCLASSIFIED_ASSETS_SQL = """
SELECT count(DISTINCT n.asset_id)
FROM graph_edge e
JOIN node n ON n.node_id IN (e.src_node, e.dst_node) AND n.node_kind = 'asset'
WHERE e.status = 'active' AND e.topic->>'topic_ko' = %s
"""

# 생성시 LLM 이 미분류로 분류한(decided_by='classify') topic 층 alias 의 원본 라벨(범주 추가 후보).
_CLASSIFY_UNCLASSIFIED_ALIAS_SQL = """
SELECT raw_ko
FROM topic_alias
WHERE parent_topic IS NULL
  AND decided_by = 'classify'
  AND canonical_ko = %s
ORDER BY raw_ko
"""


def fetch_active_topic_rows(conn) -> list[dict[str, Any]]:
    """active 엣지의 (topic_ko, subtopic_ko) 평탄화 목록(결정적 정렬·읽기전용)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_ACTIVE_TOPIC_SQL)
        return [dict(r) for r in cur.fetchall()]


def count_unclassified_assets(conn) -> int:
    """미분류 active 엣지가 잇는 distinct 자산 수(읽기전용)."""
    with conn.cursor() as cur:
        cur.execute(_UNCLASSIFIED_ASSETS_SQL, (_UNCLASSIFIED_KO,))
        return int(cur.fetchone()[0])


def fetch_classify_unclassified_aliases(conn) -> list[str]:
    """classify 로 미분류에 매핑된 topic 층 alias 원본 라벨 목록(읽기전용·범주 추가 후보)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_CLASSIFY_UNCLASSIFIED_ALIAS_SQL, (_UNCLASSIFIED_KO,))
        return [str(r["raw_ko"]) for r in cur.fetchall()]


def run_report(*, env: str, threshold: float = _DEFAULT_THRESHOLD) -> dict[str, Any]:
    """감시 리포트 실행(읽기전용 DB·LLM 0). .env.{env} 로드 → init_settings → 집계."""
    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    dotenv_path = _REPO_ROOT / f".env.{env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(env)

    db = PostgresUtil()
    with db, db.connection() as conn:
        edge_rows = fetch_active_topic_rows(conn)
        n_assets = count_unclassified_assets(conn)
        classify_labels = fetch_classify_unclassified_aliases(conn)
    report = build_report(
        edge_rows,
        n_unclassified_assets=n_assets,
        classify_raw_labels=classify_labels,
        threshold=threshold,
    )
    report["env"] = env
    return report


def main() -> int:
    p = argparse.ArgumentParser(
        description="미분류 감시 리포트 — 닫힌 분류체계 건강 지표(spec 058 v2 · G13 · 읽기전용)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument(
        "--threshold", type=float, default=_DEFAULT_THRESHOLD,
        help="미분류율 건강 임계(기본 0.05 = 5%%)",
    )
    p.add_argument("--json", dest="json_out", default=None, help="리포트 JSON 저장 경로(선택)")
    args = p.parse_args()

    report = run_report(env=args.env, threshold=args.threshold)
    print("\n".join(format_report_lines(report)))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  리포트 JSON 저장: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
