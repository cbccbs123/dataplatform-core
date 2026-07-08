"""자기주제 근거율(grounding)·분포 리포트 — 스펙 근거 스크립트 정식화(spec 065·FR-602·SC-02/03).

무엇을 측정하나
    065 는 자산 주제를 관계 이웃 엣지 투영이 아니라 **자기 내용(summary/keywords)에서 확정한 정본**
    (``asset_topic``)으로 둔다. 이 리포트는 그 정본의 건강 지표를 **읽기전용**으로 관측한다:

    · 집계 A(분포) — ``topic_ko`` 별·``(topic_ko, subtopic_ko)`` 별 카운트 + 부여율(SC-03).
        미부여 사유 추정: ``no_text``(메타에 summary/keywords 없음) vs ``분류실패``(텍스트 있는데 행 없음).
    · 집계 B(근거율·SC-02) — 자산 주제(``topic_ko``·``subtopic_ko``)가 그 자산의 자기 텍스트에
        **문자열로 등장**하는 비율. 등장하면 grounded, 없으면 polluted(오염). 문자열 매칭은
        스펙과 동일한 **보수적 하한**(의미적 근거는 이보다 높다). SC-02 = 오염율 ≤ 5% 목표.
    · (참고) ``--compare-projection`` — 옛 이웃-엣지 투영 방식이었다면 같은 grounding 지표가
        얼마였을지 비교(``graph_edge.topic`` 재구성·다중 라벨 중 하나라도 근거 있으면 grounded).

읽기전용·결정성 (헌법 3조)
    ``asset_topic``/``asset_metadata``/``graph_edge`` 를 일절 변경하지 않는다(LLM 0·순수 집계).
    같은 DB 상태 → 같은 리포트. 집계는 순수 함수로 분리해 단위테스트로 덮고, DB 조회는 얇게 둔다.

실행 (백필 전/후로 실행해 오염율 변화를 비교 — FR-503 백필 실행 자체는 사람 게이트)
    conda activate AuroraFS
    python scripts/topic_grounding_report.py --env dev
    python scripts/topic_grounding_report.py --env dev --compare-projection
    python scripts/topic_grounding_report.py --env dev --json out.json
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


# ────────────────────────────────────────────────────────────────────────────
# 1) 순수 집계 (실 DB/LLM 없이 단위테스트로 덮는다)
# ────────────────────────────────────────────────────────────────────────────
def topic_distribution(topic_rows: list[dict[str, Any]]) -> Counter:
    """``topic_ko`` 별 자산 수 분포(순수·빈 topic_ko 제외)."""
    c: Counter = Counter()
    for r in topic_rows:
        tk = str(r.get("topic_ko") or "").strip()
        if tk:
            c[tk] += 1
    return c


def pair_distribution(topic_rows: list[dict[str, Any]]) -> Counter:
    """``(topic_ko, subtopic_ko)`` 짝별 자산 수 분포(순수). subtopic 빈값은 None 으로 정규화."""
    c: Counter = Counter()
    for r in topic_rows:
        tk = str(r.get("topic_ko") or "").strip()
        if not tk:
            continue
        sub = r.get("subtopic_ko")
        sub = str(sub).strip() if sub and str(sub).strip() else None
        c[(tk, sub)] += 1
    return c


def _label_in_text(label: Any, text: str) -> bool:
    """라벨(topic_ko/subtopic_ko)이 자기 텍스트에 문자열로 등장하는지(순수·보수적 하한).

    한국어는 대소문자가 없어 그대로 부분문자열 매칭한다. 빈/None 라벨은 매칭 대상 아님.
    """
    if not label:
        return False
    s = str(label).strip()
    return bool(s) and s in text


def _asset_grounded(labels: list, self_text: str) -> bool:
    """자산 라벨 중 하나라도 자기 텍스트에 근거가 있으면 grounded(순수).

    ``labels`` 는 ``[(topic_ko, subtopic_ko), ...]``. 정본은 라벨 1개, 투영 비교는 여러 개다.
    각 라벨은 subtopic(더 구체) 또는 topic 이 텍스트에 등장하면 근거 있음으로 본다.
    """
    text = self_text or ""
    for topic_ko, subtopic_ko in labels:
        if _label_in_text(subtopic_ko, text) or _label_in_text(topic_ko, text):
            return True
    return False


def build_grounding_report(asset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """근거율/오염율 리포트(순수·집계 B·SC-02).

    Args:
        asset_rows: ``[{asset_id, labels:[(topic_ko, subtopic_ko)], self_text}]``.

    Returns:
        ``{n_assets, n_grounded, n_polluted, grounding_rate, pollution_rate,
        polluted_samples}``. ``pollution_rate`` = 오염 자산 / 주제 보유 자산(SC-02 측정치).
    """
    total = len(asset_rows)
    grounded = 0
    polluted_samples: list[dict[str, Any]] = []
    for r in asset_rows:
        if _asset_grounded(r.get("labels") or [], r.get("self_text") or ""):
            grounded += 1
        elif len(polluted_samples) < 20:  # 검수용 오염 샘플(결정적 앞 20)
            polluted_samples.append(
                {"asset_id": r.get("asset_id"), "labels": [list(x) for x in (r.get("labels") or [])]}
            )
    polluted = total - grounded
    return {
        "n_assets": total,
        "n_grounded": grounded,
        "n_polluted": polluted,
        "grounding_rate": round(grounded / total, 4) if total else 0.0,
        "pollution_rate": round(polluted / total, 4) if total else 0.0,
        "polluted_samples": polluted_samples,
    }


def build_distribution_report(
    *, topic_rows: list[dict[str, Any]], text_asset_ids: set[str], n_registered: int
) -> dict[str, Any]:
    """분포·부여율·미부여 사유 리포트(순수·집계 A·SC-03).

    Args:
        topic_rows: ``asset_topic`` 행 ``[{asset_id, topic_ko, subtopic_ko}]``.
        text_asset_ids: registered 중 자기 텍스트(summary/keywords) 보유 자산 id 집합.
        n_registered: registered 자산 총수.

    부여율(``assignment_rate``)은 **텍스트 보유 자산 대비**(SC-03: 텍스트 보유의 95%↑ 목표).
    미부여 사유: ``no_text`` = registered 인데 텍스트 없음, ``분류실패`` = 텍스트 있는데 주제 행 없음.
    """
    topic_ids = {str(r["asset_id"]) for r in topic_rows}
    n_text = len(text_asset_ids)
    assigned_with_text = len(topic_ids & text_asset_ids)
    return {
        "n_registered": n_registered,
        "n_with_text": n_text,
        "n_with_topic": len(topic_ids),
        "n_no_text": max(n_registered - n_text, 0),
        "n_classify_failed": len(text_asset_ids - topic_ids),
        "assignment_rate": round(assigned_with_text / n_text, 4) if n_text else 0.0,
        "topic_distribution": dict(topic_distribution(topic_rows).most_common()),
        "pair_distribution": {
            (f"{t}>{s}" if s else t): c
            for (t, s), c in pair_distribution(topic_rows).most_common()
        },
    }


def group_label_rows(flat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """평탄 행 → 자산별 라벨 묶음(순수·정본/투영 공용).

    입력 ``[{asset_id, topic_ko, subtopic_ko, self_text}]``(자산당 여러 행 가능·투영) →
    ``[{asset_id, labels:[(topic_ko, subtopic_ko)], self_text}]``. 자기 텍스트는 자산당 동일
    가정(첫 등장 보존). 등장 순서를 보존해 결정적(헌법 3조).
    """
    by_asset: dict[str, dict[str, Any]] = {}
    for r in flat_rows:
        aid = str(r["asset_id"])
        entry = by_asset.get(aid)
        if entry is None:
            entry = {"asset_id": aid, "labels": [], "self_text": r.get("self_text") or ""}
            by_asset[aid] = entry
        tk = r.get("topic_ko")
        if tk:
            entry["labels"].append((tk, r.get("subtopic_ko")))
    return list(by_asset.values())


def format_report_lines(report: dict[str, Any]) -> list[str]:
    """리포트 dict → 콘솔 줄(순수·사람 검수용)."""
    dist = report["distribution"]
    grd = report["grounding"]
    a_pct = 100.0 * dist["assignment_rate"]
    p_pct = 100.0 * grd["pollution_rate"]
    lines = [
        "[자기주제 grounding 리포트 · 065 · 읽기전용]",
        "  — 집계 A: 부여율·분포 —",
        f"  registered {dist['n_registered']} · 텍스트 보유 {dist['n_with_text']} · "
        f"주제 부여 {dist['n_with_topic']}",
        f"  부여율(텍스트 보유 대비): {a_pct:.1f}%  (SC-03 목표 95%↑)",
        f"  미부여 사유: no_text {dist['n_no_text']} · 분류실패 {dist['n_classify_failed']}",
        "  topic 분포:",
    ]
    for topic, cnt in dist["topic_distribution"].items():
        lines.append(f"      {cnt:>4}  {topic}")
    lines += [
        "  — 집계 B: 근거율/오염율(SC-02) —",
        f"  주제 보유 {grd['n_assets']} · 근거 있음 {grd['n_grounded']} · "
        f"오염 {grd['n_polluted']}",
        f"  오염율: {p_pct:.1f}%  (SC-02 목표 ≤5% · 문자열 매칭 보수적 하한)",
    ]
    proj = report.get("projection_grounding")
    if proj is not None:
        pj_pct = 100.0 * proj["pollution_rate"]
        lines += [
            "  — (참고) 옛 이웃-엣지 투영 방식 대비 —",
            f"  투영 주제 보유 {proj['n_assets']} · 오염율: {pj_pct:.1f}%"
            f"  (정본 {p_pct:.1f}% 와 비교)",
        ]
    return lines


# ────────────────────────────────────────────────────────────────────────────
# 2) DB 경로 (읽기전용 — 얇게)
# ────────────────────────────────────────────────────────────────────────────
# 자기주제 정본 + 자기 텍스트 소스(summary/keywords/labels) — 결정적 정렬.
_GROUNDING_SQL = """
SELECT at.asset_id, at.topic_ko, at.subtopic_ko,
       m.ext_meta->>'summary' AS summary,
       m.ext_meta->'keywords' AS keywords,
       m.ext_meta->'labels'   AS labels
FROM asset_topic at
JOIN asset a ON a.asset_id = at.asset_id
LEFT JOIN asset_metadata m ON m.asset_id = at.asset_id
WHERE a.status = 'registered'
ORDER BY at.asset_id
"""

# registered 중 자기 텍스트(summary 비지 않음 OR keywords 비지 않은 배열) 보유 자산 id.
_TEXT_ASSET_SQL = """
SELECT a.asset_id
FROM asset a
JOIN asset_metadata m ON m.asset_id = a.asset_id
WHERE a.status = 'registered'
  AND (
    COALESCE(m.ext_meta->>'summary', '') <> ''
    OR (
      jsonb_typeof(m.ext_meta->'keywords') = 'array'
      AND jsonb_array_length(m.ext_meta->'keywords') > 0
    )
  )
"""

_REGISTERED_COUNT_SQL = "SELECT count(*) FROM asset WHERE status = 'registered'"

# (참고) 옛 이웃-엣지 투영 재구성 — active 엣지의 topic 을 양끝 자산에 투영(옛 project_asset_topics
# 와 동형: 엣지 하나가 src·dst 두 자산에 그 topic 을 준다). 의료(PHI) 제외(헌법 10조).
_PROJECTION_SQL = """
SELECT n.asset_id AS asset_id,
       ge.topic->>'topic_ko'    AS topic_ko,
       ge.topic->>'subtopic_ko' AS subtopic_ko,
       m.ext_meta->>'summary' AS summary,
       m.ext_meta->'keywords' AS keywords,
       m.ext_meta->'labels'   AS labels
FROM graph_edge ge
JOIN node sn ON sn.node_id = ge.src_node AND sn.node_kind = 'asset'
JOIN node dn ON dn.node_id = ge.dst_node AND dn.node_kind = 'asset'
JOIN LATERAL (VALUES (sn.asset_id), (dn.asset_id)) AS n(asset_id) ON TRUE
JOIN asset a ON a.asset_id = n.asset_id AND a.status = 'registered'
LEFT JOIN asset_metadata m ON m.asset_id = n.asset_id
WHERE ge.status = 'active'
  AND COALESCE(ge.topic->>'topic_ko', '') <> ''
  AND a.domain_label IS DISTINCT FROM 'medical'
ORDER BY n.asset_id
"""


def _build_self_text(row: dict[str, Any]) -> str:
    """행의 summary/keywords/labels → 자기 텍스트(분류 seam 과 동일 구성·중복 구현 금지)."""
    from src.classify.asset_topic import build_self_text

    return build_self_text(row.get("summary"), row.get("keywords"), row.get("labels"))


def fetch_grounding_rows(conn) -> list[dict[str, Any]]:
    """자기주제 정본 + 자기 텍스트 평탄 행(읽기전용). ``self_text`` 는 여기서 구성."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_GROUNDING_SQL)
        rows = cur.fetchall()
    return [
        {
            "asset_id": str(r["asset_id"]),
            "topic_ko": r["topic_ko"],
            "subtopic_ko": r["subtopic_ko"],
            "self_text": _build_self_text(r),
        }
        for r in rows
    ]


def fetch_text_asset_ids(conn) -> set[str]:
    """registered 중 자기 텍스트 보유 자산 id 집합(읽기전용·부여율 분모)."""
    with conn.cursor() as cur:
        cur.execute(_TEXT_ASSET_SQL)
        return {str(r[0]) for r in cur.fetchall()}


def fetch_registered_count(conn) -> int:
    """registered 자산 총수(읽기전용)."""
    with conn.cursor() as cur:
        cur.execute(_REGISTERED_COUNT_SQL)
        return int(cur.fetchone()[0])


def fetch_projection_rows(conn) -> list[dict[str, Any]]:
    """(참고) 옛 이웃-엣지 투영 재구성 평탄 행(읽기전용·--compare-projection)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_PROJECTION_SQL)
        rows = cur.fetchall()
    return [
        {
            "asset_id": str(r["asset_id"]),
            "topic_ko": r["topic_ko"],
            "subtopic_ko": r["subtopic_ko"],
            "self_text": _build_self_text(r),
        }
        for r in rows
    ]


def run_report(*, env: str, compare_projection: bool = False) -> dict[str, Any]:
    """리포트 실행(읽기전용 DB·LLM 0). .env.{env} 로드 → init_settings → 집계."""
    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    dotenv_path = _REPO_ROOT / f".env.{env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(env)

    db = PostgresUtil()
    with db, db.connection() as conn:
        grounding_flat = fetch_grounding_rows(conn)
        text_ids = fetch_text_asset_ids(conn)
        n_reg = fetch_registered_count(conn)
        proj_flat = fetch_projection_rows(conn) if compare_projection else None

    # topic_rows 는 grounding_flat 에서 파생(2차 쿼리 회피).
    topic_rows = [
        {"asset_id": r["asset_id"], "topic_ko": r["topic_ko"], "subtopic_ko": r["subtopic_ko"]}
        for r in grounding_flat
    ]
    report: dict[str, Any] = {
        "env": env,
        "distribution": build_distribution_report(
            topic_rows=topic_rows, text_asset_ids=text_ids, n_registered=n_reg
        ),
        "grounding": build_grounding_report(group_label_rows(grounding_flat)),
    }
    if proj_flat is not None:
        report["projection_grounding"] = build_grounding_report(group_label_rows(proj_flat))
    return report


def main() -> int:
    p = argparse.ArgumentParser(
        description="자기주제 근거율(grounding)·분포 리포트(spec 065·FR-602·읽기전용)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument(
        "--compare-projection", dest="compare_projection", action="store_true",
        help="(참고) 옛 이웃-엣지 투영 방식이었다면 오염율이 얼마였을지 비교",
    )
    p.add_argument("--json", dest="json_out", default=None, help="리포트 JSON 저장 경로(선택)")
    args = p.parse_args()

    report = run_report(env=args.env, compare_projection=args.compare_projection)
    print("\n".join(format_report_lines(report)))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  리포트 JSON 저장: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
