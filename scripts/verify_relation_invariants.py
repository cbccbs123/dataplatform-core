"""관계 그래프 **전수 기계 불변식** 검사 — spec 081 SC-08 ②.

**무엇을 하는가**: 전량 재실행 뒤 "게이트가 실제로 지켜졌나"를 표본이 아니라 **전 행**에서 센다.
각 검사는 "위반 행 수"를 돌려주고, 전부 0이어야 통과다.

**무엇을 하지 않는가**: 관계가 **타당한지**(내용)는 보지 않는다. 그건 골든(`tests/golden/relations`)과
층화 표본 판정(`scripts/judge_relations.py`)의 일이다. 이 경계를 흐리면 "기계 검사 통과 = 품질 좋음"
이라는 잘못된 결론이 나온다 — 기계 불변식은 **전수**, 내용 판정은 **골든+표본**이다.

⚠️ **후보 밖 타깃(LLM 환각)은 여기서 검사할 수 없다.** 후보 집합은 영속화되지 않으므로 사후에
"그 타깃이 후보였는지" 확인할 방법이 없다(그 게이트는 `graph_persist` 실행 시점에만 판정 가능하고,
`tests/test_graph_persist_approval.py`·`scripts/judge_snapshot.py:collect_pairs` 가 대신 지킨다).
계획서에 이 항목을 적었으나 검증 불가로 제외했다 — 못 재는 것을 잰 것처럼 두면 안 된다.

읽기 전용 — 모든 SQL 이 SELECT 다(테스트가 봉인한다).

실행
    conda activate AuroraFS
    python scripts/verify_relation_invariants.py --env dev
    python scripts/verify_relation_invariants.py --env dev --fail-on-violation   # CI·게이트용
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.domain.status_vocab import GraphEdgeStatus  # noqa: E402
from src.relations.approval_policy import SIMILARITY_KINDS, parse_kind_set  # noqa: E402

# 검사 이름의 정본 순서 — 테스트가 이 목록과 `build_checks` 결과의 일치를 봉인한다.
# 조건부 검사(게이트를 끄면 빠지는 것)는 앞쪽 둘이다.
CHECK_NAMES: tuple[str, ...] = (
    "유사도_계열_저신뢰_잔존",
    "자동승인_제외_kind가_active",
    "자기참조_엣지",
    "닫힌_status_어휘_위반",
    "대칭_중복행",
    "대칭_캐논순서_위반",
    "비활성_kind_엣지",
    "비-asset_노드_참조",
)


def build_checks(
    *,
    min_conf_similarity: float,
    exclude_kinds: frozenset[str],
) -> list[dict[str, Any]]:
    """불변식 검사 목록을 만든다(순수 함수 · DB 접속 없음).

    게이트를 **끈 설정**에서는 해당 검사를 목록에서 뺀다 — 끈 게이트의 "위반"을 보고하면
    거짓 경보이고, 거짓 경보가 한 번 나오면 다음부터 아무도 이 도구를 믿지 않는다.

    Args:
        min_conf_similarity: 유사도 계열 영속화 하한. ``0`` 이하면 그 검사를 만들지 않는다.
        exclude_kinds: 자동승인 제외 종류. 비면 그 검사를 만들지 않는다.

    Returns:
        ``{name, sql, params}`` 리스트. ``sql`` 은 위반 행 수를 ``n`` 으로 돌려주는 SELECT 이고,
        본문에 ``-- check:<name>`` 주석을 심어 둔다(로그·테스트에서 어느 검사인지 식별).
    """
    checks: list[dict[str, Any]] = []
    sim = sorted(SIMILARITY_KINDS)
    statuses = sorted(s.value for s in GraphEdgeStatus)

    if min_conf_similarity > 0.0:
        checks.append({
            "name": "유사도_계열_저신뢰_잔존",
            "sql": """SELECT count(*) AS n
FROM graph_edge ge
JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
WHERE rk.kind_code = ANY(%s)
  AND (ge.confidence IS NULL OR ge.confidence < %s)
-- check:유사도_계열_저신뢰_잔존""",
            "params": [sim, min_conf_similarity],
        })

    if exclude_kinds:
        checks.append({
            "name": "자동승인_제외_kind가_active",
            "sql": """SELECT count(*) AS n
FROM graph_edge ge
JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
WHERE rk.kind_code = ANY(%s) AND ge.status = 'active'
-- check:자동승인_제외_kind가_active""",
            "params": [sorted(exclude_kinds)],
        })

    checks.append({
        "name": "자기참조_엣지",
        "sql": """SELECT count(*) AS n FROM graph_edge WHERE src_node = dst_node
-- check:자기참조_엣지""",
        "params": [],
    })
    checks.append({
        "name": "닫힌_status_어휘_위반",
        "sql": """SELECT count(*) AS n FROM graph_edge WHERE status <> ALL(%s)
-- check:닫힌_status_어휘_위반""",
        "params": [statuses],
    })
    # 대칭 kind 는 (min,max) 캐논 순서 1행으로만 저장된다(graph_persist._canonical_pair).
    # 같은 쌍이 두 행으로 갈라지면 조회가 이중으로 세고 검토도 두 번 하게 된다.
    checks.append({
        "name": "대칭_중복행",
        "sql": """SELECT count(*) AS n FROM (
  SELECT least(ge.src_node::text, ge.dst_node::text) AS a,
         greatest(ge.src_node::text, ge.dst_node::text) AS b,
         ge.relation_kind_id
  FROM graph_edge ge
  JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
  WHERE rk.is_symmetric
  GROUP BY 1, 2, 3 HAVING count(*) > 1
) t
-- check:대칭_중복행""",
        "params": [],
    })
    checks.append({
        "name": "대칭_캐논순서_위반",
        "sql": """SELECT count(*) AS n
FROM graph_edge ge
JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
WHERE rk.is_symmetric AND ge.src_node::text > ge.dst_node::text
-- check:대칭_캐논순서_위반""",
        "params": [],
    })
    # 미검토(inactive) kind 는 엣지가 되지 않아야 한다 — graph_persist 가 active kind 만 통과시킨다.
    checks.append({
        "name": "비활성_kind_엣지",
        "sql": """SELECT count(*) AS n
FROM graph_edge ge
JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
WHERE rk.status <> 'active'
-- check:비활성_kind_엣지""",
        "params": [],
    })
    # 양 끝은 asset 노드여야 한다. entity 노드(단계 D 의료 ER)는 asset_id 가 NULL 이므로
    # 섞이면 조회에서 None 자산이 튀어나온다.
    checks.append({
        "name": "비-asset_노드_참조",
        "sql": """SELECT count(*) AS n
FROM graph_edge ge
JOIN node n1 ON n1.node_id = ge.src_node
JOIN node n2 ON n2.node_id = ge.dst_node
WHERE n1.node_kind <> 'asset' OR n2.node_kind <> 'asset'
-- check:비-asset_노드_참조""",
        "params": [],
    })
    return checks


def run_verify(db: Any, *, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """검사를 전부 실행해 위반 수를 집계한다(읽기 전용).

    **하나도 건너뛰지 않는다** — 일부만 돌고 "통과"를 내면 그 통과가 거짓이 된다.

    Args:
        db: DB 핸들(``transaction()`` → ``conn.cursor(row_factory=dict_row)``).
        checks: ``build_checks`` 결과.

    Returns:
        ``{ok, violations, results, failed}``. ``results`` 는 검사 순서대로
        ``{name, count}``, ``failed`` 는 그중 ``count>0`` 인 것만.
    """
    results: list[dict[str, Any]] = []
    with db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
        for c in checks:
            cur.execute(c["sql"], tuple(c["params"]) if c["params"] else None)
            row = cur.fetchone()
            # ⚠️ dict_row 커서다 — dict(cur.fetchall()) 류로 감싸면 키/값이 뒤집힌다.
            results.append({"name": c["name"], "count": int(row["n"] if row else 0)})
    failed = [r for r in results if r["count"] > 0]
    return {"ok": not failed, "violations": sum(r["count"] for r in failed),
            "results": results, "failed": failed}


def format_report(report: dict[str, Any], *, scanned: int) -> list[str]:
    """사람이 읽는 보고 줄을 만든다(순수 함수).

    Args:
        report: ``run_verify`` 결과.
        scanned: 검사 대상 엣지 총 행수(맥락 — 0건 통과와 전수 통과를 구분하려고).

    Returns:
        출력할 줄 목록.
    """
    lines = [f"관계 불변식 전수 검사 — graph_edge {scanned}행 대상 · 검사 {len(report['results'])}종"]
    for r in report["results"]:
        mark = "✅" if r["count"] == 0 else "🔴"
        lines.append(f"  {mark} {r['name']}: {r['count']}건")
    if report["ok"]:
        lines.append(f"✅ 위반 0 — 전수 통과({scanned}행)")
    else:
        lines.append(f"🔴 위반 {report['violations']}건 — "
                     + ", ".join(f"{v['name']}({v['count']})" for v in report["failed"]))
    return lines


def main(argv: list[str] | None = None) -> int:
    """설정에서 게이트 값을 읽어 전수 검사를 돌린다.

    Args:
        argv: 명령행 인자. ``None`` 이면 실제 인자를 읽는다(테스트 주입용).

    Returns:
        0=통과. ``--fail-on-violation`` 과 함께 위반이 있으면 1(CI·게이트용).
    """
    ap = argparse.ArgumentParser(description="관계 그래프 전수 기계 불변식 검사(읽기 전용)")
    ap.add_argument("--env", choices=["dev", "prod"], default="dev",
                    help="설정 프로파일(기본: dev). .env.<env> 를 읽어 초기화한다")
    ap.add_argument("--fail-on-violation", action="store_true",
                    help="위반이 있으면 종료코드 1(기본은 보고만 하고 0)")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv

    from src.config.settings import get_current_settings, init_settings
    from src.database.postgres_util import PostgresUtil

    dotenv_path = _REPO_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)
    rel = get_current_settings().relations

    checks = build_checks(
        min_conf_similarity=rel.persist_min_conf_similarity,
        exclude_kinds=parse_kind_set(rel.auto_approve_exclude_kinds, default=frozenset()))
    print(f"게이트 설정 — persist_min={rel.persist_min_conf_similarity} · "
          f"exclude_kinds={rel.auto_approve_exclude_kinds!r}")

    db = PostgresUtil()
    with db:
        with db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT count(*) AS n FROM graph_edge")
            scanned = int(cur.fetchone()["n"])
        report = run_verify(db, checks=checks)

    print("\n".join(format_report(report, scanned=scanned)))
    if not report["ok"] and args.fail_on_violation:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
