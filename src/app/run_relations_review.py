"""관계 HITL 검토 CLI — proposed 엣지 큐 조회·승인/반려 + relation_kind 승격.

**HITL(Human-In-The-Loop) 흐름**:
run_relations(LLM 제안) → graph_edge.status='proposed' 적재 →
이 CLI 로 사람이 검토 → approved(active) 또는 rejected 전환.
자동 승인(auto_approve) 임계를 통과한 엣지는 이미 'active' 상태이므로 --list 에 나타나지 않는다.

**동작별 트랜잭션 범위**: 각 동작(--approve/--reject/--promote-kind)은 단일 트랜잭션으로 처리된다.
idempotent=False — 동일 edge_id 를 두 번 approve 하면 두 번째는 ok=False 를 반환한다.

예)
    python -m src.app.run_relations_review --env dev --list
    python -m src.app.run_relations_review --env dev --approve <edge_id> --reviewer bc
    python -m src.app.run_relations_review --env dev --reject <edge_id> --reviewer bc
    python -m src.app.run_relations_review --env dev --promote-kind gaming_hardware --reviewer bc
"""
from __future__ import annotations

from typing import Any

from psycopg import Connection

from src.database.postgres_util import PostgresUtil
from src.relations.review import (
    approve_edge,
    list_proposed_edges,
    promote_relation_kind,
    reject_edge,
)


def main() -> int:
    """CLI: proposed 엣지 큐 조회 / 승인 / 반려 / relation_kind 승격."""
    import argparse
    import json
    from pathlib import Path

    from dotenv import load_dotenv

    from src.config.settings import init_settings

    parser = argparse.ArgumentParser(description="관계 HITL 검토 (proposed 엣지 승인/반려·kind 승격)")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--reviewer", default="cli")  # 감사 로그용 검토자 식별자
    parser.add_argument("--limit", type=int, default=100, help="--list 출력 상한")
    # 동작은 상호 배타(하나만 지정 가능).
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="proposed 엣지 큐 출력")
    group.add_argument("--approve", metavar="EDGE_ID", help="엣지 승인(active)")
    group.add_argument("--reject", metavar="EDGE_ID", help="엣지 반려(rejected)")
    group.add_argument("--promote-kind", dest="promote_kind", metavar="KIND_CODE",
                       help="relation_kind 승격(inactive→active)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    db = PostgresUtil()
    with db:
        if args.list:
            def _run(conn: Connection[Any]) -> list[dict[str, Any]]:
                return list_proposed_edges(conn, limit=args.limit)
            rows = db.execute_in_transaction(_run, idempotent=True)
            print(json.dumps({"proposed": rows}, ensure_ascii=False, default=str))
            return 0

        if args.approve:
            def _run(conn: Connection[Any]) -> bool:
                return approve_edge(conn, edge_id=args.approve, reviewer=args.reviewer)
            ok = db.execute_in_transaction(_run, idempotent=False)
            print(json.dumps({"approved": args.approve, "ok": ok}, ensure_ascii=False))
            return 0 if ok else 1

        if args.reject:
            def _run(conn: Connection[Any]) -> bool:
                return reject_edge(conn, edge_id=args.reject, reviewer=args.reviewer)
            ok = db.execute_in_transaction(_run, idempotent=False)
            print(json.dumps({"rejected": args.reject, "ok": ok}, ensure_ascii=False))
            return 0 if ok else 1

        def _run(conn: Connection[Any]) -> bool:
            return promote_relation_kind(conn, kind_code=args.promote_kind, reviewer=args.reviewer)
        ok = db.execute_in_transaction(_run, idempotent=False)
        print(json.dumps({"promoted_kind": args.promote_kind, "ok": ok}, ensure_ascii=False))
        return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
