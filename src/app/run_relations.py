"""F-3.5 관계 제안 배치 CLI — registered 자산에 대해 ``asset_relation`` 엣지를 생성한다.

수집/추출(run_ingest)과 분리된 후속 배치다. 임베딩이 있는 ``status='registered'`` 자산만 대상으로
``propose_relations_for_asset`` 을 호출한다(온프레미스 LLM). 자산 단위 격리(한 건 실패가 배치를 멈추지 않음).

예)
    python -m src.app.run_relations --env dev --all
    python -m src.app.run_relations --env dev <asset_uuid> <asset_uuid> ...
"""

from __future__ import annotations

import logging
from typing import Any

from psycopg import Connection

from src.database.postgres_util import PostgresUtil
from src.relations.asset_candidates import EmbeddingKindFilter
from src.relations.asset_entry import propose_relations_for_asset

_LOG = logging.getLogger("meta_extract.run_relations")


def _fetch_registered_asset_ids(db: PostgresUtil) -> list[str]:
    """임베딩을 가진 ``registered`` 자산 id 전부(생성순)."""

    def _run(conn: Connection[Any]) -> list[str]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.asset_id
                FROM asset a
                WHERE a.status = 'registered'
                  AND EXISTS (SELECT 1 FROM asset_embedding e WHERE e.asset_id = a.asset_id)
                ORDER BY a.created_at
                """
            )
            return [str(r[0]) for r in cur.fetchall()]

    return db.execute_in_transaction(_run, idempotent=True)


def run_relations(
    asset_ids: list[str],
    *,
    db: PostgresUtil,
    top_k: int | None = None,
    embedding_kind: EmbeddingKindFilter = "both",
) -> dict[str, list[Any]]:
    """자산 리스트 순회하며 관계 제안. 자산 단위 예외 흡수."""
    result: dict[str, list[Any]] = {"done": [], "failed": []}
    for aid in asset_ids:
        try:
            cat_s, cat_k, edges_u, edges_k = propose_relations_for_asset(
                db, aid, top_k=top_k, embedding_kind=embedding_kind
            )
            result["done"].append((aid, edges_u, edges_k))
            _LOG.info(
                "relations %s: catalog=%s/%s edges=%s/%s", aid, cat_s, cat_k, edges_u, edges_k
            )
        except Exception as exc:  # noqa: BLE001 — 자산 단위 격리
            _LOG.warning("relations failed %s: %s", aid, exc)
            result["failed"].append((aid, str(exc)))
    return result


def main() -> int:
    """CLI: registered 자산에 대해 관계 제안 배치."""
    import argparse
    import json
    from pathlib import Path

    from dotenv import load_dotenv

    from src.config.settings import init_settings

    parser = argparse.ArgumentParser(description="관계 제안 배치 (asset_relation 적재)")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--all", action="store_true", help="registered 자산 전체 대상")
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument(
        "--embedding-kind", dest="embedding_kind", choices=["st", "clip", "both"], default="both"
    )
    parser.add_argument("asset_ids", nargs="*", metavar="ASSET_ID")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    db = PostgresUtil()
    with db:
        if args.all:
            asset_ids = _fetch_registered_asset_ids(db)
        else:
            asset_ids = list(args.asset_ids)
        if not asset_ids:
            print(json.dumps({"done": 0, "failed": 0}, ensure_ascii=False))
            return 0
        result = run_relations(
            asset_ids, db=db, top_k=args.top_k, embedding_kind=args.embedding_kind
        )
    print(json.dumps({k: len(v) for k, v in result.items()}, ensure_ascii=False))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
