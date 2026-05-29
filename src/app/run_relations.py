"""F-3.5 관계 제안 배치 CLI — registered 자산에 대해 ``graph_edge`` 엣지를 생성한다(관계 카탈로그는 relation_kind).

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
from src.pipeline.packs import GENERAL_PACK, for_domain
from src.relations.asset_candidates import EmbeddingKindFilter
from src.relations.asset_entry import propose_relations_for_asset

_LOG = logging.getLogger("meta_extract.run_relations")


def _fetch_domain_label(db: PostgresUtil, asset_id: str) -> str:
    """자산의 domain_label(분류 결과). 없으면 'general'.

    NULL 이나 자산 미존재는 모두 'general' 로 폴백한다. for_domain 의 보수적 폴백 정책과 일치.
    """

    def _run(conn: Connection[Any]) -> str:
        with conn.cursor() as cur:
            cur.execute("SELECT domain_label FROM asset WHERE asset_id = %s LIMIT 1", (asset_id,))
            row = cur.fetchone()
        return str(row[0]) if row and row[0] else "general"

    return db.execute_in_transaction(_run, idempotent=True)


def _fetch_registered_asset_ids(db: PostgresUtil) -> list[str]:
    """임베딩을 가진 ``registered`` 자산 id 전부(생성순).

    **필터 조건**: status='registered' + asset_embedding 존재.
    임베딩이 없으면 candidates 단계에서 후보 자체가 나오지 않아 관계 제안이 무의미하므로 제외한다.
    deferred/failed 상태 자산은 임베딩 미적재이므로 이 조건으로 자연히 걸러진다.
    """

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
    """자산 리스트 순회하며 관계 제안. 자산 단위 예외 흡수.

    반환값 구조: {"done": [(aid, edges_upserted, edges_kept), ...], "failed": [(aid, msg), ...]}.
    배치 전체 성공 여부는 ``failed`` 리스트가 비어있는지로 판별한다(main 의 종료코드 참조).
    """
    result: dict[str, list[Any]] = {"done": [], "failed": []}
    for aid in asset_ids:
        try:
            domain = _fetch_domain_label(db, aid)
            pack = for_domain(domain)
            # 일반 묶음(_GENERAL_CROSS)을 가리키는 팩은 propose_relations_for_asset 로 위임한다.
            # 의료도 현재 동일 묶음이라 위임됨(stopgap: 의료 자산은 일반 추출/임베딩 경로 사용).
            # **내용 비교가 의도**다 — 단계 D 에서 의료가 전용 cross_asset 묶음을 갖게 되면 이 비교가
            # 트립되어 슬롯별 전략 배선을 강제한다(미배선 방지 forward 가드). 팩 이름 비교로 바꾸면
            # 일반 묶음을 쓰는 다른 도메인까지 잘못 막으므로 안 된다.
            if pack.cross_asset != GENERAL_PACK.cross_asset:
                raise NotImplementedError(f"cross_asset 전략 미구현(도메인 {pack.name})")
            cat_s, cat_k, edges_u, edges_k = propose_relations_for_asset(
                db, aid, top_k=top_k, embedding_kind=embedding_kind
            )
            result["done"].append((aid, edges_u, edges_k))
            _LOG.info(
                "relations %s: kinds=%s/%s edges=%s/%s", aid, cat_s, cat_k, edges_u, edges_k
            )
        except Exception as exc:  # noqa: BLE001 — 자산 단위 격리
            _LOG.warning("relations failed %s: %s", aid, exc)
            result["failed"].append((aid, str(exc)))
    return result


# ── 초기 설정(부트스트랩) 절차 ────────────────────────────────────────────────
# [import 시점] 이 진입점은 전략 레지스트리·도메인 프로파일을 로드하지 않는다
#   (builtins/cascade import 없음). 관계 제안은 propose_relations_for_asset 로 묶음 위임하며
#   registry.resolve 를 쓰지 않는다(단계 D 의료 분기에서 팩별 slot resolve 로 전환 예정).
# [런타임·main() 안·순서 중요]:
#   1) load_dotenv(.env.{env}, override=False)  2) init_settings(env): 필수 환경변수 검증+frozen 설정
#   3) PostgresUtil() + `with db:`: 연결 풀+PG17 검증.  온프레미스 LLM 클라이언트는 propose 내부 첫 호출 시 지연 생성.
def main() -> int:
    """CLI: registered 자산에 대해 관계 제안 배치."""
    import argparse
    import json
    from pathlib import Path

    from dotenv import load_dotenv

    from src.config.settings import init_settings

    parser = argparse.ArgumentParser(description="관계 제안 배치 (graph_edge 적재)")
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
