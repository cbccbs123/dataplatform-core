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
from src.relations.resolution_persist import (
    decide_resolution_status,
    fetch_unresolved_asset_ids,
    upsert_resolution,
)

_LOG = logging.getLogger("meta_extract.run_relations")

# 큐 last_reason 비식별 표식(헌법 10조 — PHI/풀경로 금지).
# 고립(엣지0)은 표식 1개, 예외는 예외 **타입명**만 기록한다(메시지·경로 미포함).
_REASON_ISOLATED = "isolated:no_edges"


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


def _fetch_unresolved_asset_ids(db: PostgresUtil) -> list[str]:
    """``--retry`` 대상 자산 id — pending(미해소) + 미시도만, 결정적 정렬.

    베이스(status='registered' + 임베딩 존재)는 ``_fetch_registered_asset_ids`` 와 공유하고,
    ``relation_resolution`` LEFT JOIN 으로 (rr.status='pending' OR rr.asset_id IS NULL) 만 고른다.
    resolved/failed(DLQ) 자산은 제외된다. 정렬은 attempts ASC, created_at ASC(결정적, 헌법 3조).
    실제 쿼리는 resolution_persist.fetch_unresolved_asset_ids(conn-우선)에 위임한다.
    """

    def _run(conn: Connection[Any]) -> list[str]:
        return fetch_unresolved_asset_ids(conn)

    return db.execute_in_transaction(_run, idempotent=True)


def _record_resolution(
    db: PostgresUtil, aid: str, edges_upserted: int, *, error: Exception | None, max_attempts: int
) -> None:
    """한 자산 처리 결과를 큐에 반영 — **별도 fresh 트랜잭션**으로 격리(run_ingest 패턴 차용).

    핵심(SC-008): 한 자산의 큐 upsert 실패가 다른 자산 처리나 이미 적재된 관계를 롤백하면 안 된다.
    그래서 큐 갱신은 propose_relations_for_asset 트랜잭션 **밖**, 자산별 독립 fresh 트랜잭션에서 수행하고,
    여기서 또 예외가 나면 로그만 남기고 흡수한다(배치·다른 자산에 전파 금지).

    last_reason 은 비식별만(헌법 10조): 고립은 _REASON_ISOLATED 표식, 예외는 예외 **타입명**만.
    예외 메시지·파일 경로는 PHI/풀경로 누출 위험이 있어 큐에 담지 않는다.
    """
    cur_attempts = _fetch_attempts(db, aid)
    status, next_attempts = decide_resolution_status(
        edges_upserted, cur_attempts, error=error, max_attempts=max_attempts
    )
    reason = type(error).__name__ if error is not None else _REASON_ISOLATED
    try:
        def _run(conn: Connection[Any]) -> None:
            upsert_resolution(conn, aid, status=status, attempts=next_attempts, reason=reason)

        db.execute_in_transaction(_run, idempotent=False)
    except Exception as exc:  # noqa: BLE001 — 큐 갱신 실패 격리(다른 자산 미롤백)
        _LOG.warning("resolution queue upsert failed %s: %s", aid, exc)


def _fetch_attempts(db: PostgresUtil, aid: str) -> int:
    """자산의 현재 큐 attempts(없으면 0). decide_resolution_status 입력으로 쓴다."""

    def _run(conn: Connection[Any]) -> int:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT attempts FROM relation_resolution WHERE asset_id = %s LIMIT 1", (aid,)
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    return db.execute_in_transaction(_run, idempotent=True)


def run_relations(
    asset_ids: list[str],
    *,
    db: PostgresUtil,
    top_k: int | None = None,
    embedding_kind: EmbeddingKindFilter = "both",
    max_attempts: int | None = None,
) -> dict[str, list[Any]]:
    """자산 리스트 순회하며 관계 제안. 자산 단위 예외 흡수 + 미해소 큐 갱신.

    반환값 구조: {"done": [(aid, edges_upserted, edges_kept), ...], "failed": [(aid, msg), ...]}.
    배치 전체 성공 여부는 ``failed`` 리스트가 비어있는지로 판별한다(main 의 종료코드 참조).

    각 자산 처리 직후 결과(edges_upserted/예외)로 ``decide_resolution_status`` 를 호출해
    relation_resolution 큐를 **별도 fresh 트랜잭션**으로 갱신한다(자산별 격리, SC-008).
    ``max_attempts`` 미지정 시 현재 설정 ``relation_retry_max_attempts`` 를 사용한다.
    """
    if max_attempts is None:
        from src.config.settings import get_current_settings

        max_attempts = get_current_settings().relation_retry_max_attempts
    result: dict[str, list[Any]] = {"done": [], "failed": []}
    for aid in asset_ids:
        edges_u = 0
        err: Exception | None = None
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
            err = exc
            _LOG.warning("relations failed %s: %s", aid, exc)
            result["failed"].append((aid, str(exc)))
        # 자산 처리 결과를 큐에 반영(엣지0/예외/성공 모두). fresh 트랜잭션 격리.
        _record_resolution(db, aid, edges_u, error=err, max_attempts=max_attempts)
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
    parser.add_argument(
        "--retry",
        action="store_true",
        help="미해소(pending) + 미시도 자산만 골라 재시도(relation_resolution 큐 기반). --all 동시 지정 시 --retry 우선",
    )
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
        # 선택 분기: --retry(미해소+미시도) > --all(registered 전체) > 명시 asset_ids.
        # --all·명시 asset_ids 경로는 무변경(회귀 0) — --retry 만 큐 기반 LEFT JOIN 선택을 쓴다.
        if args.retry:
            asset_ids = _fetch_unresolved_asset_ids(db)
        elif args.all:
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
