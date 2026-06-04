"""F-3.5 관계 제안 배치 CLI — registered 자산에 대해 ``graph_edge`` 엣지를 생성한다(관계 카탈로그는 relation_kind).

수집/추출(run_ingest)과 분리된 후속 배치다. 임베딩이 있는 ``status='registered'`` 자산만 대상으로
``propose_relations_for_asset`` 을 호출한다(온프레미스 LLM). 자산 단위 격리(한 건 실패가 배치를 멈추지 않음).

예)
    python -m src.app.run_relations --env dev --all
    python -m src.app.run_relations --env dev <asset_uuid> <asset_uuid> ...
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from psycopg import Connection

from src.database.postgres_util import PostgresUtil

# builtins import 부수효과로 DEFAULT_REGISTRY 에 cross_asset 전략이 등록된다(register_defaults).
# 슬롯 resolve(_resolve_cross_asset_slots)가 빈 레지스트리를 만나지 않도록 진입부에서 강제 로드.
# run_ingest 와 동일 관용(별칭 _builtins) — 부수효과 import 라 직접 참조하지 않는다.
from src.pipeline import builtins as _builtins  # noqa: F401 — DEFAULT_REGISTRY 등록 부수효과
from src.pipeline.packs import GENERAL_PACK, DomainPack, for_domain
from src.pipeline.registry import DEFAULT_REGISTRY, StrategyRegistry
from src.relations.asset_candidates import EmbeddingKindFilter
from src.relations.asset_entry import propose_relations_for_asset
from src.relations.resolution_persist import (
    decide_resolution_status,
    fetch_unresolved_asset_ids,
    upsert_resolution,
)

_LOG = logging.getLogger("meta_extract.run_relations")

# cross_asset 슬롯 중 레지스트리에서 Callable 로 resolve 되는 슬롯(008).
# 'decide'(confidence)는 propose_relations_for_asset 내부 auto_approve 임계로 처리되어
# 별도 Callable 이 등록돼 있지 않으므로 resolve 대상에서 제외한다(packs.py·builtins.py 주석 참조).
_RESOLVED_CROSS_SLOTS: tuple[str, ...] = ("candidates", "score", "persist_edges")

# 큐 last_reason 비식별 표식(009, 헌법 10조 — PHI/풀경로 금지).
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


def _resolve_cross_asset_slots(
    pack: DomainPack, *, registry: StrategyRegistry = DEFAULT_REGISTRY
) -> dict[str, Callable[..., Any]]:
    """팩의 cross_asset 슬롯명을 레지스트리에서 Callable 로 resolve(미배선 가드, FR-002).

    **목적(헌법 4조)**: 도메인 차이를 코드 if/else 가 아니라 "팩이 고른 전략 이름"으로 표현한다.
    여기서 슬롯명을 registry.resolve 로 검증함으로써, 의료 ER(단계 D)이 전용 cross_asset
    전략(예: blocking_5keys)을 **레지스트리에 등록만 하면** core 파이프라인 수정 없이 갈리는
    자리를 만든다. 등록 전 상태(미배선)면 KeyError → NotImplementedError 로 승격해 자산 단위
    격리(run_relations 의 except)로 흘려보낸다 — 배치는 중단되지 않는다.

    'decide'(confidence)는 propose_relations_for_asset 내부 auto_approve 임계로 처리되어 별도
    Callable 등록이 없으므로 resolve 대상(_RESOLVED_CROSS_SLOTS)에서 제외한다.

    Returns:
        슬롯 이름 → resolve 된 Callable. (검증 통과 시에만 반환)
    Raises:
        NotImplementedError: 슬롯이 가리키는 전략이 레지스트리에 미등록(단계 D 전 의료 등).
    """
    resolved: dict[str, Callable[..., Any]] = {}
    for slot in _RESOLVED_CROSS_SLOTS:
        name = pack.cross_asset.get(slot)
        if name is None:
            raise NotImplementedError(
                f"cross_asset 슬롯 '{slot}' 미정의(도메인 {pack.name})"
            )
        try:
            resolved[slot] = registry.resolve(slot, name)
        except KeyError as e:
            # 미등록 전략 — 의료 cross_asset 전략이 단계 D 전이라 배선되지 않은 경우 등.
            raise NotImplementedError(
                f"cross_asset 전략 미구현(도메인 {pack.name}): {slot}={name}"
            ) from e
    return resolved


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

    **cross_asset 슬롯 resolve(008, FR-001~003)**: 자산 domain_label → for_domain 으로 팩을 고르고,
    팩의 cross_asset 슬롯을 _resolve_cross_asset_slots 로 레지스트리에서 resolve 한다(미배선 가드).
    일반 팩(슬롯이 GENERAL_PACK.cross_asset 과 동일)은 결과 동치(회귀 0, FR-001/SC-001)를
    구조적으로 보장하기 위해 기존 propose_relations_for_asset 로 위임한다 — 슬롯 전환의 목적은
    "의료 전용 전략이 끼워질 자리 만들기"이지 일반 동작을 바꾸는 것이 아니다.

    **미해소 큐 갱신(009, SC-008)**: 각 자산 처리 직후 결과(edges_upserted/예외)로
    ``decide_resolution_status`` 를 호출해 relation_resolution 큐를 **별도 fresh 트랜잭션**으로
    갱신한다(자산별 격리). ``max_attempts`` 미지정 시 설정 ``relation_retry_max_attempts`` 를 쓴다.
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
            pack = for_domain(domain)  # 미지정/review → GENERAL_PACK 폴백(FR-003)
            # 슬롯 resolve(FR-002): 미배선 전략은 NotImplementedError → 아래 except 에서 failed 격리.
            # 일반 팩은 전부 등록돼 있어 통과하고, 의료가 전용 전략을 등록(단계 D)하면 그 전략으로 갈린다.
            _resolve_cross_asset_slots(pack)
            if pack.cross_asset == GENERAL_PACK.cross_asset:
                # 일반 cross_asset 묶음 — 기존 경로에 위임해 결과를 100% 동치로 유지(FR-001, 헌법 8조).
                # 슬롯별 전략을 잘게 호출하는 대신 검증된 propose_relations_for_asset 를 재사용한다.
                cat_s, cat_k, edges_u, edges_k = propose_relations_for_asset(
                    db, aid, top_k=top_k, embedding_kind=embedding_kind
                )
            else:
                # 일반 묶음이 아닌 전용 cross_asset 전략(단계 D 의료 등): 슬롯이 모두 등록돼 있어도
                # 묶음 실행 어댑터가 아직 없으므로 명시적 미구현으로 격리한다(자리만 확보된 상태).
                raise NotImplementedError(
                    f"전용 cross_asset 묶음 실행 미구현(도메인 {pack.name}) — 단계 D"
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
# [import 시점] 이 진입점은 src.pipeline.builtins 를 import 해 DEFAULT_REGISTRY 에 cross_asset
#   전략(candidates/score/persist_edges)을 등록한다(register_defaults 부수효과). run_relations 가
#   팩의 cross_asset 슬롯을 registry.resolve 로 검증(미배선 가드)하기 때문이다. 일반 팩은 검증 통과
#   후 propose_relations_for_asset 로 위임해 결과를 기존과 동치로 유지한다(FR-001).
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
