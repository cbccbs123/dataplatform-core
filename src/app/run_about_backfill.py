"""aboutness 개체 백필 배치 CLI — 저장된 summary 로 ``ext_meta['about']`` 소급 확정(spec 073·FR-002).

무엇을 하나
    수집(run_ingest)은 이제 자산마다 aboutness 개체("무엇에 관한 것인가" 명사 1~3개)를 즉시 확정하지만,
    073 배선 **이전** 자산은 ``ext_meta`` 에 about 키가 없다. 이 배치는 재수집 없이 저장된 summary 로
    ``extract_and_persist_about`` 을 일괄 호출해 소급 확정하고, 대상 자산의 OS 문서 ``about`` 필드를
    부분 갱신한다(``update_asset_about`` — 전체 재색인 불요·구 인덱스엔 ``ensure_about_mapping`` 1회).

    - **대상**: ``status='registered'`` + ``asset_metadata`` 존재(의료 제외). ``--only-missing``(기본)
      이면 ``ext_meta->'about'`` 키가 없는 자산만 — 재실행 멱등(빈 [] 도 "시도함" 기록이라 스킵됨).
    - **자산별 짧은 트랜잭션 + 격리**: 한 자산의 예외는 삼켜(failed 카운트) 배치를 멈추지 않는다.
    - **결정적 순서**(헌법 3조): ``asset_id`` 오름차순.

실행 (backfill 실행은 사람 게이트 — 065 run_topic_backfill 과 동일 관례)
    conda activate AuroraFS
    python -m src.app.run_about_backfill --env dev --report          # 현황만(쓰기 0)
    python -m src.app.run_about_backfill --env dev --all             # about 미확정 자산 전체
    python -m src.app.run_about_backfill --env dev --all --limit 100 # 앞 100건(배치·재개)
    python -m src.app.run_about_backfill --env dev --all --no-only-missing  # 재추출(--refresh 동의어)

설계 주의
    코어(``backfill_about``)는 추출·OS 갱신을 seam(``extract_persist_fn``/``os_update_fn``)으로
    주입받아 실 DB/LLM 없이 순수 단위 검증한다. IO 부트스트랩은 얇게(run_topic_backfill 동형).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from typing import Any

_LOG = logging.getLogger("meta_extract.run_about_backfill")

# 대상 스캔 — registered + 메타 보유 + 비의료. only_missing 이면 about 키 부재만(멱등 재실행).
_TARGET_SQL = """
SELECT a.asset_id, m.ext_meta->>'summary' AS summary
FROM asset a
JOIN asset_metadata m ON m.asset_id = a.asset_id
WHERE a.status = 'registered' AND a.domain_label IS DISTINCT FROM 'medical'
  {missing_clause}
ORDER BY a.asset_id
{limit_clause}
"""
_MISSING_CLAUSE = "AND (m.ext_meta -> 'about') IS NULL"

_COUNT_SQL = """
SELECT count(*) AS total,
       count(*) FILTER (WHERE (m.ext_meta -> 'about') IS NOT NULL) AS done
FROM asset a
JOIN asset_metadata m ON m.asset_id = a.asset_id
WHERE a.status = 'registered' AND a.domain_label IS DISTINCT FROM 'medical'
"""


def backfill_about(
    targets: list[dict[str, Any]],
    *,
    extract_persist_fn: Callable[[Any, str | None], list[str]],
    os_update_fn: Callable[[Any, list[str]], None] | None = None,
) -> dict[str, int]:
    """대상 자산을 순회하며 추출·저장·OS 갱신한다(seam 주입 — 순수 단위 검증 가능).

    - ``extract_persist_fn(asset_id, summary) -> about``: 격리 트랜잭션 안의 추출+저장(IO 층이 배선).
    - ``os_update_fn(asset_id, about)``: OS 부분 갱신(None=생략). OS 실패는 삼켜 failed 로 세지
      않는다(DB 정본은 이미 커밋 — OS 는 다음 재색인이 회복).
    - 한 자산의 추출/저장 예외는 삼켜 ``failed`` 카운트 후 계속(배치 격리).
    """
    counts = {"done": 0, "empty": 0, "failed": 0, "os_failed": 0}
    for t in targets:
        asset_id, summary = t["asset_id"], t.get("summary")
        try:
            about = extract_persist_fn(asset_id, summary)
        except Exception as exc:  # noqa: BLE001 — 자산 격리(배치 계속)
            counts["failed"] += 1
            _LOG.warning("aboutness 백필 실패(계속): asset_id=%s (%s)", asset_id, exc)
            continue
        counts["done"] += 1
        if not about:
            counts["empty"] += 1
        if os_update_fn is not None:
            try:
                os_update_fn(asset_id, about)
            except Exception as exc:  # noqa: BLE001 — OS 실패는 정본(DB) 커밋과 무관
                counts["os_failed"] += 1
                _LOG.warning("OS about 갱신 실패(계속): asset_id=%s (%s)", asset_id, exc)
    return counts


def _bootstrap(env: str) -> Any:
    """.env.{env} 로드 → init_settings(운영 진입점 순서·run_topic_backfill 동형)."""
    from src.config.bootstrap import bootstrap_env

    return bootstrap_env(env)


def run(env: str, *, only_missing: bool, limit: int | None, os_sync: bool, report: bool) -> dict[str, int]:
    """IO 오케스트레이션 — 스캔→(리포트|백필). 백필은 자산별 짧은 트랜잭션."""
    from psycopg.rows import dict_row

    from src.classify.aboutness import extract_and_persist_about
    from src.database.postgres_util import PostgresUtil

    settings = _bootstrap(env)
    db = PostgresUtil()
    db.open_pool()
    try:
        def _scan(conn: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(_COUNT_SQL)
                status = dict(cur.fetchone())
                sql = _TARGET_SQL.format(
                    missing_clause=_MISSING_CLAUSE if only_missing else "",
                    limit_clause=f"LIMIT {int(limit)}" if limit else "",
                )
                cur.execute(sql)
                return status, list(cur.fetchall())

        status, targets = db.execute_in_transaction(_scan, idempotent=True)
        print(f"현황: 대상 {status['total']}건 중 about 확정 {status['done']}건 · 이번 대상 {len(targets)}건")
        if report:
            return {"targets": len(targets), **{k: int(v) for k, v in status.items()}}

        # OS 부분 갱신기(옵션) — 구 인덱스 매핑 보강 1회 후 자산별 update. 실패는 코어가 삼킴.
        os_update_fn = None
        if os_sync and settings.opensearch.sync_enabled:
            from src.search.opensearch_sync import (
                ensure_about_mapping,
                get_client,
                update_asset_about,
            )

            os_client = get_client()
            index = settings.opensearch.index
            ensure_about_mapping(os_client, index)

            def os_update_fn(asset_id: Any, about: list[str]) -> None:
                update_asset_about(os_client, index, asset_id, about)

        def extract_persist_fn(asset_id: Any, summary: str | None) -> list[str]:
            # 자산별 짧은 트랜잭션(격리) — 추출(LLM)+저장(jsonb 병합)을 원자로.
            def _one(conn: Any) -> list[str]:
                return extract_and_persist_about(conn, asset_id, summary=summary, client=None)

            return db.execute_in_transaction(_one)

        counts = backfill_about(
            targets, extract_persist_fn=extract_persist_fn, os_update_fn=os_update_fn
        )
        print(
            f"완료: done={counts['done']} (빈 about={counts['empty']}) "
            f"failed={counts['failed']} os_failed={counts['os_failed']}"
        )
        return counts
    finally:
        db.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description="aboutness 개체 백필 — 저장된 summary 로 ext_meta['about'] 소급 확정(073)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument("--all", action="store_true", help="대상 자산 전체(필수 플래그 — 실수 방지)")
    p.add_argument(
        "--only-missing", action=argparse.BooleanOptionalAction, default=True,
        help="about 미확정 자산만(기본 on; --no-only-missing 로 전체 재추출)",
    )
    p.add_argument("--limit", type=int, default=None, help="처리 자산 수 상한(배치·재개)")
    p.add_argument("--report", action="store_true", help="추출 없이 현황만(쓰기 0)")
    p.add_argument(
        "--os-sync", action=argparse.BooleanOptionalAction, default=True,
        help="OS about 필드 부분 갱신(기본 on; --no-os-sync 로 끔)",
    )
    args = p.parse_args()
    if not args.report and not args.all:
        p.error("--all 또는 --report 중 하나가 필요합니다")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(
        args.env, only_missing=args.only_missing, limit=args.limit,
        os_sync=args.os_sync, report=args.report,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
