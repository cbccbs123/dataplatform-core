"""자산 자기주제 백필 배치 CLI — 저장된 summary/keywords 로 ``asset_topic`` 정본을 소급 부여(spec 065·G4·T401).

무엇을 하나 (FR-501·FR-502·FR-503)
    수집(run_ingest)은 이제 자산마다 자기주제를 즉시 부여하지만(FR-301), 065 배선 **이전에** 적재된
    기존 자산은 ``asset_topic`` 행이 없다. 이 배치는 재수집·재추출 없이 이미 적재된 자기 텍스트
    (``asset_metadata.ext_meta`` 의 summary/keywords/labels)로 ``classify_asset_topic`` 을 일괄 호출해
    정본을 채운다. 백필로 주제가 생긴 자산만 OS 를 재색인해 색인된 ``topics`` 필드를 갱신한다.

    - **대상**: ``status='registered'`` + ``asset_metadata`` 존재. ``--only-missing``(기본)이면
      ``asset_topic`` 행이 없는 자산만(LEFT JOIN ... IS NULL) — 재실행 멱등(이미 부여된 자산 스킵).
    - **자산별 짧은 트랜잭션 + 격리(FR-204)**: 자산마다 독립 트랜잭션으로 분류·upsert 하고,
      한 자산의 예외는 삼켜(failed 카운트) 배치를 멈추지 않는다.
    - **결정적 순서**(헌법 3조): 대상 스캔은 ``asset_id`` 오름차순.

실행 (FR-503 — 백필 실행은 사람 게이트: 재수집 완료·데이터 확정 후 사용자가 지시)
    conda activate AuroraFS
    python -m src.app.run_topic_backfill --env dev --report            # 현황만(분류 0·쓰기 0)
    python -m src.app.run_topic_backfill --env dev --all               # 미부여 자산 전체 백필
    python -m src.app.run_topic_backfill --env dev --all --limit 100   # 앞 100건만(배치·재개)
    python -m src.app.run_topic_backfill --env dev <asset_uuid> ...    # 지정 자산만
    python -m src.app.run_topic_backfill --env dev --all --no-os-sync  # OS 재색인 생략
    python -m src.app.run_topic_backfill --env dev --all --reclassify  # 품질 재백필(기존 행도 재분류·T604)

설계 주의
    분류·OS 재색인·존재확인은 seam(``classify_fn``/``os_sync_fn``/``has_topic_fn``)으로 주입 가능해
    ``backfill_assets`` 를 실 DB/LLM 없이 순수 단위로 검증한다(테스트가 분기만 확인). DB 접속부
    (``run_backfill``·``_make_os_syncer``)는 얇게 두고 IO 검증은 사람 게이트(RUN_DB_E2E·백필 실행) 몫.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("meta_extract.run_topic_backfill")

# 재실행 멱등 스킵 센티넬 — 트랜잭션 콜백이 "이미 부여됨(재분류 안 함)"을 카운터에 전달하는 표식.
# None(미부여)·dict(부여)와 구분해야 하므로 별도 객체를 쓴다.
_SKIP = object()

# 재분류 미부여 전이 센티넬(T604) — reclassify 모드에서 재분류 결과가 None(미부여)이라 기존 행을 삭제
# 했음을 카운터에 전달한다. 트랜잭션 콜백은 (_UNASSIGNED, deleted_rowcount) 튜플로 돌려준다.
_UNASSIGNED = object()

# 재분류 미부여 전이 시 기존 자기주제 행 삭제(T604). 삭제된 행 수(rowcount)를 돌려줘 실제 전이만 카운트.
_DELETE_TOPIC_SQL = "DELETE FROM asset_topic WHERE asset_id = %s"


# ────────────────────────────────────────────────────────────────────────────
# 1) 대상 스캔 (읽기전용 — 얇은 DB)
# ────────────────────────────────────────────────────────────────────────────
# 전체 대상(--all + --no-only-missing): registered + 메타 보유. 결정적 순서(asset_id asc).
_TARGET_ALL_SQL = """
SELECT a.asset_id
FROM asset a
WHERE a.status = 'registered'
  AND EXISTS (SELECT 1 FROM asset_metadata m WHERE m.asset_id = a.asset_id)
ORDER BY a.asset_id
"""
# 미부여만(--only-missing·기본): 위에 asset_topic LEFT JOIN + IS NULL 을 더해 이미 부여된 자산 제외.
_TARGET_MISSING_SQL = """
SELECT a.asset_id
FROM asset a
LEFT JOIN asset_topic at ON at.asset_id = a.asset_id
WHERE a.status = 'registered'
  AND EXISTS (SELECT 1 FROM asset_metadata m WHERE m.asset_id = a.asset_id)
  AND at.asset_id IS NULL
ORDER BY a.asset_id
"""

_HAS_TOPIC_SQL = "SELECT 1 FROM asset_topic WHERE asset_id = %s LIMIT 1"

# --report 현황 카운트(읽기전용 1쿼리) — registered / 메타 보유 / 자기주제 부여 자산 수.
_STATUS_SQL = """
SELECT
  (SELECT count(*) FROM asset WHERE status = 'registered') AS n_registered,
  (SELECT count(*) FROM asset a
     WHERE a.status = 'registered'
       AND EXISTS (SELECT 1 FROM asset_metadata m WHERE m.asset_id = a.asset_id)
  ) AS n_with_meta,
  (SELECT count(*) FROM asset a
     JOIN asset_topic at ON at.asset_id = a.asset_id
     WHERE a.status = 'registered'
  ) AS n_with_topic
"""


def _fetch_target_asset_ids(conn, *, only_missing: bool = True, limit: int | None = None) -> list[str]:
    """백필 대상 asset_id 목록(읽기전용·결정적). ``only_missing`` 이면 미부여 자산만."""
    sql = _TARGET_MISSING_SQL if only_missing else _TARGET_ALL_SQL
    params: tuple = ()
    if limit is not None:
        sql = sql + "LIMIT %s\n"
        params = (int(limit),)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [str(r[0]) for r in cur.fetchall()]


def _asset_has_topic(conn, asset_id) -> bool:
    """자산에 자기주제 정본 행이 이미 있는지(재실행 멱등 스킵 판정·읽기전용)."""
    with conn.cursor() as cur:
        cur.execute(_HAS_TOPIC_SQL, (asset_id,))
        return cur.fetchone() is not None


def _delete_asset_topic(conn, asset_id) -> int:
    """재분류 미부여 전이(T604) — 기존 자기주제 행 삭제. 실제 삭제된 행 수(rowcount) 반환.

    ``--reclassify`` 에서 재분류 결과가 None(미부여)이면 기존 행을 지워 "주제 없음"으로 전이시킨다.
    원래 미부여였던 자산은 삭제할 행이 없어 rowcount 0(no-op) — deleted 카운트에 잡히지 않는다.
    """
    with conn.cursor() as cur:
        cur.execute(_DELETE_TOPIC_SQL, (asset_id,))
        return int(cur.rowcount or 0)


def _fetch_status_counts(conn) -> dict[str, int]:
    """--report 현황 카운트 조회(읽기전용)."""
    with conn.cursor() as cur:
        cur.execute(_STATUS_SQL)
        row = cur.fetchone()
    return {
        "n_registered": int(row[0]),
        "n_with_meta": int(row[1]),
        "n_with_topic": int(row[2]),
    }


# ────────────────────────────────────────────────────────────────────────────
# 2) 배치 루프 (순수 — seam 주입)
# ────────────────────────────────────────────────────────────────────────────
def backfill_assets(
    db,
    asset_ids: list[str],
    *,
    skip_existing: bool = True,
    reclassify: bool = False,
    classify_fn: Callable[..., dict | None] | None = None,
    os_sync_fn: Callable[[Any], bool] | None = None,
    has_topic_fn: Callable[[Any, Any], bool] | None = None,
    delete_fn: Callable[[Any, Any], int] | None = None,
    settings: Any = None,
    client: Any = None,
    log_every: int = 50,
) -> dict[str, int]:
    """자산 리스트를 순회하며 자기주제를 부여한다(자산별 짧은 트랜잭션·격리·멱등).

    Args:
        db: PostgresUtil(자산별 ``execute_in_transaction`` 으로 분류·upsert 를 짧게 커밋).
        asset_ids: 대상 자산 id 목록(호출부가 스캔·지정으로 확정).
        skip_existing: True 면 자산별로 ``has_topic_fn`` 을 확인해 이미 부여된 자산은 재분류 없이
            스킵한다(--only-missing 로 지정 자산을 재실행할 때의 멱등 — --all 스캔 경로는 이미
            스캔에서 걸러져 False 로 넘어온다). ``reclassify`` 와 상호배타(재분류 시 False).
        reclassify: True 면 품질 재백필 모드(T604·FR-704) — 기존 asset_topic 행도 재분류하고, 재분류
            결과가 None(미부여)이면 ``delete_fn`` 으로 기존 행을 삭제(미부여 전이)한다. dict 결과는
            classify_fn 이 upsert(정본 갱신).
        classify_fn: (conn, asset_id, *, settings, client) → dict|None. 미주입=운영 분류 seam.
        os_sync_fn: (asset_id) → bool(색인 성공 여부). None 이면 OS 재색인 생략(--no-os-sync).
        has_topic_fn: (conn, asset_id) → bool. 미주입=``_asset_has_topic``.
        delete_fn: (conn, asset_id) → int(삭제 rowcount). 미주입=``_delete_asset_topic``. reclassify
            미부여 전이에만 호출.

    Returns:
        요약 카운트 dict — ``{scanned, classified, skipped_existing, no_text, deleted, failed,
        os_synced}``. ``no_text`` = 분류가 None(미부여)을 반환한 자산(대부분 자기 텍스트 없음 — 정밀
        사유 구분은 ``scripts/topic_grounding_report.py`` 후분석 몫). ``deleted`` = reclassify 로 기존
        행이 실제 삭제된(미부여 전이) 자산 수. ``failed`` = 자산 처리 중 예외(격리).
    """
    if classify_fn is None:
        from src.classify.asset_topic import classify_asset_topic

        classify_fn = classify_asset_topic
    if has_topic_fn is None:
        has_topic_fn = _asset_has_topic
    if delete_fn is None:
        delete_fn = _delete_asset_topic

    summary = {
        "scanned": 0,
        "classified": 0,
        "skipped_existing": 0,
        "no_text": 0,
        "deleted": 0,
        "failed": 0,
        "os_synced": 0,
    }
    total = len(asset_ids)
    for i, aid in enumerate(asset_ids, 1):
        summary["scanned"] += 1
        try:
            # 자산별 짧은 트랜잭션: 존재 확인(멱등 스킵) + 분류·upsert 를 한 트랜잭션으로 커밋.
            # aid 는 기본인자로 바인딩(루프 늦은 바인딩 footgun 차단·ruff B023).
            def _txn(conn, _aid=aid):
                if skip_existing and has_topic_fn(conn, _aid):
                    return _SKIP  # 재실행 멱등 — 이미 부여됨
                r = classify_fn(conn, _aid, settings=settings, client=client)
                if r is None and reclassify:
                    # 재분류 결과 미부여 → 기존 행 삭제(미부여 전이). rowcount 로 실제 전이만 카운트.
                    return (_UNASSIGNED, delete_fn(conn, _aid))
                return r

            result = db.execute_in_transaction(_txn, idempotent=False)
            if result is _SKIP:
                summary["skipped_existing"] += 1
                continue
            if isinstance(result, tuple) and result and result[0] is _UNASSIGNED:
                summary["no_text"] += 1  # 미부여(재분류로 주제 회수)
                if result[1]:  # rowcount>0 = 있던 행을 실제 삭제(미부여 전이)
                    summary["deleted"] += 1
                continue
            if result is None:
                summary["no_text"] += 1  # 미부여(자기 텍스트 없음/후보 없음/닫힌집합 실패)
                continue
            summary["classified"] += 1
            # OS 재색인은 topic 이 생긴 자산만(색인된 topics 필드 갱신). --no-os-sync 면 os_sync_fn=None.
            if os_sync_fn is not None and os_sync_fn(aid):
                summary["os_synced"] += 1
        except Exception as exc:  # noqa: BLE001 — 자산 단위 격리(실패 카운트·배치 계속)
            summary["failed"] += 1
            _LOG.warning("자기주제 백필 실패(격리): asset_id=%s (%s)", aid, exc)
        if log_every and i % log_every == 0:
            _LOG.info(
                "자기주제 백필 진행 %d/%d — 분류 %d · 스킵 %d · 미부여 %d · 실패 %d",
                i, total, summary["classified"], summary["skipped_existing"],
                summary["no_text"], summary["failed"],
            )
    return summary


# ────────────────────────────────────────────────────────────────────────────
# 3) 현황 리포트(--report) — 순수 집계 + 포맷
# ────────────────────────────────────────────────────────────────────────────
def build_status_report(counts: dict[str, int]) -> dict[str, Any]:
    """현황 카운트 → 리포트(순수). 부여율(메타 보유 대비)·미부여(백필 후보) 수 파생."""
    n_reg = int(counts.get("n_registered", 0))
    n_meta = int(counts.get("n_with_meta", 0))
    n_topic = int(counts.get("n_with_topic", 0))
    n_missing = max(n_meta - n_topic, 0)  # 메타 보유 중 미부여 = 백필 후보
    rate = (n_topic / n_meta) if n_meta else 0.0
    return {
        "n_registered": n_reg,
        "n_with_meta": n_meta,
        "n_with_topic": n_topic,
        "n_missing": n_missing,
        "assignment_rate": round(rate, 4),
    }


def format_status_lines(report: dict[str, Any]) -> list[str]:
    """현황 리포트 dict → 콘솔 줄(순수·사람 검수용)."""
    pct = 100.0 * report["assignment_rate"]
    return [
        "[자기주제 백필 현황 · 065 · 읽기전용]",
        f"  registered 자산      : {report['n_registered']}",
        f"  메타 보유(분류 가능) : {report['n_with_meta']}",
        f"  자기주제 부여        : {report['n_with_topic']}"
        f"  (부여율 {pct:.1f}% · 메타 보유 대비)",
        f"  미부여(백필 후보)    : {report['n_missing']}",
    ]


# ────────────────────────────────────────────────────────────────────────────
# 4) 실행(IO) — 부트스트랩 · OS 재색인기 · 오케스트레이션
# ────────────────────────────────────────────────────────────────────────────
def _bootstrap(env: str) -> Any:
    """.env.{env} 로드 → init_settings(운영 진입점 순서). frozen settings 반환."""
    from src.config.bootstrap import bootstrap_env

    return bootstrap_env(env)


def _make_os_syncer(db, settings) -> Callable[[Any], bool] | None:
    """백필 OS 재색인기 — 대상 자산 1건을 ``index_asset`` 로 전체문서 재색인(topics 정본 포함).

    ``opensearch_sync_enabled`` off 면 None(검색 미도입 환경 — 재색인 생략). ``index_asset`` 은
    자산의 (메타 + 평균 임베딩) 1행을 읽어 ``fetch_asset_topic`` 자기주제 정본을 실어 upsert 색인한다
    (run_ingest 증분 훅과 동일 개별 sync 경로 재사용·FR-502). OS 실패는 삼켜(주제 부여는 이미 커밋)
    False 를 돌려준다. 지연 import — 플래그 off 환경에서 opensearch-py 미설치를 허용한다.
    """
    if not getattr(settings, "opensearch_sync_enabled", False):
        _LOG.info("opensearch_sync_enabled off — OS 재색인 생략(주제는 DB 정본에만 부여)")
        return None

    from src.config.settings import active_embed_channel
    from src.search.opensearch_sync import get_client, index_asset

    client = get_client(settings.opensearch_url)
    channel = active_embed_channel(settings)
    index = settings.opensearch_index
    noise = getattr(settings, "opensearch_filename_noise_patterns", ())

    def _sync(asset_id) -> bool:
        def _run(conn):
            return index_asset(
                client, conn, str(asset_id),
                index=index, channel=channel, noise_patterns=noise,
            )

        try:
            return db.execute_in_transaction(_run, idempotent=True) is not None
        except Exception as exc:  # noqa: BLE001 — OS 실패는 주제 부여를 되돌리지 않는다
            _LOG.warning("OS 재색인 실패(무시): asset_id=%s (%s)", asset_id, exc)
            return False

    return _sync


def run_report(env: str) -> dict[str, Any]:
    """--report: 백필 없이 현황만 집계(읽기전용 DB·분류 0·쓰기 0)."""
    from src.database.postgres_util import PostgresUtil

    _bootstrap(env)
    db = PostgresUtil()
    with db:
        counts = db.execute_in_transaction(_fetch_status_counts, idempotent=True)
    report = build_status_report(counts)
    report["env"] = env
    return report


def run_backfill(
    env: str,
    *,
    asset_ids: list[str] | None = None,
    only_missing: bool = True,
    limit: int | None = None,
    os_sync: bool = True,
    reclassify: bool = False,
) -> dict[str, int]:
    """백필 오케스트레이션 — 대상 확정 → OS 재색인기 준비 → ``backfill_assets`` 루프.

    지정 자산(``asset_ids``)이 있으면 그 자산만 처리하고(``only_missing`` 이면 이미 부여된 자산은
    루프에서 스킵), 없으면 ``--all`` 로 보고 대상을 스캔한다(``only_missing`` 필터·``limit`` 는
    스캔에서 적용되므로 루프의 존재확인은 생략).

    ``reclassify``(T604·FR-704): 품질 재백필 — 기존 asset_topic 행도 재분류한다. 미부여 필터를 풀고
    (only_missing=False) 재실행 스킵도 끈 채(skip_existing=False) 전 자산을 고정 레지스트리 기준으로
    다시 분류하고, 재분류 결과가 None 이면 기존 행을 삭제(미부여 전이)한다.
    """
    from src.database.postgres_util import PostgresUtil

    # 재분류 모드는 기존 행 포함이 목적 — 미부여 필터를 강제로 풀어 이미 부여된 자산도 대상에 넣는다.
    if reclassify:
        only_missing = False

    settings = _bootstrap(env)
    db = PostgresUtil()
    with db:
        if asset_ids:
            targets = list(asset_ids)
            if limit is not None:
                targets = targets[:limit]
            # 지정 자산 재실행 멱등은 루프에서 확인(재분류면 스킵 없이 모두 다시 분류).
            skip_existing = only_missing and not reclassify
        else:
            targets = db.execute_in_transaction(
                lambda conn: _fetch_target_asset_ids(
                    conn, only_missing=only_missing, limit=limit
                ),
                idempotent=True,
            )
            skip_existing = False  # 스캔이 이미 미부여만 골랐다(재확인 불요)
        os_sync_fn = _make_os_syncer(db, settings) if os_sync else None
        summary = backfill_assets(
            db, targets, skip_existing=skip_existing, reclassify=reclassify,
            os_sync_fn=os_sync_fn, settings=settings,
        )
    _LOG.info("자기주제 백필 완료: %s", summary)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(
        description="자산 자기주제 백필 배치 — 저장된 summary/keywords 로 asset_topic 정본 소급 부여(065)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument("--all", action="store_true", help="registered + 메타 보유 자산 전체 대상")
    p.add_argument(
        "--only-missing", action=argparse.BooleanOptionalAction, default=True,
        help="asset_topic 행이 없는 자산만(기본 on; --no-only-missing 로 전체 재분류)",
    )
    p.add_argument("--limit", type=int, default=None, help="처리 자산 수 상한(배치·재개)")
    p.add_argument("--report", action="store_true", help="분류 없이 현황 리포트만(쓰기 0)")
    p.add_argument(
        "--reclassify", action="store_true",
        help="품질 재백필(T604) — 기존 asset_topic 행도 재분류(미부여 필터 해제); None 결과는 행 삭제",
    )
    p.add_argument(
        "--os-sync", action=argparse.BooleanOptionalAction, default=True,
        help="백필로 주제가 생긴 자산 OS 재색인(기본 on; --no-os-sync 로 끔)",
    )
    p.add_argument("--json", dest="json_out", default=None, help="요약/리포트 JSON 저장 경로(선택)")
    p.add_argument("asset_ids", nargs="*", metavar="ASSET_ID")
    args = p.parse_args()

    if args.report:
        report = run_report(args.env)
        print("\n".join(format_status_lines(report)))
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  리포트 JSON 저장: {args.json_out}")
        return 0

    if not args.all and not args.asset_ids:
        print("대상 미지정 — --all 또는 asset_id 를 지정하라(현황만 보려면 --report).")
        return 2

    summary = run_backfill(
        args.env,
        asset_ids=list(args.asset_ids) or None,
        only_missing=args.only_missing,
        limit=args.limit,
        os_sync=args.os_sync,
        reclassify=args.reclassify,
    )
    print(json.dumps(summary, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
