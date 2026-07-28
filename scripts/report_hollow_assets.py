"""껍데기 자산(hollow) 감시 리포트 — 적재됐는데 검색에 안 걸리는 자산을 관측한다.

목적
    자산이 `registered` 인데 **임베딩이 하나도 없으면** 검색·관계·추천 어디에도 나타나지 않는다.
    적재는 성공했으니 오류 로그도 남지 않는다 — **조용히 사라지는 실패**다. 이 리포트가 그것을
    드러낸다.

왜 필요한가 (spec 064 의 대체물)
    064 는 "cv2 번들 ffmpeg 가 AV1 을 못 풀어 키프레임 0 → 임베딩 0" 을 시스템 ffmpeg 폴백으로
    고치려던 spec 이다. 그런데 2026-07-29 실측에서 **AV1 115/242 건이 정상 처리**되고 hollow 가
    0건이었다 — cv2 4.13.0 의 번들 ffmpeg(avcodec 61.19.101)가 그사이 AV1 디코딩을 지원하게 됐다.
    즉 지금은 고칠 대상이 없고, 그 상태로 폴백을 구현하면 **한 번도 실행되지 않는 죽은 코드**가
    된다. 그래서 064 를 보류로 내리고 **재개 조건을 자동 감지하는 이 지표**로 대체한다.

    ⚠️ cv2 의 코덱 지원은 **번들 ffmpeg 버전에 의존**한다. 패키지 다운그레이드·다른 환경·향후
    유입될 새 코덱에서 다시 깨질 수 있다. hollow 가 1건이라도 나오면 064 를 재개할 신호다.

읽기전용·결정성 (헌법 3조)
    `SELECT` 만 한다. LLM 0·재실행 동일. 어떤 테이블도 변경하지 않는다.

실행
    conda activate AuroraFS
    python scripts/report_hollow_assets.py --env dev
    python scripts/report_hollow_assets.py --env dev --json out.json
    python scripts/report_hollow_assets.py --env dev --fail-on-hollow   # CI/게이트용
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

# 임베딩이 필요 없는 모달리티는 없다 — 넷 다 검색 대상이다. 다만 modality='unknown' 은
# 빈 STT 로 격리된 자산이라(결정: 그대로 둔다) hollow 판정에서 제외한다.
_EXEMPT_MODALITIES = frozenset({"unknown"})


# ────────────────────────────────────────────────────────────────────────────
# 1) 순수 집계 (실 DB 없이 단위테스트로 덮는다)
# ────────────────────────────────────────────────────────────────────────────
def hollow_rows(asset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """임베딩이 0개인 자산만 골라낸다(순수 함수).

    Args:
        asset_rows: ``modality``·``embedding_count`` 키를 가진 자산 행 목록.
            ``embedding_count`` 가 ``None`` 이면 0으로 본다(LEFT JOIN 결과).

    Returns:
        hollow 자산 행 목록. 면제 모달리티(``unknown``)는 빠진다.
    """
    return [r for r in asset_rows
            if str(r.get("modality") or "") not in _EXEMPT_MODALITIES
            and int(r.get("embedding_count") or 0) == 0]


def hollow_by_modality(asset_rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """모달리티별 ``(hollow 건수, 전체 건수)`` 를 센다(순수 함수).

    **분모를 함께 내는 이유**: hollow 3건은 영상 5건 중이면 심각하고 영상 500건 중이면
    개별 파일 문제다. 건수만으로는 판단할 수 없다.

    Args:
        asset_rows: ``modality``·``embedding_count`` 키를 가진 자산 행 목록.

    Returns:
        ``{모달리티: (hollow, 전체)}``. 면제 모달리티는 키에 없다.
    """
    total: Counter = Counter()
    holl: Counter = Counter()
    for r in asset_rows:
        m = str(r.get("modality") or "")
        if m in _EXEMPT_MODALITIES:
            continue
        total[m] += 1
        if int(r.get("embedding_count") or 0) == 0:
            holl[m] += 1
    return {m: (holl[m], total[m]) for m in sorted(total)}


def build_report(asset_rows: list[dict[str, Any]], *, sample_limit: int = 10) -> dict[str, Any]:
    """리포트 dict 를 조립한다(순수 함수).

    Args:
        asset_rows: 자산 행 목록(``asset_id``·``fs_path``·``modality``·``embedding_count``).
        sample_limit: 사례로 표시할 hollow 자산 수 상한. 전량을 찍으면 로그가 넘친다.

    Returns:
        ``{healthy, hollow_total, asset_total, by_modality, samples, exempt_modalities}``.
        ``healthy`` 는 hollow 가 0건일 때만 참이다 — **1건이라도 있으면 조사 대상**이라서
        비율 임계를 두지 않는다(064 재개 신호).
    """
    holl = hollow_rows(asset_rows)
    by_mod = hollow_by_modality(asset_rows)
    considered = sum(t for _, t in by_mod.values())
    return {
        "healthy": len(holl) == 0,
        "hollow_total": len(holl),
        "asset_total": considered,
        "by_modality": {m: {"hollow": h, "total": t} for m, (h, t) in by_mod.items()},
        "samples": [
            {"asset_id": str(r.get("asset_id") or ""),
             "modality": str(r.get("modality") or ""),
             # 파일명만 남긴다 — 전체 경로는 로그에 남길 필요가 없다.
             "name": str(r.get("fs_path") or "").rsplit("/", 1)[-1][:80]}
            for r in holl[:sample_limit]
        ],
        "exempt_modalities": sorted(_EXEMPT_MODALITIES),
    }


def format_report_lines(report: dict[str, Any]) -> list[str]:
    """리포트를 사람이 읽는 줄 목록으로 만든다(순수 함수).

    Args:
        report: ``build_report`` 결과.

    Returns:
        출력할 줄 목록.
    """
    out = ["", "【껍데기 자산(hollow) 감시 — 적재됐으나 임베딩 0개】"]
    tot, n = report["asset_total"], report["hollow_total"]
    out.append(f"  대상 자산 {tot}건 (면제 모달리티 제외: {', '.join(report['exempt_modalities'])})")
    out.append(f"  hollow {n}건" + (f" ({100 * n / tot:.1f}%)" if tot else ""))
    out.append("")
    out.append(f"  {'모달리티':10} {'hollow':>7} {'전체':>6} {'비율':>7}")
    for m, v in report["by_modality"].items():
        h, t = v["hollow"], v["total"]
        out.append(f"  {m:10} {h:7} {t:6} {100 * h / t if t else 0:6.1f}%")
    if report["samples"]:
        out.append("")
        out.append("  사례:")
        for s in report["samples"]:
            out.append(f"    · [{s['modality']}] {s['name']}")
    out.append("")
    if report["healthy"]:
        out.append("  ✅ hollow 0건 — 모든 자산이 검색 가능하다.")
    else:
        out.append("  🔴 hollow 발견 — 적재는 성공했으나 검색·관계에 나타나지 않는 자산이다.")
        out.append("     원인 후보: ① 코덱 미지원(키프레임 0) ② 빈 전사/본문 ③ 임베딩 단계 실패.")
        out.append("     영상이면 **spec 064(ffmpeg 트랜스코딩 폴백) 재개 신호**다 — 보류 해제를 검토하라.")
    return out


# ────────────────────────────────────────────────────────────────────────────
# 2) DB 조회 (읽기 전용)
# ────────────────────────────────────────────────────────────────────────────
_ASSET_SQL = """
    SELECT a.asset_id::text AS asset_id, a.fs_path, a.modality,
           coalesce(e.n, 0) AS embedding_count
    FROM asset a
    LEFT JOIN (SELECT asset_id, count(*) AS n FROM asset_embedding GROUP BY 1) e
           ON e.asset_id = a.asset_id
    WHERE a.status = 'registered'
    ORDER BY a.asset_id
"""


def fetch_asset_rows(conn: Any) -> list[dict[str, Any]]:
    """등록 자산과 임베딩 개수를 읽는다(읽기 전용).

    Args:
        conn: DB 커넥션.

    Returns:
        ``asset_id``·``fs_path``·``modality``·``embedding_count`` 를 가진 행 목록.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_ASSET_SQL)
        # ⚠️ dict_row 커서다 — dict(cur.fetchall()) 로 감싸면 키/값이 뒤집힌다.
        return list(cur.fetchall())


def run_report(*, env: str, sample_limit: int = 10) -> dict[str, Any]:
    """설정을 초기화하고 DB 를 읽어 리포트를 만든다.

    Args:
        env: 설정 프로파일(``dev``·``prod``). ``.env.<env>`` 를 읽어 초기화한다.
        sample_limit: 사례 표시 상한.

    Returns:
        ``build_report`` 결과.
    """
    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    dotenv_path = _REPO_ROOT / f".env.{env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(env)

    db = PostgresUtil()
    with db, db.transaction() as conn:
        rows = fetch_asset_rows(conn)
    return build_report(rows, sample_limit=sample_limit)


def main() -> int:
    """임베딩이 0개인 자산을 집계해 보고한다 — 조용히 사라지는 적재 실패를 드러낸다.

    Returns:
        0=정상(또는 hollow 가 있으나 ``--fail-on-hollow`` 미지정),
        1=hollow 가 있고 ``--fail-on-hollow`` 지정(CI 차단용).
    """
    p = argparse.ArgumentParser(
        description="껍데기 자산 감시 리포트 — 적재됐으나 임베딩 0개인 자산(읽기전용)")
    p.add_argument("--env", choices=["dev", "prod"], default="dev",
                   help="설정 프로파일(기본: dev). .env.<env> 를 읽어 초기화한다")
    p.add_argument("--sample-limit", dest="sample_limit", type=int, default=10,
                   help="사례로 표시할 hollow 자산 수 상한(기본 10)")
    p.add_argument("--json", dest="json_out", default=None, help="리포트 JSON 저장 경로(선택)")
    p.add_argument("--fail-on-hollow", dest="fail_on_hollow", action="store_true",
                   help="hollow 가 1건이라도 있으면 비영 종료(CI 게이트용)")
    args = p.parse_args()

    report = run_report(env=args.env, sample_limit=args.sample_limit)
    print("\n".join(format_report_lines(report)))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  리포트 JSON 저장: {args.json_out}")
    return 1 if (args.fail_on_hollow and not report["healthy"]) else 0


if __name__ == "__main__":
    sys.exit(main())
