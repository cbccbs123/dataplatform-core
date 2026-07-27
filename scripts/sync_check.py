#!/usr/bin/env python3
"""문서 동기화 가드 — 마이그레이션·설계 변경이 문서에 반영됐는지 검사.

CI(`.github/workflows/ci.yml`)와 로컬에서 공유한다. 표준 라이브러리만 사용(의존성 0).
"docs-in-sync" 패턴: 코드(마이그레이션)는 바뀌었는데 문서를 안 고치면 실패시켜
설계서·DB이력이 코드와 어긋나는 것을 막는다.

검사 항목
  1) [차단] migrations/alembic/versions/vNNN_*.py 의 모든 리비전이
     docs/DB_변경이력.md 에 언급돼 있는가. (누락 = 미기록 마이그레이션)
  2) [경고] (--changed 모드) 이번 변경분에 마이그레이션이 있는데
     docs/DB_변경이력.md·docs/설계서.md 가 함께 바뀌지 않았는가.

사용
    python scripts/sync_check.py                 # 전수 검사(1) — CI 기본
    python scripts/sync_check.py --changed A B    # A..B diff 기반(2) 경고 동반
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSIONS_DIR = ROOT / "migrations" / "alembic" / "versions"
DB_HISTORY = ROOT / "docs" / "DB_변경이력.md"
DESIGN = ROOT / "docs" / "설계서.md"

REV_RX = re.compile(r"^(v\d{3})_.*\.py$")


def _all_revisions() -> list[str]:
    """마이그레이션 디렉터리에 실재하는 리비전 번호를 모은다.

    Returns:
        정렬된 리비전 목록(중복 제거). 디렉터리가 없으면 빈 목록.
    """
    if not VERSIONS_DIR.exists():
        return []
    revs = [
        m.group(1)
        for p in VERSIONS_DIR.glob("v*.py")
        if (m := REV_RX.match(p.name))
    ]
    return sorted(set(revs))


def _documented_revisions() -> set[str]:
    """DB 변경이력 문서에 적힌 리비전 번호를 모은다.

    Returns:
        문서에서 발견한 리비전 집합. 문서가 없으면 빈 집합.
    """
    if not DB_HISTORY.exists():
        return set()
    return set(re.findall(r"v\d{3}", DB_HISTORY.read_text(encoding="utf-8")))


def check_all() -> int:
    """실재하는 마이그레이션이 모두 문서에 기록됐는지 확인한다.

    스키마를 바꿔 놓고 이력을 안 남기면, 나중에 왜 이 컬럼이 생겼는지 알 길이 없어진다.

    Returns:
        0=이상 없음, 0이 아니면 차단.
    """
    revs = _all_revisions()
    if not revs:
        print("마이그레이션 리비전 없음 — 건너뜀")
        return 0
    documented = _documented_revisions()
    missing = [r for r in revs if r not in documented]

    print(f"리비전 {len(revs)}개, DB이력 기록 {len(documented)}개")
    if missing:
        print(f"\n🔴 DB_변경이력.md 에 누락된 리비전 {len(missing)}개:")
        for r in missing:
            f = next((p.name for p in VERSIONS_DIR.glob(f"{r}_*.py")), r)
            print(f"  - {r}  ({f})")
        print("\n→ docs/DB_변경이력.md 표에 추가하세요(`/db-history` 또는 수동).")
        return 1
    print("✅ 모든 마이그레이션 리비전이 DB이력에 기록됨.")
    return 0


def _git_changed(base: str, head: str) -> list[str]:
    """두 시점 사이에 바뀐 파일 목록을 얻는다.

    Returns:
        변경된 경로 목록. 비교에 실패하면 빈 목록(검사를 막지 않는다).
    """
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", f"{base}...{head}"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def check_changed(base: str, head: str) -> int:
    """이번 변경분에 마이그레이션이 있으면 문서 동반 변경을 경고(비차단)."""
    changed = _git_changed(base, head)
    if not changed:
        print("변경 파일 없음 — 건너뜀")
        return 0
    mig_changed = [f for f in changed if f.startswith("migrations/")]
    if not mig_changed:
        print("마이그레이션 변경 없음 — 문서 동기화 검사 불필요.")
        return 0
    db_touched = any(f == "docs/DB_변경이력.md" for f in changed)
    design_touched = any(f == "docs/설계서.md" for f in changed)
    print(f"마이그레이션 변경 {len(mig_changed)}건 감지.")
    if not db_touched:
        print("🟡 경고: 마이그레이션이 바뀌었으나 docs/DB_변경이력.md 가 함께 변경되지 않았습니다.")
    if not design_touched:
        print("🟡 경고: 스키마 변경 시 docs/설계서.md(§5 DB·ERD) 갱신 여부를 확인하세요.")
    if db_touched:
        print("✅ DB이력 동반 변경 확인.")
    return 0  # 경고만(비차단). 차단은 check_all 이 담당.


def main(argv: list[str] | None = None) -> int:
    """마이그레이션과 DB 변경이력이 어긋나지 않는지 확인한다(CI 게이트).

    Args:
        argv: 명령행 인자. ``None`` 이면 실제 인자를 읽는다(테스트 주입용).

    Returns:
        0=이상 없음, 0이 아니면 차단.
    """
    ap = argparse.ArgumentParser(description="문서 동기화 가드")
    ap.add_argument("--changed", nargs=2, metavar=("BASE", "HEAD"),
                    help="BASE..HEAD diff 기반 경고(비차단)")
    args = ap.parse_args(argv)

    rc = check_all()
    if args.changed:
        check_changed(args.changed[0], args.changed[1])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
