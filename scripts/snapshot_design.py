#!/usr/bin/env python3
"""설계서 스냅샷 — design-history/ 에 버전별로 누적(덮어쓰기 없음).

기능 개발마다 현재 `docs/설계서.md`(기본 소스)를 통째로 복사해
`design-history/vNNN_YYYY-MM-DD_<기능명>.md` 로 보존한다. 과거 버전은 절대
변경하지 않으므로 "복구"가 필요 없다 — 각 스냅샷이 그 시점의 동결본이다.

사용:
    python scripts/snapshot_design.py "하이브리드검색"            # docs/설계서.md 스냅샷
    python scripts/snapshot_design.py "의료팩" --source docs/설계서.md
    python scripts/snapshot_design.py "검색" --note "RRF 도입 반영"  # 인덱스에 비고 추가
    python scripts/snapshot_design.py --list                       # 기존 스냅샷 목록

표준 라이브러리만 사용(의존성 0). vNNN 은 design-history/ 의 기존 최대 번호 +1.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIST_DIR = ROOT / "design-history"
DEFAULT_SOURCE = ROOT / "docs" / "설계서.md"
INDEX = HIST_DIR / "README.md"

VERSION_RX = re.compile(r"^v(\d{3})_")
# 파일명에 못 쓰는 문자 → '_' (한글·영숫자·-_ 는 보존)
SAFE_RX = re.compile(r"[^0-9A-Za-z가-힣\-_]+")


def _next_version() -> int:
    """design-history/ 내 vNNN_*.md 의 최대 번호 + 1 (없으면 1)."""
    if not HIST_DIR.exists():
        return 1
    nums = [
        int(m.group(1))
        for p in HIST_DIR.glob("v*_*.md")
        if (m := VERSION_RX.match(p.name))
    ]
    return (max(nums) + 1) if nums else 1


def _slug(feature: str) -> str:
    s = SAFE_RX.sub("_", feature.strip()).strip("_")
    return s or "untitled"


def _list_snapshots() -> list[Path]:
    if not HIST_DIR.exists():
        return []
    return sorted(HIST_DIR.glob("v*_*.md"))


def _update_index(snapshot: Path, feature: str, note: str, source: Path) -> None:
    """README.md 표에 한 줄 추가(없으면 헤더 생성)."""
    ver = snapshot.name.split("_", 1)[0]
    date = _dt.date.today().isoformat()
    row = f"| {ver} | {date} | {feature} | `{source.relative_to(ROOT)}` | {note or '—'} | [`{snapshot.name}`](./{snapshot.name}) |\n"

    header = (
        "# 설계서 스냅샷 이력 (design-history)\n\n"
        "기능 개발 시점마다 `docs/설계서.md`(또는 지정 소스)를 통째로 동결한 **버전별 스냅샷**이다.\n"
        "과거 스냅샷은 **절대 변경하지 않는다** — 각 파일이 그 시점의 설계 동결본이며, 복구가 필요 없다.\n"
        "현행(살아있는) 설계서는 `docs/설계서.md`, 진척·로드맵은 `ROADMAP.md`가 권위다.\n\n"
        "생성: `python scripts/snapshot_design.py \"<기능명>\"` 또는 `/report-writer <기능명>`.\n\n"
        "| 버전 | 날짜 | 기능 | 소스 | 비고 | 스냅샷 |\n"
        "|---|---|---|---|---|---|\n"
    )
    if INDEX.exists():
        content = INDEX.read_text(encoding="utf-8")
        if "| 버전 | 날짜 |" not in content:
            content = header
    else:
        content = header
    INDEX.write_text(content + row, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="설계서 버전 스냅샷 생성")
    ap.add_argument("feature", nargs="?", help="이번 스냅샷의 기능명(파일명·인덱스에 사용)")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="스냅샷할 원본(기본: docs/설계서.md)")
    ap.add_argument("--note", default="", help="인덱스에 남길 한 줄 비고")
    ap.add_argument("--list", action="store_true", help="기존 스냅샷 목록만 출력")
    args = ap.parse_args(argv)

    if args.list:
        snaps = _list_snapshots()
        if not snaps:
            print("스냅샷 없음 (design-history/ 비어 있음)")
        else:
            for p in snaps:
                print(p.name)
        return 0

    if not args.feature:
        ap.error("기능명을 지정하세요. 예: python scripts/snapshot_design.py \"하이브리드검색\"")

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        print(f"[오류] 원본이 없습니다: {source}", file=sys.stderr)
        return 1

    HIST_DIR.mkdir(exist_ok=True)
    ver = _next_version()
    date = _dt.date.today().isoformat()
    fname = f"v{ver:03d}_{date}_{_slug(args.feature)}.md"
    dest = HIST_DIR / fname

    if dest.exists():  # 같은 날 같은 기능 재실행 — 덮어쓰지 않고 알림
        print(f"[중단] 이미 존재: {dest.name} (덮어쓰지 않음). 비고만 바꾸려면 파일명을 다르게.", file=sys.stderr)
        return 1

    # 원본 위에 스냅샷 메타 헤더를 얹어 복사(원본은 불변)
    banner = (
        f"<!-- 설계서 스냅샷 · {ver:03d} · {date} · 기능: {args.feature} -->\n"
        f"<!-- 원본: {source.relative_to(ROOT)} · 이 파일은 동결본이며 수정하지 않는다. -->\n\n"
    )
    dest.write_text(banner + source.read_text(encoding="utf-8"), encoding="utf-8")
    _update_index(dest, args.feature, args.note, source)

    print(f"✅ 스냅샷 생성: design-history/{fname}")
    print(f"   원본: {source.relative_to(ROOT)} → 동결본(불변)")
    print(f"   인덱스 갱신: design-history/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
