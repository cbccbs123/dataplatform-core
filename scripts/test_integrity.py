#!/usr/bin/env python3
"""테스트 무결성 가드 — 자율 루프가 "테스트를 약화시켜 통과 위조"하는 것을 차단.

자율 개발 루프(feature-builder 서브에이전트)가 테스트를 green 으로 만들 때,
정답을 구현하는 대신 테스트를 지우거나(assert 삭제) skip 으로 무력화하는 함정에 빠질 수 있다.
본 스크립트는 기준 시점(base) 대비 tests/ 의 "검증 강도"가 줄지 않았는지 검사한다.

검사(BASE..현재 작업트리):
  1) [차단] 테스트 함수(def test_*) 총개수 감소
  2) [차단] assert 계열 호출(self.assert*, assert ) 총개수 감소
  3) [차단] @skip/@skipIf/@expectedFailure 총개수 증가(무조건 무력화 패턴)
  → 하나라도 걸리면 exit 1. "테스트를 약하게 만들어 통과"를 거부한다.

  주의: ``@skipUnless`` 는 **조건부 실행 게이트**(실 DB ``RUN_DB_E2E`` e2e·선택 의존성 등)로
  이 코드베이스의 표준 e2e 패턴이라 약화가 아니다 → 증가 검사에서 제외한다(신규 e2e 추가 시 오탐 방지).
  무조건 무력화(@skip/@skipIf/@expectedFailure)와 assert/테스트 함수 감소만 차단한다.

사용:
    python scripts/test_integrity.py            # main 대비
    python scripts/test_integrity.py --base HEAD~1
표준 라이브러리만 사용(의존성 0).
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = "tests/"

TESTDEF_RX = re.compile(r"^\s*def\s+test_\w+", re.M)
ASSERT_RX = re.compile(r"\bself\.assert\w+\(|^\s*assert\s", re.M)
# skipUnless 는 조건부 실행 게이트(e2e/선택 의존성)라 약화 아님 → 제외.
# 무조건 무력화 패턴(skip·skipIf·expectedFailure)만 증가 차단 대상.
SKIP_RX = re.compile(r"@(?:unittest\.)?(?:skip|skipIf|expectedFailure)\b")


def _counts(text: str) -> tuple[int, int, int]:
    return (
        len(TESTDEF_RX.findall(text)),
        len(ASSERT_RX.findall(text)),
        len(SKIP_RX.findall(text)),
    )


def _git_show(ref: str, path: str) -> str:
    try:
        return subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return ""  # base 에 없던 파일(신규) → 빈 문자열


def _list_test_files(ref: str | None) -> list[str]:
    if ref is None:  # 현재 작업트리
        return [
            str(p.relative_to(ROOT))
            for p in (ROOT / "tests").glob("test_*.py")
        ]
    out = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, TESTS],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln.endswith(".py") and "/test_" in f"/{ln}"]


def _aggregate(ref: str | None) -> tuple[int, int, int]:
    td = ta = ts = 0
    files = set(_list_test_files(ref)) | set(_list_test_files(None) if ref else [])
    for f in files:
        text = (ROOT / f).read_text(encoding="utf-8") if ref is None and (ROOT / f).exists() \
            else _git_show(ref, f) if ref else ""
        a, b, c = _counts(text)
        td += a
        ta += b
        ts += c
    return td, ta, ts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="테스트 무결성 가드")
    ap.add_argument("--base", default="main", help="기준 ref(기본 main)")
    args = ap.parse_args(argv)

    # base 존재 확인
    if subprocess.run(["git", "rev-parse", "--verify", args.base],
                      cwd=ROOT, capture_output=True).returncode != 0:
        print(f"기준 ref '{args.base}' 없음 — 검사 건너뜀(첫 커밋/CI 환경 가능).")
        return 0

    b_td, b_ta, b_ts = _aggregate(args.base)
    c_td, c_ta, c_ts = _aggregate(None)

    print(f"기준({args.base}) → 현재")
    print(f"  test 함수 : {b_td} → {c_td}")
    print(f"  assert    : {b_ta} → {c_ta}")
    print(f"  skip 데코  : {b_ts} → {c_ts}")

    fails = []
    if c_td < b_td:
        fails.append(f"테스트 함수가 {b_td}→{c_td} 로 감소(삭제 의심)")
    if c_ta < b_ta:
        fails.append(f"assert 가 {b_ta}→{c_ta} 로 감소(검증 약화 의심)")
    if c_ts > b_ts:
        fails.append(f"skip 데코가 {b_ts}→{c_ts} 로 증가(테스트 무력화 의심)")

    if fails:
        print("\n🔴 테스트 무결성 위반 — 통과를 위해 테스트를 약화시킨 정황:")
        for f in fails:
            print(f"  - {f}")
        print("\n→ 테스트를 지우지 말고 구현으로 통과시키세요. 의도적 테스트 정리면 사람이 검토 후 진행.")
        return 1
    print("\n✅ 테스트 무결성 유지(약화 없음).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
