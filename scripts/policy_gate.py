#!/usr/bin/env python3
"""정책 게이트 — constitution 하드 불변식의 기계 검증(표준 라이브러리만).

CI(`.github/workflows/ci.yml`)와 로컬(`python scripts/policy_gate.py`)이 공유한다.
🔴 차단(block) 발견 시 exit 1, 🟡 경고(warn)는 출력만. 주석·문자열 리터럴은 제외(휴리스틱).

**스캔 범위는 `src/` + `scripts/`** 다. `scripts/` 를 넣은 이유: 측정·판정 도구가 `src/llm/client.py`
seam 을 경유해 **실제로 LLM 을 호출**하는데(`judge_relations`·`judge_snapshot`), `src/` 만 훑던
동안 이 파일들은 헌법 게이트 밖에 있었다 — 2026-07-30 정책 감사가 지적한 구멍이다.
편입 시점에 `scripts/` 선존 위반은 0건이었다(측정 후 편입).
"""
from __future__ import annotations

import io
import os
import re
import sys
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
# 스캔 루트 목록. `scripts/` 도 LLM seam 을 호출하므로 함께 훑는다(모듈 docstring 참조).
# `SRC` 는 하위호환으로 남긴다 — 외부에서 참조하는 코드가 있을 수 있다.
SCAN_ROOTS = (SRC, os.path.join(ROOT, "scripts"))
SEAM = "src/llm/client.py"  # LLM 단일 seam(자기 자신은 검사 제외)

TEMP_RX = re.compile(r"temperature\s*=\s*([0-9]+(?:\.[0-9]+)?)")

# (라벨, 정규식, 심각도)
CHECKS = [
    ("학습 배제(1조)", re.compile(r"\.fit\(|loss\.backward|optimizer\.step|requires_grad\s*=\s*True|fine[_-]?tune"), "block"),
    ("LLM 직접 import(2조)", re.compile(r"^\s*(?:from|import)\s+openai\b"), "warn"),
    ("직접 HTTP 호출(2조)", re.compile(r"\b(?:requests|httpx|aiohttp)\.(?:post|get|request)\("), "warn"),
]


# A2: 문자열/주석/f-string 리터럴 토큰 — 학습·정책 정규식의 오탐 원천이라 검사에서 제외한다.
_MASK_TOKEN_TYPES = {tokenize.STRING, tokenize.COMMENT}
for _n in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
    _t = getattr(tokenize, _n, None)
    if _t is not None:
        _MASK_TOKEN_TYPES.add(_t)


def _code_only_lines(source: str) -> list[str]:
    """소스에서 문자열/주석/f-string 리터럴 토큰을 공백으로 가린 라인 목록(1-based: ``[0]=''``).

    학습 배제·결정성 정규식이 **코드 토큰만** 보도록 마스킹한다 — 독스트링의 'fine-tuning 배제' 서술이나
    문자열 속 ``ImageOps.fit(`` 언급 같은 오탐을 없애되, 실제 코드(``loss.backward``·실제 ``.fit(`` 호출·
    f-string 내 표현식)는 그대로 남겨 검출된다. 토크나이즈 실패(부분·비정형 파일) 시엔 보수적으로 원본을
    쓴다(미검출보다 오탐을 감수 — 게이트는 놓치지 않는 편이 안전)."""
    lines = source.splitlines()
    grid = [list(ln) for ln in lines]
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type not in _MASK_TOKEN_TYPES:
                continue
            (sr, sc), (er, ec) = tok.start, tok.end
            for r in range(sr, er + 1):
                if r - 1 >= len(grid):
                    break
                row = grid[r - 1]
                c0 = sc if r == sr else 0
                c1 = ec if r == er else len(row)
                for c in range(c0, min(c1, len(row))):
                    row[c] = " "
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return ["", *lines]  # 토크나이즈 실패 시 원본(보수적)
    return ["", *("".join(r) for r in grid)]


def iter_py(*bases: str):
    """검사 대상 파이썬 파일 **경로**를 훑는다(`__pycache__` 제외 · 없는 루트는 건너뜀).

    Args:
        *bases: 스캔할 디렉터리 경로들. 존재하지 않는 경로는 조용히 건너뛴다
            (레포 구성에 따라 `scripts/` 가 없을 수 있다).

    Yields:
        파이썬 파일의 절대 경로. 정렬해 내보내 출력 순서를 결정적으로 만든다.
    """
    for base in bases:
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in sorted(os.walk(base)):
            dns[:] = sorted(d for d in dns if d != "__pycache__")
            for fn in sorted(fns):
                if fn.endswith(".py"):
                    yield os.path.join(dp, fn)


def main() -> int:
    """프로젝트 하드 규칙 위반을 정적으로 훑는다(외부 LLM 호출·학습 코드·결정성 위반 등).

    Returns:
        0=차단 없음. 위반이 있으면 0이 아닌 값으로 CI 를 실패시킨다.
        경고(🟡)는 종료 코드에 영향을 주지 않는다 — 판단이 필요한 항목이라 사람이 본다.
    """
    roots = [r for r in SCAN_ROOTS if os.path.isdir(r)]
    if not roots:
        print("스캔할 디렉터리 없음(src/·scripts/ 부재) — 건너뜀")
        return 0
    print("스캔 대상: " + ", ".join(os.path.relpath(r, ROOT) + "/" for r in roots))
    found: dict[str, list[str]] = {"block": [], "warn": []}
    for path in iter_py(*roots):
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        with open(path, encoding="utf-8") as f:
            source = f.read()
        raw_lines = source.splitlines()
        code_lines = _code_only_lines(source)  # 1-based; 문자열/주석/f-string 리터럴을 가린 코드만 스캔
        for i, raw in enumerate(raw_lines, 1):
            code = code_lines[i] if i < len(code_lines) else ""
            if not code.strip():
                continue
            # 결정성: temperature 비-0
            m = TEMP_RX.search(code)
            if m and float(m.group(1)) != 0:
                found["block"].append(f"  {rel}:{i}: [결정성(3조) temperature={m.group(1)}] {raw.strip()[:90]}")
            for label, rx, sev in CHECKS:
                if label.startswith("LLM 직접") or label.startswith("직접 HTTP"):
                    if rel == SEAM:
                        continue
                if rx.search(code):
                    found[sev].append(f"  {rel}:{i}: [{label}] {raw.strip()[:90]}")

    for sev, title in [("block", "🔴 차단"), ("warn", "🟡 경고(검토 권장)")]:
        items = found[sev]
        if items:
            print(f"{title} ({len(items)}건):")
            print("\n".join(items))
        else:
            print(f"{title}: 없음")
        print()

    if found["block"]:
        print("정책 게이트 실패 — 차단 항목을 수정하세요.")
        return 1
    print("정책 게이트 통과(차단 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
