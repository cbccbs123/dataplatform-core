#!/usr/bin/env python3
"""정책 게이트 — constitution 하드 불변식의 기계 검증(표준 라이브러리만).

CI(`.github/workflows/ci.yml`)와 로컬(`python scripts/policy_gate.py`)이 공유한다.
🔴 차단(block) 발견 시 exit 1, 🟡 경고(warn)는 출력만. src/ 만 스캔하며 주석은 제외(휴리스틱).
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
SEAM = "src/llm/client.py"  # LLM 단일 seam(자기 자신은 검사 제외)

TEMP_RX = re.compile(r"temperature\s*=\s*([0-9]+(?:\.[0-9]+)?)")

# (라벨, 정규식, 심각도)
CHECKS = [
    ("학습 배제(1조)", re.compile(r"\.fit\(|loss\.backward|optimizer\.step|requires_grad\s*=\s*True|fine[_-]?tune"), "block"),
    ("LLM 직접 import(2조)", re.compile(r"^\s*(?:from|import)\s+openai\b"), "warn"),
    ("직접 HTTP 호출(2조)", re.compile(r"\b(?:requests|httpx|aiohttp)\.(?:post|get|request)\("), "warn"),
]


def iter_py(base: str):
    for dp, dns, fns in os.walk(base):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            if fn.endswith(".py"):
                yield os.path.join(dp, fn)


def main() -> int:
    if not os.path.isdir(SRC):
        print("src/ 없음 — 건너뜀")
        return 0
    found: dict[str, list[str]] = {"block": [], "warn": []}
    for path in iter_py(SRC):
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        with open(path, encoding="utf-8") as f:
            for i, raw in enumerate(f, 1):
                code = raw.split("#", 1)[0]  # 주석 제외(휴리스틱)
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
