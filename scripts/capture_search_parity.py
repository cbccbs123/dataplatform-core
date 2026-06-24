#!/usr/bin/env python3
"""라이브 검색 결과 스냅샷 — 045 bucket_policy 동작 동일 검증용."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_search(query: str, *, env: str, limit: int) -> dict:
    raw = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "src.app.run_search",
            "--env",
            env,
            "--query",
            query,
            "--limit",
            str(limit),
        ],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return json.loads(raw)


def _slim(d: dict, *, limit: int) -> dict:
    return {
        "query": d["query"],
        "meta": d.get("meta"),
        "results": {
            bucket: [
                {
                    "id": r.get("id"),
                    "similarity": r.get("similarity"),
                    "summary": (r.get("summary") or "")[:60],
                }
                for r in rows[:limit]
            ]
            for bucket, rows in (d.get("results") or {}).items()
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="검색 parity 스냅샷 캡처")
    p.add_argument("--env", default="dev")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("queries", nargs="+")
    args = p.parse_args()

    out: dict[str, dict] = {}
    for q in args.queries:
        out[q] = _slim(_run_search(q, env=args.env, limit=args.limit), limit=args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    for q, v in out.items():
        td = v["results"].get("text_documents", [])
        print(f"{q}: text_documents={len(td)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
