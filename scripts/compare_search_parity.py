#!/usr/bin/env python3
"""리팩터 전·후 라이브 검색 parity 비교."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _slim(d: dict, *, limit: int) -> dict:
    return {
        "meta": d.get("meta"),
        "results": {
            bucket: [{"id": r["id"], "similarity": r["similarity"]} for r in rows[:limit]]
            for bucket, rows in (d.get("results") or {}).items()
        },
    }


def _run_search(query: str, *, env: str, limit: int) -> dict:
    raw = subprocess.check_output(
        [sys.executable, "-m", "src.app.run_search", "--env", env, "--query", query, "--limit", str(limit)],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return json.loads(raw)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--env", default="dev")
    p.add_argument("--limit", type=int, default=5)
    args = p.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    failed = 0
    for query, snap in baseline.items():
        pre = _slim(snap, limit=args.limit)
        post = _slim(_run_search(query, env=args.env, limit=args.limit), limit=args.limit)
        meta_ok = pre["meta"] == post["meta"]
        res_ok = pre["results"] == post["results"]
        status = "OK" if meta_ok and res_ok else "FAIL"
        print(f"{query}: {status} meta={meta_ok} results={res_ok}")
        if not meta_ok:
            failed += 1
            print("  meta mismatch")
        if not res_ok:
            failed += 1
            for bucket in sorted(set(pre["results"]) | set(post["results"])):
                if pre["results"].get(bucket) != post["results"].get(bucket):
                    print(f"  {bucket}: pre={pre['results'].get(bucket)} post={post['results'].get(bucket)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
