#!/usr/bin/env python3
"""048 T401 — 영상 키프레임 dedup off/on 비교 측정(SC-006 프록시).

샘플 영상들에 대해 ``extract_video_representative_frame_bytes`` 를 dedup **off**(현행)와
**on**(settings 기본값)으로 각각 돌려, 키프레임 수 감소율을 측정한다. VLM 호출은 키프레임당 1회
이므로 키프레임 수 감소 = VLM 호출 감소(SC-006의 직접 프록시). skip 사유 분포도 집계한다.

※ SC-007(recall@10)은 재색인 + 골든 측정이 필요해 본 스크립트 범위 밖이다(별도·사람 실행):
   `python -m src.app.run_opensearch_resync --env dev --recreate` 후 `scripts/measure_search_golden.py`.

사용:
    conda activate AuroraFS
    python scripts/measure_keyframe_dedup.py --env dev --input-dir <영상디렉터리>
    python scripts/measure_keyframe_dedup.py --env dev <영상파일>...
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _collect_videos(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.input_dir:
        for p in sorted(Path(args.input_dir).rglob("*")):
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
                paths.append(p)
    paths.extend(Path(p) for p in args.videos)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="키프레임 dedup off/on 측정(SC-006)")
    ap.add_argument("--env", default="dev", choices=["dev", "prod"])
    ap.add_argument("--input-dir", default=None, help="영상 디렉터리(재귀)")
    ap.add_argument("videos", nargs="*", help="개별 영상 경로")
    args = ap.parse_args()

    from src.config.settings import init_settings

    cfg = init_settings(args.env)

    from src.preprocess.keyframe_dedup import KeyframeDedupConfig, dedup_keyframes
    from src.preprocess.video_keyframes import extract_video_representative_frame_bytes

    dedup_on = KeyframeDedupConfig(
        enabled=True,
        hash_max=cfg.video.dedup_hash_max,
        ssim_min=cfg.video.dedup_ssim_min,
        ssim_gray_lo=cfg.video.dedup_ssim_gray_lo,
        hist_min=cfg.video.dedup_hist_min,
        compare_mode=cfg.video.dedup_compare_mode,
        recent_window=cfg.video.dedup_recent_window,
    )

    videos = _collect_videos(args)
    if not videos:
        print("측정할 영상이 없습니다(--input-dir 또는 경로 인자).", file=sys.stderr)
        return 2

    total_off = total_on = 0
    reasons: Counter[str] = Counter()
    n_measured = 0
    print(f"## dedup 측정 — {len(videos)}개 영상 (mode={dedup_on.compare_mode}·N={dedup_on.recent_window})")
    for v in videos:
        try:
            # off: 전 장면 추출(cap 없이 dedup 전 baseline 키프레임 수).
            off_frames = extract_video_representative_frame_bytes(video_path=v, max_frames=None)
            kept, skips = dedup_keyframes(off_frames, dedup_on)
        except Exception as exc:  # noqa: BLE001 — 측정 스크립트(개별 실패는 건너뛰고 집계 지속)
            print(f"  skip: {v.name} — {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        n_off, n_on = len(off_frames), len(kept)
        total_off += n_off
        total_on += n_on
        for s in skips:
            reasons[s["reason"]] += 1
        n_measured += 1
        red = (1 - n_on / n_off) * 100 if n_off else 0.0
        print(f"  {v.name}: {n_off} → {n_on} ({red:.1f}%↓)")

    if n_measured:
        agg = (1 - total_on / total_off) * 100 if total_off else 0.0
        print(f"## 집계({n_measured}건): 키프레임 {total_off} → {total_on} · 평균 {agg:.1f}%↓ "
              f"· VLM 호출 동일 비율 감소")
        print(f"## skip 사유: {dict(reasons)}")
        print(f"## SC-006 게이트(≥15%↓): {'충족' if agg >= 15.0 else '미달 — 사유 문서화 필요'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
