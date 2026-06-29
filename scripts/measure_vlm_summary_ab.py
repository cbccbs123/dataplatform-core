#!/usr/bin/env python3
"""049 T401 — VLM 요약 v1 vs v2 프롬프트 A/B 비교(FR-501·SC-004).

샘플 영상(또는 키프레임 JPEG 디렉터리)에 대해 `VLM_SUMMARY_PROMPT_V2` **off(v1)/on(v2)** 로
각각 키프레임 캡션 + 영상 레벨 reduce 를 실행해 **요약·키워드를 나란히 출력**한다. v2 의
검색지향성·구체성을 개발자가 눈으로 비교(SC-006)하기 위한 하니스다.

`--judge`(또는 `VLM_SUMMARY_AB_JUDGE=1`) 시 LLM-judge(`src/llm/client` 단일 seam·temp=0)로
"어느 쪽 키워드/요약이 더 검색지향적인지"를 판정한다(옵션·비결정 의존 최소·주 판정은 recall).

※ video 검색 recall(SC-005)은 본 하니스 밖 — 골든 영상 재캡션+재색인 후
   `scripts/measure_search_golden.py`(사람·코퍼스 게이트).

사용(실 VLM 필요):
    conda activate AuroraFS
    python scripts/measure_vlm_summary_ab.py --env dev <영상.mp4>...
    python scripts/measure_vlm_summary_ab.py --env dev --keyframe-dir <JPEG디렉터리> [--judge]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _keyframe_jpegs(args: argparse.Namespace) -> list[tuple[str, list[bytes]]]:
    """(라벨, [jpeg_bytes...]) 목록 — --keyframe-dir 면 디렉터리 JPEG, 아니면 영상에서 추출(dedup on)."""
    out: list[tuple[str, list[bytes]]] = []
    if args.keyframe_dir:
        d = Path(args.keyframe_dir)
        jpegs = [p.read_bytes() for p in sorted(d.iterdir())
                 if p.is_file() and p.suffix.lower() in _IMG_EXTS]
        out.append((d.name, jpegs))
        return out
    from src.config.settings import get_current_settings
    from src.preprocess.keyframe_dedup import KeyframeDedupConfig
    from src.preprocess.video_keyframes import extract_video_representative_frame_bytes
    cfg = get_current_settings()
    dedup = KeyframeDedupConfig(
        enabled=cfg.video_keyframe_dedup_enabled,
        hash_max=cfg.video_keyframe_dedup_hash_max,
        ssim_min=cfg.video_keyframe_dedup_ssim_min,
        ssim_gray_lo=cfg.video_keyframe_dedup_ssim_gray_lo,
        hist_min=cfg.video_keyframe_dedup_hist_min,
        compare_mode=cfg.video_keyframe_dedup_compare_mode,
        recent_window=cfg.video_keyframe_dedup_recent_window,
    )
    for v in args.videos:
        p = Path(v)
        if not p.is_file() or p.suffix.lower() not in _VIDEO_EXTS:
            print(f"  skip(영상 아님): {v}", file=sys.stderr)
            continue
        frames = extract_video_representative_frame_bytes(
            video_path=p, max_frames=cfg.video_max_keyframes, dedup=dedup
        )
        out.append((p.name, [f["jpeg_bytes"] for f in frames]))
    return out


def _run_variant(env: str, v2: str, jpegs: list[bytes]) -> dict:
    """toggle 을 env 로 켜고 settings 재초기화 후 캡션+reduce 1회. 반환 {summary, keywords}."""
    os.environ["VLM_SUMMARY_PROMPT_V2"] = v2
    from src.config.settings import init_settings
    init_settings(env)  # 전역 settings 를 toggle 반영본으로 재초기화
    from src.llm.image_summarizer import summarize_image_caption_keywords_objects_from_jpeg_bytes
    from src.llm.video_summarizer import summarize_video_from_scene_results
    scene_results = []
    for i, jb in enumerate(jpegs, start=1):
        summ = summarize_image_caption_keywords_objects_from_jpeg_bytes(jb)
        scene_results.append({
            "scene_index": i, "start_sec": float(i), "end_sec": i + 1.0,
            "frame_sec": i + 0.5, "summary": summ,
        })
    video = summarize_video_from_scene_results(scene_results)
    return {"video": video, "frames": [s["summary"] for s in scene_results]}


def main() -> int:
    ap = argparse.ArgumentParser(description="VLM 요약 v1 vs v2 A/B (049)")
    ap.add_argument("--env", default="dev", choices=["dev", "prod"])
    ap.add_argument("--keyframe-dir", default=None, help="키프레임 JPEG 디렉터리(영상 추출 생략)")
    ap.add_argument("--judge", action="store_true", help="LLM-judge 로 검색지향성 비교(seam·temp=0)")
    ap.add_argument("videos", nargs="*", help="영상 경로")
    args = ap.parse_args()

    # 키프레임 준비는 현재 toggle 무관(추출은 dedup 만). 먼저 한 번 init.
    from src.config.settings import init_settings
    init_settings(args.env)
    items = _keyframe_jpegs(args)
    if not items:
        print("측정할 키프레임이 없습니다(--keyframe-dir 또는 영상 경로).", file=sys.stderr)
        return 2

    for label, jpegs in items:
        if not jpegs:
            continue
        v1 = _run_variant(args.env, "0", jpegs)
        v2 = _run_variant(args.env, "1", jpegs)
        print(f"\n===== {label} ({len(jpegs)} keyframes) =====")
        print(f"[v1] summary: {v1['video'].get('summary','')}")
        print(f"[v2] summary: {v2['video'].get('summary','')}")
        print(f"[v1] keywords: {v1['video'].get('keywords',[])}")
        print(f"[v2] keywords: {v2['video'].get('keywords',[])}")
        if args.judge or os.getenv("VLM_SUMMARY_AB_JUDGE", "0") in {"1", "true"}:
            from src.llm.client import complete_json
            prompt = (
                "두 영상 요약(A=v1, B=v2)을 검색 적합성 기준으로 비교해 JSON만 출력:\n"
                '{ "better": "A"|"B"|"tie", "reason": "한 문장" }\n'
                "검색 적합성 = 구체 개체·주제어·검색에 쓰일 명사가 많고 일반어가 적은 쪽.\n"
                f"A: summary={v1['video'].get('summary','')} keywords={v1['video'].get('keywords',[])}\n"
                f"B: summary={v2['video'].get('summary','')} keywords={v2['video'].get('keywords',[])}"
            )
            verdict = complete_json(prompt)
            print(f"[judge] {verdict.get('better')}: {verdict.get('reason','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
