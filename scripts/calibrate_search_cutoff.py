"""023 G4 — OS 검색 적합도 컷오프 calibration 하니스(실OS·읽기 전용).

있음/없음 질의 라벨셋으로 modality 별 **실 plain-knn probe** 원시 코사인 top·baseline(표본 평균)을
재서, 기본 게이트(EPS·FLOOR)가 has-match 를 통과(과필터 0)·no-match 를 차단하는지 측정한다.
또 cutoff on/off 로 search_assets_os 버킷 크기를 비교해 "없는 데이터 → 빈 버킷"을 확인한다.

부트스트랩은 앱 진입점과 동일(load_dotenv(.env.dev) → init_settings). 읽기 전용(헌법 6조).

실행: conda run -n AuroraFS python scripts/calibrate_search_cutoff.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# scripts/ 직접 실행 시 'src' 패키지를 찾도록 저장소 루트를 sys.path 에 추가(import 보다 먼저).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 있음(코퍼스에 자산 존재) vs 없음(부재) 라벨셋 — 진단(아이패드 no-match)에서 출발.
HAS_MATCH = ["아이폰", "무선충전기", "주식", "자전거 정비"]
NO_MATCH = ["아이패드", "양자컴퓨터", "에펠탑 야경"]
MODALITIES = ["text", "audio", "image", "video"]


def _bootstrap() -> None:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=_REPO_ROOT / ".env.dev", override=False)
    from src.config.settings import init_settings

    init_settings("dev")


def main() -> None:
    _bootstrap()
    from src.config.settings import get_current_settings
    from src.search.opensearch_search import (
        _DEFAULT_CUTOFF_EPS,
        _DEFAULT_CUTOFF_FLOOR,
        _DEFAULT_PROBE_K,
        _MODALITY_VALUES,
        embed_query,
        get_client,
        passes_cutoff,
        probe_relevance,
        search_assets_os,
    )

    cfg = get_current_settings()
    client = get_client()
    eps, floor, probe_k = _DEFAULT_CUTOFF_EPS, _DEFAULT_CUTOFF_FLOOR, _DEFAULT_PROBE_K
    pipeline = getattr(cfg, "opensearch_search_pipeline", "assets-hybrid")
    index = getattr(cfg, "opensearch_index", "assets")
    weights = getattr(cfg, "opensearch_fusion_weights", (0.5, 0.5))

    print(f"# calibration — EPS={eps} FLOOR={floor} probe_k={probe_k} index={index}\n")

    def probe_row(query: str) -> dict[str, tuple[float, float, bool]]:
        vec = embed_query(query, channel=cfg.active_embed_channel)
        out: dict[str, tuple[float, float, bool]] = {}
        for label in MODALITIES:
            values = _MODALITY_VALUES.get(label, frozenset({label}))
            top, mean = probe_relevance(
                client, vec, modality_values=values, k=probe_k, index=index
            )
            keep = passes_cutoff(top, mean, eps=eps, floor=floor)
            out[label] = (top, mean, keep)
        return out

    # ── 1) probe 신호 분포(원시 코사인 top·mean·게이트) ──────────────────────
    for tag, queries in (("HAS-MATCH(있음)", HAS_MATCH), ("NO-MATCH(없음)", NO_MATCH)):
        print(f"## {tag}")
        for q in queries:
            row = probe_row(q)
            parts = []
            for label in MODALITIES:
                top, mean, keep = row[label]
                mark = "KEEP" if keep else "cut "
                parts.append(f"{label}: top={top:.3f} mean={mean:.3f} Δ={top-mean:.3f} [{mark}]")
            print(f"  · {q!r:14} " + " | ".join(parts))
        print()

    # ── 2) cutoff on/off 버킷 크기(없는 데이터 → 빈 버킷) ──────────────────────
    print("## search_assets_os 버킷 크기 (cutoff OFF → ON)")
    for tag, queries in (("HAS", HAS_MATCH), ("NO", NO_MATCH)):
        for q in queries:
            off = search_assets_os(
                client, q, modalities=MODALITIES, k=20, channel=cfg.active_embed_channel, weights=weights,
                index=index, pipeline_name=pipeline, cutoff_enabled=False,
            )
            on = search_assets_os(
                client, q, modalities=MODALITIES, k=20, channel=cfg.active_embed_channel, weights=weights,
                index=index, pipeline_name=pipeline, cutoff_enabled=True,
                cutoff_eps=eps, cutoff_floor=floor, cutoff_probe_k=probe_k,
            )
            off_n = {m: len(off.get(m, [])) for m in MODALITIES}
            on_n = {m: len(on.get(m, [])) for m in MODALITIES}
            print(f"  [{tag}] {q!r:14} OFF={off_n}  →  ON={on_n}")
    print()

    # ── 3) probe 비용(멀티모달 게이트 p95 Δ) ──────────────────────────────────
    def timed(enabled: bool, q: str) -> float:
        t0 = time.perf_counter()
        search_assets_os(
            client, q, modalities=MODALITIES, k=20, channel=cfg.active_embed_channel, weights=weights,
            index=index, pipeline_name=pipeline, cutoff_enabled=enabled,
            cutoff_eps=eps, cutoff_floor=floor, cutoff_probe_k=probe_k,
        )
        return (time.perf_counter() - t0) * 1000.0

    print("## probe 비용 (멀티모달 1회, ms)")
    for q in ["아이폰", "아이패드"]:
        # 워밍업 1회 후 5회 측정(p95 근사 = 최대).
        timed(True, q)
        off_ms = sorted(timed(False, q) for _ in range(5))
        on_ms = sorted(timed(True, q) for _ in range(5))
        print(f"  {q!r:10} OFF p50={off_ms[2]:.0f} max={off_ms[-1]:.0f} | "
              f"ON p50={on_ms[2]:.0f} max={on_ms[-1]:.0f} | Δp50={on_ms[2]-off_ms[2]:.0f}ms")


if __name__ == "__main__":
    main()
