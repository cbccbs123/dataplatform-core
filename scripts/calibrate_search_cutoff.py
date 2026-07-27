"""027 G4 — OS 검색 버킷 게이트 calibration 하니스(실OS·읽기 전용).

027 클라이언트 융합 전환에 맞춰 재작성했다. 종전(023)은 모달리티당 plain-knn **probe**(추가 검색)로
원시 코사인을 재서 게이트를 측정했으나, 027 은 융합이 클라이언트로 오면서 같은 kNN 표본에 원시 코사인이
보존된다 — 따라서 별도 probe 없이 ``build_knn_body`` + ``client.search`` 로 모달리티별 kNN 코사인 리스트를
모아 ``gate_signal``(top·**robust baseline**=하위 절반 평균)로 신호를 잰다.

있음/없음 질의 라벨셋으로 운영 임계(EPS·FLOOR — cfg 에서 읽음)가 has-match 를 통과(과필터 0)·no-match 를
차단하는지 측정하고, ``search_assets_os`` 를 게이트 on/off 로 비교해 "없는 데이터 → 빈 버킷"과 per-result
컷(RESULT_FLOOR)의 효과를 확인한다. 부트스트랩은 앱 진입점과 동일(load_dotenv(.env.dev) → init_settings).
읽기 전용(헌법 6조).

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

# 게이트 신호용 kNN 표본 하한(robust baseline 안정) — 프로덕션 게이트와 **같은 상수를 import**
# 해야 보정 결과가 운영 표본과 일치한다(2026-07-15 B6·P2-25: 종전 리터럴 50 사본은 '동치' 주석만
# 있고 실제론 갈라질 수 있는 이중 정의였다 — search_constants 단일 출처로 통합).
from src.config.search_constants import OS_KNN_SAMPLE_K as _SIGNAL_K  # noqa: E402


def _bootstrap() -> None:
    """설정·로깅을 초기화한다(스크립트 진입 시 1회)."""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=_REPO_ROOT / ".env.dev", override=False)
    from src.config.settings import init_settings

    init_settings("dev")


def main() -> None:
    """게이트 임계를 고르기 위한 신호를 뽑는다 — 질의별 최고·배경 유사도와 통과 여부.

    임계는 추측이 아니라 이 표를 보고 정한다(어느 값에서 어떤 질의가 잘리는지).
    """
    _bootstrap()
    from src.config.settings import get_current_settings
    from src.search.opensearch_search import (
        _MODALITY_VALUES,
        build_knn_body,
        embed_query,
        gate_signal,
        get_client,
        knn_score_to_cosine,
        passes_cutoff,
        search_assets_os,
    )
    from src.search.search_tuning import SearchTuning

    cfg = get_current_settings()
    client = get_client()
    # 운영 임계(cfg)를 기준으로 측정한다 — search_constants 단일 출처가 settings resolver 로 들어온 값.
    eps = cfg.search.os_cutoff_eps
    floor = cfg.search.os_cutoff_floor
    result_floor = cfg.search.os_result_floor
    index = cfg.opensearch.index
    weights = cfg.search.fusion_weights

    print(
        f"# calibration(027) — EPS={eps} FLOOR={floor} RESULT_FLOOR={result_floor} "
        f"signal_k={_SIGNAL_K} index={index}\n"
    )

    def knn_cosines(vec: list[float], label: str) -> list[float]:
        """그 modality 의 kNN 코사인 리스트(robust baseline 표본). 별도 검색 1회(probe 아님 — 같은 본문)."""
        values = _MODALITY_VALUES.get(label, frozenset({label}))
        body = build_knn_body(vec, modality_values=values, k=_SIGNAL_K)
        resp = client.search(index=index, body=body)
        hits = ((resp or {}).get("hits") or {}).get("hits") or []
        return [knn_score_to_cosine(h.get("_score")) for h in hits]

    def signal_row(query: str) -> dict[str, tuple[float, float, bool]]:
        """질의 하나의 모달리티별 게이트 신호 — ``{모달리티: (최고, 배경, 통과 여부)}``."""
        vec = embed_query(query, channel=cfg.embed.active_channel)
        out: dict[str, tuple[float, float, bool]] = {}
        for label in MODALITIES:
            top, baseline = gate_signal(knn_cosines(vec, label))
            keep = passes_cutoff(top, baseline, eps=eps, floor=floor)
            out[label] = (top, baseline, keep)
        return out

    # ── 1) 게이트 신호 분포(원시 코사인 top·robust baseline·게이트) ──────────────
    for tag, queries in (("HAS-MATCH(있음)", HAS_MATCH), ("NO-MATCH(없음)", NO_MATCH)):
        print(f"## {tag}")
        for q in queries:
            row = signal_row(q)
            parts = []
            for label in MODALITIES:
                top, baseline, keep = row[label]
                mark = "KEEP" if keep else "cut "
                parts.append(
                    f"{label}: top={top:.3f} base={baseline:.3f} Δ={top-baseline:.3f} [{mark}]"
                )
            print(f"  · {q!r:14} " + " | ".join(parts))
        print()

    # ── 2) cutoff on/off 버킷 크기(없는 데이터 → 빈 버킷·per-result 컷 효과) ──────
    print("## search_assets_os 버킷 크기 (cutoff OFF → ON)")
    for tag, queries in (("HAS", HAS_MATCH), ("NO", NO_MATCH)):
        for q in queries:
            off, _ = search_assets_os(
                client, q, modalities=MODALITIES, k=20, channel=cfg.embed.active_channel,
                index=index, tuning=SearchTuning(weights=weights, cutoff_enabled=False),
            )
            on, _ = search_assets_os(
                client, q, modalities=MODALITIES, k=20, channel=cfg.embed.active_channel,
                index=index,
                tuning=SearchTuning(weights=weights, cutoff_enabled=True,
                                    cutoff_eps=eps, cutoff_floor=floor, result_floor=result_floor),
            )
            off_n = {m: len(off.get(m, [])) for m in MODALITIES}
            on_n = {m: len(on.get(m, [])) for m in MODALITIES}
            print(f"  [{tag}] {q!r:14} OFF={off_n}  →  ON={on_n}")
    print()

    # ── 3) 비용(멀티모달 게이트 p50/max Δ, ms) ────────────────────────────────────
    def timed(enabled: bool, q: str) -> float:
        """게이트를 켜고/끄고 같은 질의를 돌려 걸린 시간(초)을 잰다."""
        t0 = time.perf_counter()
        search_assets_os(
            client, q, modalities=MODALITIES, k=20, channel=cfg.embed.active_channel,
            index=index,
            tuning=SearchTuning(weights=weights, cutoff_enabled=enabled,
                                cutoff_eps=eps, cutoff_floor=floor, result_floor=result_floor),
        )
        return (time.perf_counter() - t0) * 1000.0

    print("## 비용 (멀티모달 1회, ms)")
    for q in ["아이폰", "아이패드"]:
        # 워밍업 1회 후 5회 측정(p95 근사 = 최대).
        timed(True, q)
        off_ms = sorted(timed(False, q) for _ in range(5))
        on_ms = sorted(timed(True, q) for _ in range(5))
        print(f"  {q!r:10} OFF p50={off_ms[2]:.0f} max={off_ms[-1]:.0f} | "
              f"ON p50={on_ms[2]:.0f} max={on_ms[-1]:.0f} | Δp50={on_ms[2]-off_ms[2]:.0f}ms")


if __name__ == "__main__":
    main()
