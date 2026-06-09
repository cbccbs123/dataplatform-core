#!/usr/bin/env python3
"""청크 집계 방식 KPI 측정 러너 — 같은 골든셋·채널에 ``agg`` 방식별 비교표를 출력한다 (019 G3).

017 A/B(KoSimCSE vs BGE-M3) 하니스(``tests/test_embedding_ab_kpi.py``)가 같은 골든셋을 **채널/모델**
축으로 돌려 recall@20/MRR/nDCG 를 산출한 것과 **동형**으로, 본 러너는 같은 골든셋·같은 채널을
**집계 방식**(``agg ∈ {max, topk_mean(k=3), mix}``) 축으로 돌려 같은 지표 + **무관 노이즈 지표**
(정답 아닌 자산의 평균 순위·상위-N 노출률)를 산출·비교한다(FR-005·SC-002).

설계 — 계산부 / 실DB 호출의 경계(docs/테스트_가이드.md §0 하이브리드)
  - **순수 계산·조립부(단위 검증, DB 무관)**: ``compute_agg_kpi_table``(메트릭 집계)·
    ``format_comparison_table``(표 포매팅)·``build_agg_variants``(측정 축)·``make_db_search_fn``
    (검색 seam 조립). 검색 실행은 ``search_fn(query, payload)`` **주입 seam** 으로 분리돼,
    ``tests/test_chunk_agg_kpi.py`` 가 가짜 랭킹을 주입해 DB·모델 없이 로직만 단위로 덮는다.
  - **실DB 검색 실행(G4·사람)**: ``main()`` 은 ``RUN_DB_E2E=1`` 게이트 뒤에서만 실제 ``search_hybrid``
    를 호출한다(017 골든셋 로더·``_merge_ranked_ids`` 재사용). 골든셋은 dev DB UUID 종속이라
    미커밋 — ``scripts/build_golden_ko_draft.py`` 로 재생성 후 사람이 검수한 ``golden_ko.json`` 사용.

017 재사용: 골든 로더(``load_golden``)·멀티모달 합산 랭킹(``_merge_ranked_ids``)·지표
(``tests/fixtures/search/metrics.py``)·골든 생성기(``build_golden_ko_draft.py``)를 그대로 쓰고,
**축만** 채널→집계 방식으로 바꾼다(노이즈 지표는 019 신규).

결정성(헌법 3조): 집계 방식별 검색이 결정적이고 메트릭 산술이 결정적이라 2회 실행 동일 수치.
읽기 전용(검색·조회만, 쓰기·스키마 0). 학습 0(모델 inference only).

실행(G4·사람)
    conda activate AuroraFS
    RUN_DB_E2E=1 python scripts/measure_chunk_agg_kpi.py --env dev
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# scripts/ 직접 실행 시 'src'·'tests' 패키지를 찾도록 저장소 루트를 sys.path 에 추가(import 보다 먼저).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.search.metrics import (  # noqa: E402 — sys.path 부트스트랩 뒤에 와야 함
    mrr,
    ndcg_at_k,
    nonrelevant_exposure_at_k,
    nonrelevant_mean_rank,
    recall_at_k,
)

from src.config.settings import ChunkAggConfig  # noqa: E402 — sys.path 부트스트랩 뒤에 와야 함

# 검색 seam: (질의, 집계 payload) → 평가 풀 한정·합산된 자산 id 랭킹. payload 는 seam 이 해석한다
# (실DB 경로는 ChunkAggConfig, 단위 테스트는 임의 가짜). 메트릭 집계부는 payload 를 불투명하게 통과만 한다.
SearchFn = Callable[[str, Any], Sequence[str]]

_DEFAULT_K = 20            # recall@K·nDCG@K (017 과 동일)
_DEFAULT_EXPOSE_NS = (5, 10)  # 무관 노출률 상위-N(긴 무관 영상 관찰)
_FETCH = 100              # 버킷별 후보 수(평가 풀 한정 후에도 top-K 충분하게)


def build_agg_variants(*, k: int = 3, mix_w: float = 0.5) -> list[tuple[str, ChunkAggConfig]]:
    """측정 축: per-asset 청크 집계 방식 3종 ``(라벨, ChunkAggConfig)``. 017 채널 축의 집계 버전.

    ``max``(기존 동치·회귀 0)·``topk_mean``(상위 k 평균, FR-002)·``mix``(w·MAX+(1-w)·AVG)를 같은
    골든셋·채널에 돌려 비교한다. k·mix_w 는 FR-002 기본(3·0.5).
    """
    return [
        ("max", ChunkAggConfig(agg="max", k=k, mix_w=mix_w)),
        ("topk_mean", ChunkAggConfig(agg="topk_mean", k=k, mix_w=mix_w)),
        ("mix", ChunkAggConfig(agg="mix", k=k, mix_w=mix_w)),
    ]


def compute_agg_kpi_table(
    goldens: list[dict],
    pool: set[str],
    variants: Sequence[tuple[str, Any]],
    search_fn: SearchFn,
    *,
    k: int = _DEFAULT_K,
    expose_ns: Sequence[int] = _DEFAULT_EXPOSE_NS,
) -> dict[str, Any]:
    """골든셋을 집계 방식 축별로 검색해 채널 고정·집계별 지표 + 노이즈 지표를 집계한다(순수).

    017 ``_compute`` 의 **집계-축 버전**: 채널 루프를 ``variants``(집계 방식) 루프로 바꾸고,
    검색은 ``search_fn(query, payload)`` 주입 seam 으로 분리한다(DB·모델 비의존). 평가 풀에 정답이
    하나도 없는 질의는 건너뛴다(017 동형 — 공정 비교 대상 아님). 각 질의·집계마다:
      - recall@k / MRR / nDCG@k (정답=relevant∩pool)
      - 무관 평균 순위(``nonrelevant_mean_rank``)·상위-N 노출률(``nonrelevant_exposure_at_k``)
    질의 평균으로 집계해 ``metrics[label][지표]`` 에 담는다. ``per_query`` 는 결정성·검수용 원자료.

    반환은 평범한 dict 라 2회 호출이 ``==`` 로 동일(헌법 3조)하다.
    """
    labels = [label for label, _ in variants]
    metric_keys = (
        [f"recall@{k}", "MRR", f"nDCG@{k}", "noise_mean_rank"]
        + [f"noise@{n}" for n in expose_ns]
    )
    per_query: dict[str, dict[str, list[float]]] = {
        label: {m: [] for m in metric_keys} for label in labels
    }
    evaluated = 0
    for g in goldens:
        rel = set(g["relevant_asset_ids"]) & pool
        if not rel:
            continue  # 평가 풀에 정답 없음 → 공정 비교 불가, 건너뜀(017 동형)
        evaluated += 1
        query = g["query"]
        for label, payload in variants:
            ranked = list(search_fn(query, payload))
            per_query[label][f"recall@{k}"].append(recall_at_k(ranked, rel, k))
            per_query[label]["MRR"].append(mrr(ranked, rel))
            per_query[label][f"nDCG@{k}"].append(ndcg_at_k(ranked, rel, k))
            per_query[label]["noise_mean_rank"].append(nonrelevant_mean_rank(ranked, rel))
            for n in expose_ns:
                per_query[label][f"noise@{n}"].append(
                    nonrelevant_exposure_at_k(ranked, rel, n)
                )

    metrics: dict[str, dict[str, float]] = {
        label: {m: (statistics.mean(vals) if vals else 0.0) for m, vals in per_query[label].items()}
        for label in labels
    }
    return {
        "k": k,
        "expose_ns": list(expose_ns),
        "evaluated": evaluated,
        "pool_size": len(pool),
        "variants": labels,
        "metric_keys": metric_keys,
        "metrics": metrics,
        "per_query": per_query,
    }


def format_comparison_table(table: dict[str, Any]) -> str:
    """집계 방식별 지표 + 노이즈 지표를 사람이 읽는 비교표 문자열로(순수).

    행=지표, 열=집계 방식. recall/MRR/nDCG 는 높을수록·noise_mean_rank 는 높을수록·noise@N 은
    낮을수록 좋음을 헤더에 표기해 해석을 돕는다. 측정 결과 단정은 하지 않는다(017 동형 — 산출만).
    """
    labels = table["variants"]
    metric_keys = table["metric_keys"]
    lines = [
        f"[청크 집계 KPI] 축=agg{labels} | "
        f"평가 질의 {table['evaluated']} / 평가 풀 {table['pool_size']} | k={table['k']}",
        "  (recall/MRR/nDCG·noise_mean_rank=↑좋음, noise@N=↓좋음)",
    ]
    header = "  " + f"{'metric':<16}" + "".join(f"{lab:>14}" for lab in labels)
    lines.append(header)
    for m in metric_keys:
        row = "  " + f"{m:<16}"
        for lab in labels:
            row += f"{table['metrics'][lab][m]:>14.4f}"
        lines.append(row)
    return "\n".join(lines)


def make_db_search_fn(
    pool: set[str],
    *,
    search_hybrid_fn: Callable[..., dict[str, Any]],
    merge_fn: Callable[[dict[str, Any], set[str]], list[str]],
    text_channel: str | None = None,
    fetch: int = _FETCH,
) -> SearchFn:
    """실DB 검색 seam 조립: ``(query, agg)`` → 평가 풀 한정 멀티모달 합산 랭킹.

    017 과 동일하게 text+audio+video 를 ``search_hybrid`` 로 검색하고 ``_merge_ranked_ids`` 로 단일
    랭킹을 만든다 — 다만 채널 대신 **집계 방식**(``chunk_agg``)을 주입한다(축 교체). 모든 변형이 같은
    채널을 보도록 ``text_channel`` 을 고정해 흘린다(``None``=활성 프로파일 해소, 018). LLM 질의
    구조화는 ``structured`` 주입으로 건너뛴다(결정성). ``search_hybrid_fn``/``merge_fn`` 은 주입점이라
    단위 테스트가 가짜로 대체해 조립 로직(축 주입·멀티모달·채널 고정)을 DB 없이 검증한다. 실제 호출은 G4.
    """

    def _search(query: str, agg: Any) -> list[str]:
        res = search_hybrid_fn(
            query,
            modalities=["text", "audio", "video"],
            limit_per_bucket=fetch,
            structured={"semantic_query": query, "semantic_query_en": ""},
            chunk_agg=agg,
            text_channel=text_channel,
        )
        return merge_fn(res["results"], pool)

    return _search


# ── 부트스트랩(017 하니스와 동일 순서, 실DB 측정은 G4·사람) ──────────────────────────
# 1) RUN_DB_E2E 게이트 → 2) load_dotenv(.env.{env}) → 3) init_settings → 4) PostgresUtil() + `with` →
# 5) 골든 로더·평가 풀(017 재사용) → 6) 집계 방식별 검색·집계 → 7) 비교표 출력. 읽기 전용(SELECT 만).
def main() -> int:
    parser = argparse.ArgumentParser(
        description="청크 집계 방식 KPI 측정 러너 (019 G3; 실DB 측정은 RUN_DB_E2E=1·G4)"
    )
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--channel", default=None, help="텍스트 채널 명시(미지정=활성 프로파일)")
    args = parser.parse_args()

    if os.getenv("RUN_DB_E2E") != "1":
        print(
            "RUN_DB_E2E=1 이 아닙니다 — 실DB 측정은 G4(사람) 단계입니다.\n"
            "  RUN_DB_E2E=1 python scripts/measure_chunk_agg_kpi.py --env dev\n"
            "  (순수 계산·조립부는 tests/test_chunk_agg_kpi.py 가 DB 없이 단위로 검증합니다.)"
        )
        return 0

    # 실DB 경로 — 017 골든 로더·합산·평가 풀을 재사용한다. test 모듈 적재는 실DB 단계에서만(lazy).
    from dotenv import load_dotenv
    from tests.test_embedding_ab_kpi import (
        _GOLDEN_PATH,
        TestEmbeddingABKpi,
        _merge_ranked_ids,
        load_golden,
    )

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil
    from src.search.search_service import search_hybrid

    dotenv_path = _REPO_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    if not _GOLDEN_PATH.is_file():
        print(
            f"골든셋이 없습니다: {_GOLDEN_PATH}\n"
            "  scripts/build_golden_ko_draft.py 로 재생성 후 사람이 검수해 golden_ko.json 확정 필요."
        )
        return 2

    goldens = load_golden(_GOLDEN_PATH)
    db = PostgresUtil()
    with db:
        # 평가 풀(017 재사용): 두 채널 백필 ∩ text/audio/video. 집계 축 비교는 같은 풀에서 공정 비교.
        pool = TestEmbeddingABKpi._load_eval_pool(db)
        # 모든 변형이 같은 채널을 보도록 text_channel 을 고정(미지정=활성 프로파일). 축은 chunk_agg 뿐.
        search_fn = make_db_search_fn(
            pool,
            search_hybrid_fn=search_hybrid,
            merge_fn=_merge_ranked_ids,
            text_channel=args.channel,
        )
        table = compute_agg_kpi_table(goldens, pool, build_agg_variants(), search_fn)

    print(format_comparison_table(table))
    if table["evaluated"] == 0:
        print("⚠️ 평가 풀에 정답이 있는 골든셋 질의가 없습니다(백필·골든셋 확인).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
