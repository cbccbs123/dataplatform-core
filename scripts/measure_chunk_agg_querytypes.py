#!/usr/bin/env python3
"""확장 골든셋(문장/단어조합/단어) × 집계 방식 다종 KPI 측정 (019 보강).

기존 골든셋(문장 29)에 더해 **파일명 주제 prefix** 로 단어조합(주제명)·단어(주제 대표어)
질의를 생성하고(토큰 기반 relevance), 집계 방식 5종(max/topk_mean k3·k5/mix0.5/avg)을
**질의 유형별로** 비교한다. 짧은 질의일수록 MAX 의 "한 청크 복권" 효과가 커지는지 관찰.

실행: RUN_DB_E2E=1 python scripts/measure_chunk_agg_querytypes.py --env dev
읽기 전용(SELECT 만)·결정성(구조화 주입·집계 결정적).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    if os.getenv("RUN_DB_E2E") != "1":
        print("RUN_DB_E2E=1 필요(실DB 측정).")
        return 0
    from dotenv import load_dotenv

    from scripts.build_golden_ko_draft import _fetch_assets, _topic_from_filename
    from scripts.measure_chunk_agg_kpi import (
        compute_agg_kpi_table,
        format_comparison_table,
        make_db_search_fn,
    )
    from src.config.settings import ChunkAggConfig, init_settings
    from src.database.postgres_util import PostgresUtil
    from src.search.search_service import search_hybrid
    from tests.test_embedding_ab_kpi import (
        TestEmbeddingABKpi,
        _merge_ranked_ids,
        _GOLDEN_PATH,
        load_golden,
    )

    env = "dev"
    if "--env" in sys.argv:
        env = sys.argv[sys.argv.index("--env") + 1]
    p = _REPO / f".env.{env}"
    if p.is_file():
        load_dotenv(dotenv_path=p, override=False)
    init_settings(env)

    # 집계 변형 5종 — 사용자 요청대로 평균 계열을 넓게(avg=전체평균=mix w=0).
    variants = [
        ("max", ChunkAggConfig(agg="max", k=3, mix_w=0.5)),
        ("topk3", ChunkAggConfig(agg="topk_mean", k=3, mix_w=0.5)),
        ("topk5", ChunkAggConfig(agg="topk_mean", k=5, mix_w=0.5)),
        ("mix0.5", ChunkAggConfig(agg="mix", k=3, mix_w=0.5)),
        ("avg", ChunkAggConfig(agg="mix", k=3, mix_w=0.0)),  # 전체 평균
    ]

    db = PostgresUtil()
    with db:
        pool = TestEmbeddingABKpi._load_eval_pool(db)
        with db.transaction() as conn:
            assets = _fetch_assets(conn)  # st_bge registered: asset_id·fs_path

        # 주제 prefix 로 자산 그룹화
        topic_assets: dict[str, set[str]] = {}
        for row in assets:
            topic = _topic_from_filename(Path(row["fs_path"]).name)
            if topic:
                topic_assets.setdefault(topic, set()).add(str(row["asset_id"]))
        topic_assets = {t: ids for t, ids in topic_assets.items() if len(ids) >= 2}

        # (b) 단어조합 질의 = 주제명(공백), relevant = 그 주제 그룹
        combo_golden = [
            {"query": t.replace("_", " ").strip(), "relevant_asset_ids": sorted(ids)}
            for t, ids in sorted(topic_assets.items())
        ]
        # (c) 단어 질의 = 주제 첫 토큰, relevant = 같은 첫 토큰 주제들의 union(토큰 기반)
        word_groups: dict[str, set[str]] = {}
        for t, ids in topic_assets.items():
            head = t.split("_")[0].strip()
            word_groups.setdefault(head, set()).update(ids)
        word_golden = [
            {"query": w, "relevant_asset_ids": sorted(ids)}
            for w, ids in sorted(word_groups.items())
        ]
        # (a) 문장 질의 = 기존 골든(사람 검수)
        sent_golden = load_golden(_GOLDEN_PATH) if _GOLDEN_PATH.is_file() else []

        search_fn = make_db_search_fn(
            pool, search_hybrid_fn=search_hybrid, merge_fn=_merge_ranked_ids
        )

        for label, golden in [
            (f"문장(sentence) n={len(sent_golden)}", sent_golden),
            (f"단어조합(combo) n={len(combo_golden)}", combo_golden),
            (f"단어(word) n={len(word_golden)}", word_golden),
        ]:
            if not golden:
                continue
            table = compute_agg_kpi_table(golden, pool, variants, search_fn)
            print(f"\n{'='*72}\n[질의유형: {label}]")
            print(format_comparison_table(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
