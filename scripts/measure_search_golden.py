"""골든 58질의 KPI 하니스(025 G3, FR-003) — 검색 변경 전후의 상시 계기판. 읽기 전용.

지표:
  - recall@20 / precision@3 (있음·엣지 중 정답≥1 질의): 활성 채널 OS 하이브리드(search_assets_os,
    production 구성 — 023 게이트·024 임계는 끄고 **순수 검색 품질**을 잰다. 게이트 효과는 차단율이 담당)
  - no-match 차단율 (없음 24질의): **production 경로(search_hybrid)** 로 전 모달리티 빈 버킷 비율
  - 코퍼스-골든 정합: 미커버 토픽 목록(비어 있어야 정상 — 신규 데이터 시 골든 질의 추가 강제)

결정성: 같은 코퍼스·설정에서 2회 동일(헌법 3조). LLM 0(2조).

실행: conda run -n AuroraFS python scripts/measure_search_golden.py [--skip-nomatch]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MODALITIES = ["text", "audio", "image", "video"]
_BUCKET_KEYS = {"text": "text_documents", "audio": "audio", "image": "image", "video": "video"}


def main() -> int:
    parser = argparse.ArgumentParser(description="골든 58질의 검색 KPI 하니스(025)")
    parser.add_argument("--skip-nomatch", action="store_true", help="production no-match 측정 생략")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env.dev", override=False)
    from src.config.settings import get_current_settings, init_settings

    init_settings("dev")
    from src.database.postgres_util import PostgresUtil
    from src.search.golden_guard import topic_of_filename, uncovered_topics
    from src.search.opensearch_search import get_client, search_assets_os
    from src.search.search_service import search_hybrid

    cfg = get_current_settings()
    fx = _REPO_ROOT / "tests" / "fixtures" / "search"
    golden = json.loads((fx / "golden_os.json").read_text(encoding="utf-8"))
    queries = golden["queries"]

    # ── 1) 코퍼스-골든 정합 가드 ─────────────────────────────────────────────
    db = PostgresUtil()
    with db, db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT fs_path FROM asset WHERE status='registered'")
        corpus_topics = {topic_of_filename(Path(r[0]).name) for r in cur.fetchall()}
    golden_topics: set[str] = set()
    for q in queries:
        golden_topics.update(q.get("topics", []))
    missing = uncovered_topics(corpus_topics, golden_topics)
    print(f"## 정합 가드 — 코퍼스 토픽 {len(corpus_topics)} · 골든 커버 {len(golden_topics)} · 미커버 {missing or '없음'}")

    # ── 2) recall@20 · p@3 (있음·엣지, 정답≥1) — 순수 검색 품질(게이트 무관) ──
    client = get_client()
    scored = [q for q in queries if q.get("relevant")]
    recalls: list[float] = []
    p3s: list[float] = []
    for q in scored:
        buckets = search_assets_os(
            client, q["query"], modalities=_MODALITIES, k=20,
            channel=cfg.active_embed_channel, weights=cfg.opensearch_fusion_weights,
            index=cfg.opensearch_index, pipeline_name=cfg.opensearch_search_pipeline,
            cutoff_enabled=False,
            bm25_operator=getattr(cfg, "search_os_bm25_operator", "or"),
        )
        # 자산 단위 합집합 랭킹(모달리티 버킷 → 점수 내림차순 dedup)
        rows = sorted(
            (r for b in buckets.values() for r in b),
            key=lambda r: (-float(r.get("similarity") or 0.0), str(r.get("id"))),
        )
        seen: list[str] = []
        for r in rows:
            rid = str(r.get("id"))
            if rid not in seen:
                seen.append(rid)
        rel = set(q["relevant"])
        top20 = set(seen[:20])
        recalls.append(len(rel & top20) / len(rel))
        top3 = seen[:3]
        p3s.append(sum(1 for a in top3 if a in rel) / max(len(top3), 1))
    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
    print(f"## recall@20 = {avg(recalls):.4f} · precision@3 = {avg(p3s):.4f} (질의 {len(scored)}·정답0 제외 {len([q for q in queries if q['category']!='absent']) - len(scored)})")

    # ── 3) no-match 차단율 (없음 24, production 경로 — 게이트·임계 포함) ─────
    if not args.skip_nomatch:
        absent = [q for q in queries if q.get("expect_empty")]
        blocked = 0
        leaks: list[str] = []
        for q in absent:
            out = search_hybrid(q["query"], min_scores=cfg.search_min_scores)
            total = sum(len(out["results"].get(_BUCKET_KEYS[m], []) or []) for m in _MODALITIES)
            if total == 0:
                blocked += 1
            else:
                leaks.append(f"{q['id']} {q['query']}({total}건)")
        print(f"## no-match 차단율 = {blocked}/{len(absent)} ({100*blocked/max(len(absent),1):.0f}%)" + (f" · 누수: {', '.join(leaks)}" if leaks else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
