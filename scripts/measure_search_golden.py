"""골든 58질의 KPI 하니스(025 G3·029 확장) — 검색 변경 전후의 상시 계기판. 읽기 전용.

지표(설정 조합별):
  - recall@20(gate-off) / precision@3 (있음·엣지 중 정답≥1 질의): 순수 검색 품질(게이트·rerank off).
  - recall@20(gate-on) · p@3 · no-match 차단율: **production 경로**(게이트 on)를 설정 조합으로 잰다.
    029: 같은 프로세스에서 모델 1회 로드로 **027(rerank off) vs augment(rerank on)** 를 직접 비교한다
    (search_assets_os 에 rerank/query-norm 파라미터를 직접 주입 — .env 무변경으로 채택 전 측정).

설정 조합(한 번 실행에 모델 1회 로드):
  ① gate-off(순수)  ② 027(gate-on·rerank off)  ③ augment(gate-on·rerank on)  ④(--query-norm) augment+질의정규화

결정성: 같은 코퍼스·설정에서 2회 동일(헌법 3조·rerank forward 결정적). 질의정규화 on 시에만 검색시점 LLM(gemma temp=0).

실행: conda run -n AuroraFS python scripts/measure_search_golden.py [--query-norm {on,off}] [--skip-nomatch]
      (augment 측정엔 reranker 모델 로드 — RUN_OS_E2E 환경의 실OS·실모델 필요)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MODALITIES = ["text", "audio", "image", "video"]


def main() -> int:
    parser = argparse.ArgumentParser(description="골든 58질의 검색 KPI 하니스(025·029)")
    parser.add_argument("--skip-nomatch", action="store_true", help="production no-match 측정 생략")
    parser.add_argument(
        "--query-norm", choices=["on", "off"], default="off",
        help="augment 위에 LLM 질의 명사구 정규화(gemma temp=0) 조합도 측정(검색시점 LLM·느림)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env.dev", override=False)
    from src.config import search_constants as sc
    from src.config.settings import get_current_settings, init_settings

    init_settings("dev")
    from src.database.postgres_util import PostgresUtil
    from src.search.golden_guard import topic_of_filename, uncovered_topics
    from src.search.opensearch_search import get_client, search_assets_os
    from src.search.query_preprocess import noun_phrase_query

    cfg = get_current_settings()
    fx = _REPO_ROOT / "tests" / "fixtures" / "search"
    golden = json.loads((fx / "golden_os.json").read_text(encoding="utf-8"))
    queries = golden["queries"]

    # 설정 단일 출처(search_constants 폴백) — 게이트·컷·rerank 임계.
    g = lambda name, d: getattr(cfg, name, d)  # noqa: E731
    cutoff_eps = g("search_os_cutoff_eps", sc.OS_CUTOFF_EPS_DEFAULT)
    cutoff_floor = g("search_os_cutoff_floor", sc.OS_CUTOFF_FLOOR_DEFAULT)
    result_floor = g("search_os_result_floor", sc.OS_RESULT_FLOOR_DEFAULT)
    bm25_op = g("search_os_bm25_operator", sc.OS_BM25_OPERATOR_DEFAULT)
    rr_top_r = g("search_os_rerank_top_r", sc.OS_RERANK_TOP_R_DEFAULT)
    rr_tau = g("search_os_rerank_tau", sc.OS_RERANK_TAU_DEFAULT)
    rr_model = g("search_os_rerank_model", sc.OS_RERANK_MODEL_DEFAULT)

    client = get_client()

    def run(query: str, *, cutoff: bool, rerank: bool, qnorm: bool) -> dict[str, list]:
        """한 설정으로 search_assets_os 를 호출(.env 무변경·파라미터 직접 주입)."""
        return search_assets_os(
            client, query, modalities=_MODALITIES, k=20,
            channel=cfg.active_embed_channel, weights=cfg.opensearch_fusion_weights,
            index=cfg.opensearch_index, cutoff_enabled=cutoff,
            cutoff_eps=cutoff_eps, cutoff_floor=cutoff_floor, result_floor=result_floor,
            bm25_operator=bm25_op,
            rerank_enabled=rerank, rerank_top_r=rr_top_r, rerank_tau=rr_tau, rerank_model=rr_model,
            query_norm_enabled=qnorm,
            query_norm_fn=(noun_phrase_query if qnorm else None),
        )[0]

    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731

    def _dedup_ranking(rows_by_bucket: dict[str, list]) -> list[str]:
        """모달리티 버킷 → 자산 단위 합집합 랭킹(점수 내림차순 dedup·결정적·헌법 3조)."""
        rows = sorted(
            (r for b in rows_by_bucket.values() for r in (b or [])),
            key=lambda r: (-float(r.get("similarity") or 0.0), str(r.get("id"))),
        )
        seen: list[str] = []
        for r in rows:
            rid = str(r.get("id"))
            if rid not in seen:
                seen.append(rid)
        return seen

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

    scored = [q for q in queries if q.get("relevant")]
    absent = [q for q in queries if q.get("expect_empty")]

    def measure(label: str, *, cutoff: bool, rerank: bool, qnorm: bool, nomatch: bool) -> dict[str, Any]:
        """한 설정 조합의 recall@20·p@3·(옵션)차단율을 잰다."""
        recalls: list[float] = []
        p3s: list[float] = []
        for q in scored:
            seen = _dedup_ranking(run(q["query"], cutoff=cutoff, rerank=rerank, qnorm=qnorm))
            rel = set(q["relevant"])
            recalls.append(len(rel & set(seen[:20])) / len(rel))
            top3 = seen[:3]
            p3s.append(sum(1 for a in top3 if a in rel) / max(len(top3), 1))
        r, p = avg(recalls), avg(p3s)
        blocked = leaks = None
        if nomatch:
            blocked = 0
            leaks = []
            for q in absent:
                seen = _dedup_ranking(run(q["query"], cutoff=cutoff, rerank=rerank, qnorm=qnorm))
                if len(seen) == 0:
                    blocked += 1
                else:
                    leaks.append(f"{q['id']}({len(seen)}건)")
        return {"label": label, "recall": r, "p3": p, "blocked": blocked, "total": len(absent), "leaks": leaks}

    def report(m: dict[str, Any], base: dict[str, Any] | None = None) -> None:
        d = ""
        if base is not None:
            d = f" (Δrecall={m['recall']-base['recall']:+.4f}·Δp@3={m['p3']-base['p3']:+.4f})"
        line = f"## [{m['label']}] recall@20={m['recall']:.4f} · p@3={m['p3']:.4f}{d}"
        if m["blocked"] is not None:
            line += f" · no-match 차단={m['blocked']}/{m['total']}"
            if m["leaks"]:
                line += f" · 누수:{','.join(m['leaks'])}"
        print(line)

    nm = not args.skip_nomatch
    # ① 순수(gate-off) — 게이트·rerank off, 차단은 의미 없음(생략).
    report(measure("gate-off·순수", cutoff=False, rerank=False, qnorm=False, nomatch=False))
    # ② 027 베이스라인(gate-on·rerank off) — SC-002 비교 기준.
    base027 = measure("027 gate-on", cutoff=True, rerank=False, qnorm=False, nomatch=nm)
    report(base027)
    # ③ augment(gate-on·rerank on) — SC-002 합격선: recall≥0.9396 ∧ 차단≥23/24 ∧ p@3≥0.8111.
    report(measure("augment(rerank)", cutoff=True, rerank=True, qnorm=False, nomatch=nm), base027)
    # ④ augment+질의정규화(검색시점 LLM gemma temp=0) — SC-003(--query-norm on 일 때만).
    if args.query_norm == "on":
        report(measure("augment+질의정규화", cutoff=True, rerank=True, qnorm=True, nomatch=nm), base027)

    return 0


if __name__ == "__main__":
    sys.exit(main())
