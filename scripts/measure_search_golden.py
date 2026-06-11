"""골든 58질의 KPI 하니스(025 G3, FR-003) — 검색 변경 전후의 상시 계기판. 읽기 전용.

지표:
  - recall@20(gate-off) / precision@3 (있음·엣지 중 정답≥1 질의): 활성 채널 OS 클라이언트 융합
    (search_assets_os, cutoff off — **순수 검색 품질**을 잰다. 게이트 효과는 차단율·gate-on recall 이 담당)
  - recall@20(gate-on, 027 신규): 같은 있음 질의를 **production 경로(search_hybrid·게이트 on)**로 재서
    오컷(게이트가 정답을 잘라내는 손실)을 가시화한다(F3). gate-off 대비 하락폭이 오컷 비용이다.
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

    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731

    def _dedup_ranking(rows_by_bucket: dict[str, list]) -> list[str]:
        """모달리티 버킷 → 자산 단위 합집합 랭킹(점수 내림차순 dedup, 결정적·헌법 3조)."""
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

    # ── 2) recall@20(gate-off) · p@3 (있음·엣지, 정답≥1) — 순수 검색 품질(게이트 off) ──
    client = get_client()
    scored = [q for q in queries if q.get("relevant")]
    recalls: list[float] = []
    p3s: list[float] = []
    for q in scored:
        # 027: search_assets_os 는 (buckets, gate_meta) 튜플 — 순수 품질엔 게이트 off(cutoff_enabled=False).
        buckets, _meta = search_assets_os(
            client, q["query"], modalities=_MODALITIES, k=20,
            channel=cfg.active_embed_channel, weights=cfg.opensearch_fusion_weights,
            index=cfg.opensearch_index, cutoff_enabled=False,
            bm25_operator=getattr(cfg, "search_os_bm25_operator", "or"),
        )
        seen = _dedup_ranking(buckets)
        rel = set(q["relevant"])
        recalls.append(len(rel & set(seen[:20])) / len(rel))
        top3 = seen[:3]
        p3s.append(sum(1 for a in top3 if a in rel) / max(len(top3), 1))
    print(f"## recall@20(gate-off) = {avg(recalls):.4f} · precision@3 = {avg(p3s):.4f} (질의 {len(scored)}·정답0 제외 {len([q for q in queries if q['category']!='absent']) - len(scored)})")

    # ── 2b) recall@20(gate-on, 027 신규) — production 경로(search_hybrid·게이트 on)로 오컷 가시화(F3) ──
    # 같은 있음 질의를 운영 게이트가 켜진 경로로 재서, 게이트가 정답을 잘라내는 손실을 측정한다.
    # gate-off 대비 하락폭이 오컷 비용 — 0 에 가까울수록 게이트가 정답을 보존(과필터 0).
    recalls_on: list[float] = []
    losses: list[str] = []
    for q in scored:
        out = search_hybrid(q["query"], modalities=_MODALITIES, min_scores=cfg.search_min_scores)
        seen = _dedup_ranking({k: out["results"].get(k, []) for k in _BUCKET_KEYS.values()})
        rel = set(q["relevant"])
        r_on = len(rel & set(seen[:20])) / len(rel)
        recalls_on.append(r_on)
    r_off_avg, r_on_avg = avg(recalls), avg(recalls_on)
    # 질의별 오컷(gate-on < gate-off 인 질의) 가시화 — 게이트가 정답을 잘라낸 질의 목록.
    for q, r_off, r_on in zip(scored, recalls, recalls_on, strict=True):
        if r_on + 1e-9 < r_off:
            losses.append(f"{q.get('id','?')} {q['query']}({r_off:.2f}→{r_on:.2f})")
    print(
        f"## recall@20(gate-on) = {r_on_avg:.4f} (gate-off 대비 Δ={r_on_avg-r_off_avg:+.4f}·오컷손실)"
        + (f" · 오컷 질의: {', '.join(losses)}" if losses else " · 오컷 질의 없음")
    )

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
