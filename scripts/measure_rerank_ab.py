"""028→029 — reranker A/B/C 측정(맥북 로컬·실모델). 골든 58질의로 세 체제를 한 번에 비교한다.

A = 027(게이트+컷·코사인 통계) / B = rerank-replace(028 평가 — 게이트 off·rerank가 대체) /
C = augment(029 채택 — 게이트 on + 켜진 버킷 안에서 rerank가 cut_rows 대체).
지표: 판정 recall@20(있음)·p@3·no-match 차단율·오컷/누수 목록·지연(p50/p95).
τ 스윕 모드(--sweep)는 후보 점수 분포(있음 정답/비정답·없음)를 출력해 τ 선택 근거를 만든다 —
채점 입력은 **프로덕션과 동일한 _rrtext**("요약: …\n키워드: …", OS mget 으로 키워드 동반).

실행: conda run -n AuroraFS python scripts/measure_rerank_ab.py [--sweep] [--tau 0.05] [--top-r 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MODALITIES = ["text", "audio", "image", "video"]


def _union_rank(buckets) -> list[str]:
    rows = sorted(
        (r for b in buckets.values() for r in b),
        key=lambda r: (-float(r.get("similarity") or 0.0), str(r.get("id"))),
    )
    seen: list[str] = []
    for r in rows:
        rid = str(r.get("id"))
        if rid not in seen:
            seen.append(rid)
    return seen


def _rrtext_map(client, index: str, ids: list[str]) -> dict[str, str]:
    """후보 id → 프로덕션 채점 텍스트 _rrtext("요약: …\n키워드: …") 매핑(OS mget·fuse_hybrid 동일 규칙).

    search_assets_os 가 응답에서 _rrtext 를 떼므로(clean), 스윕의 채점 입력을 프로덕션과 동치로 맞추려
    OS 에서 summary·keywords 를 한 번에 mget 한다(요약 단독 채점의 τ 왜곡 제거 — 028 입력변형 결론).
    """
    if not ids:
        return {}
    resp = client.mget(index=index, body={"ids": ids}, _source=["summary", "keywords"])
    out: dict[str, str] = {}
    for d in resp.get("docs", []) or []:
        src = d.get("_source") or {}
        summ = str(src.get("summary") or "")
        kw = src.get("keywords") if isinstance(src.get("keywords"), list) else []
        out[str(d.get("_id"))] = (
            f"요약: {summ}\n키워드: {' '.join(str(x) for x in kw)}" if kw else summ
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="028/029 reranker A/B/C 측정")
    parser.add_argument("--sweep", action="store_true", help="τ 분포 스윕만(채점 분포 출력·_rrtext 동치)")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--top-r", type=int, default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env.dev", override=False)
    from src.config.settings import get_current_settings, init_settings

    init_settings("dev")
    cfg = get_current_settings()
    from src.search.opensearch_search import get_client, search_assets_os
    from src.search.search_tuning import SearchTuning

    client = get_client()
    golden = json.loads((_REPO_ROOT / "tests/fixtures/search/golden_os.json").read_text(encoding="utf-8"))
    queries = golden["queries"]
    tau = args.tau if args.tau is not None else cfg.search_os_rerank_tau
    top_r = args.top_r if args.top_r is not None else cfg.search_os_rerank_top_r

    # 069 US-E(FR-E5②): 비튜닝 인자(common)와 튜닝(SearchTuning)을 분리한다. _base_tuning(weights·
    # bm25_operator)에 호출별 cutoff/rerank 를 얹어 SearchTuning 을 조립한다.
    common = {
        "modalities": _MODALITIES, "k": 20, "channel": cfg.active_embed_channel,
        "index": cfg.opensearch_index,
    }
    _base_tuning = {
        "weights": cfg.opensearch_fusion_weights,
        "bm25_operator": cfg.search_os_bm25_operator,
    }

    if args.sweep:
        # τ 근거: 모달리티 후보의 rerank 점수 분포 — 있음(정답/비정답)·없음 구분. 채점 입력은 _rrtext(프로덕션 동치).
        from src.search.reranker import score_pairs

        rel_scores, nonrel_scores, absent_scores = [], [], []
        for q in queries:
            buckets, _ = search_assets_os(
                client, q["query"],
                tuning=SearchTuning(cutoff_enabled=False, **_base_tuning), **common,
            )
            rel = set(q.get("relevant") or [])
            for b in buckets.values():
                cands = b[:top_r]
                if not cands:
                    continue
                rr = _rrtext_map(client, cfg.opensearch_index, [str(r.get("id")) for r in cands])
                texts = [rr.get(str(r.get("id"))) or str(r.get("summary") or "") for r in cands]
                scores = score_pairs(q["query"], texts, model_name=cfg.search_os_rerank_model)
                for r, sc in zip(cands, scores, strict=False):
                    if q.get("expect_empty"):
                        absent_scores.append(sc)
                    elif str(r.get("id")) in rel:
                        rel_scores.append(sc)
                    else:
                        nonrel_scores.append(sc)
            print(f"[sweep] {q['id']} 완료", file=sys.stderr)
        def stats(xs):
            if not xs:
                return "(없음)"
            xs = sorted(xs)
            return f"n={len(xs)} min={xs[0]:.4f} p25={xs[len(xs)//4]:.4f} p50={xs[len(xs)//2]:.4f} p75={xs[3*len(xs)//4]:.4f} max={xs[-1]:.4f}"
        print("## rerank 점수 분포 (τ 선택 근거·_rrtext 채점)")
        print("  있음-정답   :", stats(rel_scores))
        print("  있음-비정답 :", stats(nonrel_scores))
        print("  없음-후보   :", stats(absent_scores))
        for t in (0.01, 0.02, 0.05, 0.1, 0.2):
            keep = sum(1 for x in rel_scores if x >= t) / max(len(rel_scores), 1)
            block = sum(1 for x in absent_scores if x < t) / max(len(absent_scores), 1)
            print(f"  τ={t:0.2f}: 정답 유지율 {keep:.2%} · 없음 행 차단율 {block:.2%}")
        return 0

    # ── A/B/C 본 측정 ── A=027 / B=rerank-replace(028) / C=augment(029)
    modes = {
        "A(027)": {"cutoff_enabled": True, "rerank_enabled": False},
        "B(rerank-replace)": {"cutoff_enabled": False, "rerank_enabled": True},
        "C(augment)": {"cutoff_enabled": True, "rerank_enabled": True},
    }
    results = {}
    for mode, flags in modes.items():
        recalls, p3s = [], []
        blocked, leaks, miscuts = 0, [], []
        lat: list[float] = []
        for q in queries:
            t0 = time.perf_counter()
            buckets, _meta = search_assets_os(
                client, q["query"], **common,
                tuning=SearchTuning(
                    **flags, **_base_tuning,
                    rerank_top_r=top_r, rerank_tau=tau, rerank_model=cfg.search_os_rerank_model,
                    cutoff_eps=cfg.search_os_cutoff_eps, cutoff_floor=cfg.search_os_cutoff_floor,
                    result_floor=cfg.search_os_result_floor,
                ),
            )
            lat.append((time.perf_counter() - t0) * 1000)
            seen = _union_rank(buckets)
            if q.get("expect_empty"):
                total = sum(len(b) for b in buckets.values())
                if total == 0:
                    blocked += 1
                else:
                    leaks.append(f"{q['id']}({total})")
            elif q.get("relevant"):
                rel = set(q["relevant"])
                rec = len(rel & set(seen[:20])) / len(rel)
                recalls.append(rec)
                if rec < 1.0:
                    miscuts.append(f"{q['id']}({rec:.2f})")
                top3 = seen[:3]
                p3s.append(sum(1 for a in top3 if a in rel) / max(len(top3), 1))
        avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
        lat_s = sorted(lat)
        results[mode] = {"recall": avg(recalls), "p3": avg(p3s), "blocked": blocked,
                             "leaks": leaks, "miscuts": miscuts,
                             "p50": lat_s[len(lat_s)//2], "p95": lat_s[int(len(lat_s)*0.95)]}
        print(f"[{mode}] 측정 완료", file=sys.stderr)

    absent_n = sum(1 for q in queries if q.get("expect_empty"))
    print(f"## A/B/C (τ={tau} R={top_r})")
    for mode, r in results.items():
        print(f"  {mode}: recall@20={r['recall']:.4f} p@3={r['p3']:.4f} 차단={r['blocked']}/{absent_n} "
              f"p50={r['p50']:.0f}ms p95={r['p95']:.0f}ms")
        if r["miscuts"]:
            print(f"    부분오컷: {', '.join(r['miscuts'])}")
        if r["leaks"]:
            print(f"    누수: {', '.join(r['leaks'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
