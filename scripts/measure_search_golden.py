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


def _golden_path(fx, name: str = "golden_os.json"):
    """골든셋 경로를 해소한다 — 없으면 **무엇을 해야 하는지** 알려주고 멈춘다.

    골든셋은 실 코퍼스 자산 식별자를 담아 이 레포에 두지 않는다(별도 비공개 보관).
    ``GOLDEN_OS_PATH`` 로 파일을 직접 지정하거나 ``GOLDEN_DIR`` 로 폴더를 지정한다.

    Args:
        fx: 기본 fixture 디렉터리(레포 내 경로).
        name: 골든 파일 이름.

    Returns:
        존재하는 골든 파일 경로.

    Raises:
        SystemExit: 어느 후보에도 없을 때 — 안내 문구와 함께 종료한다.
    """
    import os
    cands = []
    if os.environ.get("GOLDEN_OS_PATH"):
        cands.append(Path(os.environ["GOLDEN_OS_PATH"]))
    if os.environ.get("GOLDEN_DIR"):
        cands.append(Path(os.environ["GOLDEN_DIR"]) / name)
    cands.append(Path(fx) / name)
    for c in cands:
        if c.is_file():
            return c
    raise SystemExit(
        f"골든셋을 찾지 못했습니다: {name}\n"
        f"  찾아본 곳: {', '.join(str(c) for c in cands)}\n"
        f"  이 파일은 실 코퍼스 자산 식별자를 담아 이 레포에 포함하지 않습니다.\n"
        f"  GOLDEN_OS_PATH=<파일> 또는 GOLDEN_DIR=<폴더> 로 지정하십시오."
    )


def main() -> int:
    """골든 질의셋으로 검색 품질을 측정한다(회수율·정확도·지연).

    Returns:
        0=성공. 기준 스냅샷을 함께 주면 이전 대비 증감까지 보여준다.
    """
    parser = argparse.ArgumentParser(description="골든 58질의 검색 KPI 하니스(025·029)")
    parser.add_argument("--skip-nomatch", action="store_true", help="production no-match 측정 생략")
    parser.add_argument(
        "--query-norm", choices=["on", "off"], default="off",
        help="augment 위에 LLM 질의 명사구 정규화(gemma temp=0) 조합도 측정(검색시점 LLM·느림)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env.dev", override=False)
    from src.config.settings import get_current_settings, init_settings

    init_settings("dev")
    from src.database.postgres_util import PostgresUtil
    from src.search.golden_guard import topic_of_filename, uncovered_topics
    from src.search.opensearch_search import get_client, search_assets_os
    from src.search.query_preprocess import noun_phrase_query
    from src.search.search_tuning import SearchTuning

    cfg = get_current_settings()
    fx = _REPO_ROOT / "tests" / "fixtures" / "search"
    golden = json.loads(_golden_path(fx).read_text(encoding="utf-8"))
    queries = golden["queries"]

    # 설정 단일 출처(search_constants 폴백) — 게이트·컷·rerank 임계.
    # PR4b: cfg.search 하위 직접 접근(방어 getattr 폐지 — 오설정 fail-fast·오타 정적 검사).
    cutoff_eps = cfg.search.os_cutoff_eps
    cutoff_floor = cfg.search.os_cutoff_floor
    result_floor = cfg.search.os_result_floor
    bm25_op = cfg.search.os_bm25_operator
    rr_top_r = cfg.search.os_rerank_top_r
    rr_tau = cfg.search.os_rerank_tau
    rr_model = cfg.search.os_rerank_model

    client = get_client()

    def run(query: str, *, cutoff: bool, rerank: bool, qnorm: bool) -> dict[str, list]:
        """한 설정 조합으로 검색을 한 번 돌린다.

        ⚠️ 설정 파일을 고치지 않고 **파라미터로 직접 주입한다** — 환경을 바꿔 가며 재는 방식은
        측정 도중 운영 설정을 오염시킨다.

        Args:
            query: 질의 문자열.
            cutoff: 적합도 컷오프를 켤지.
            rerank: 리랭커를 켤지.
            qnorm: 질의 형태소 정규화를 켤지.

        Returns:
            모달리티별 결과 버킷.
        """
        return search_assets_os(
            client, query, modalities=_MODALITIES, k=20,
            channel=cfg.embed.active_channel, index=cfg.opensearch.index,
            tuning=SearchTuning(
                weights=cfg.search.fusion_weights, cutoff_enabled=cutoff,
                cutoff_eps=cutoff_eps, cutoff_floor=cutoff_floor, result_floor=result_floor,
                bm25_operator=bm25_op,
                rerank_enabled=rerank, rerank_top_r=rr_top_r, rerank_tau=rr_tau, rerank_model=rr_model,
            ),
            query_norm_enabled=qnorm,
            query_norm_fn=(noun_phrase_query if qnorm else None),
        )[0]

    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731

    def _dedup_ranking(rows_by_bucket: dict[str, list]) -> list[str]:
        """모달리티 버킷을 자산 단위 하나의 순위로 합친다.

        측정에서는 "몇 번째에 나왔나"가 지표라 자산 단위 단일 순위가 필요하다.

        Args:
            rows_by_bucket: 모달리티별 결과 행.

        Returns:
            자산 id 순위 목록. 점수 내림차순이며 **동점은 id 로 갈라** 매번 같은 순위가 나온다.
        """
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
        """한 설정 조합의 지표를 잰다 — 재현율·상위 정확도·(선택)빈결과 차단율.

        Args:
            label: 표에 찍을 조합 이름.
            cutoff: 적합도 컷오프를 켤지.
            rerank: 리랭커를 켤지.
            qnorm: 질의 형태소 정규화를 켤지.
            nomatch: **"결과가 없어야 하는 질의"** 도 함께 잴지. 켜면 그 질의들이 실제로
                차단되는지 세고, 새어 나온 것을 목록으로 남긴다.

        Returns:
            지표 dict.
        """
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
        """측정 결과를 표로 출력한다.

        Args:
            m: 이번 측정 결과.
            base: 비교 기준. 주면 **항목마다 증감**을 함께 찍는다 — 절대값만 보면 좋아진
                것인지 나빠진 것인지 판단할 수 없다.
        """
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
