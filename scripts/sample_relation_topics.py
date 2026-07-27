"""관계 제안 topic 품질 **비저장 dry 샘플러** — spec 058 v2 · T1102(G11 관계 품질 확인).

무엇을 하나 (driver 검토용 근거·graph_edge 미변경)
    dev 의 registered 자산 소수(기본 6개)에 대해 관계 파이프라인의 **읽기 부분만** 재현한다:
      ① ``_fetch_source_row`` 소스 요약·modality
      ② ``find_embedding_candidates`` ∪ ``find_path_signal_candidates`` 후보(``propose_relations_for_asset`` 와 동일)
      ③ ``fetch_active_relation_kinds`` 카탈로그
      ④ **새 프롬프트**(``build_relation_proposal_prompt`` — T1101 닫힌 topic 목록)로 LLM 제안만 수신
         (``propose_edges_json`` = 단일 seam ``complete_json``·temp=0). **persist 안 함**.
    그리고 제안된 ``topic_ko`` 를 모아 **닫힌 27+기타 내 비율(in-list rate)**·목록 밖(환각) 값·
    ``relation_type_code``(kind) 분포를 리포트한다.

왜 비저장인가
    T1102 는 프롬프트 교체(T1101)가 관계 품질을 해치지 않았는지 **관측**만 하는 하드 정지점이다.
    ``sync_graph_edges``/kind 등록/lineage 를 호출하지 않으므로 DB 는 읽기 전용으로만 접촉한다.

헌법·불변식
    - LLM 단일 seam(2조)·temp=0(결정성 3조): ``propose_edges_json`` → ``complete_json`` 만 사용.
    - 후보·프롬프트·관계종류 로직은 프로덕션 경로(``asset_entry``)와 **완전히 동일 함수**를 재사용
      (샘플러가 별도 로직을 두지 않음 — 관측 충실성).

실행
    conda activate AuroraFS
    python -m scripts.sample_relation_topics --env dev --limit 6
    python -m scripts.sample_relation_topics --env dev --limit 8 --json out.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from psycopg import Connection


# ────────────────────────────────────────────────────────────────────────────
# 부트스트랩: .env.{env} 로드 → init_settings (운영 진입점과 동일 순서)
# ────────────────────────────────────────────────────────────────────────────
def _bootstrap(env: str) -> Any:
    """지정 환경으로 설정을 초기화해 돌려준다."""
    """지정 환경으로 설정을 초기화해 돌려준다."""
    from dotenv import load_dotenv

    from src.config.settings import get_current_settings, init_settings

    root = Path(__file__).resolve().parents[1]
    dotenv_path = root / f".env.{env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(env)
    return get_current_settings()


def _registered_source_ids(conn: Connection[Any], limit: int, *, spread: bool = True) -> list[str]:
    """registered 자산 id 를 결정적으로 ``limit`` 개 표본.

    - ``spread=True``(기본): 전체 코퍼스를 asset_id ASC 로 정렬한 뒤 **균등 간격**으로 뽑아
      한 적재 배치(동일 주제 군집)에 표본이 몰리지 않게 한다 → topic 다양성 확보.
    - ``spread=False``: 앞에서부터 ``limit`` 개(빠른 스모크).
    두 경로 모두 asset_id 순서 기반이라 **재현 가능**하다.
    """
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT asset_id FROM asset WHERE status='registered' ORDER BY asset_id")
        ids = [str(r["asset_id"]) for r in cur.fetchall()]
    if not ids or limit <= 0:
        return []
    if not spread or limit >= len(ids):
        return ids[:limit]
    step = len(ids) / limit  # 균등 간격 인덱스(floor) — 결정적
    return [ids[int(i * step)] for i in range(limit)]


def _read_prompts(
    conn: Connection[Any], cfg: Any, *, limit: int, spread: bool = True
) -> list[dict[str, Any]]:
    """표본 자산별 (asset_id, prompt, candidate 수) 를 **읽기 전용**으로 조립(LLM 미호출).

    프로덕션 ``propose_relations_for_asset`` 와 **동일 함수**로 후보·프롬프트를 만든다.
    """
    from src.relations.asset_candidates import find_embedding_candidates
    from src.relations.asset_entry import _fetch_source_row, union_candidates
    from src.relations.path_signal import find_path_signal_candidates
    from src.relations.prompt import build_relation_proposal_prompt
    from src.relations.relation_type_catalog import fetch_active_relation_kinds

    kinds = fetch_active_relation_kinds(conn)
    out: list[dict[str, Any]] = []
    for sid in _registered_source_ids(conn, limit, spread=spread):
        src = _fetch_source_row(conn, sid)
        if src is None:
            continue
        summary = str(src.get("summary") or "")
        emb = find_embedding_candidates(
            conn, source_asset_id=sid, top_k=cfg.relations.top_k,
            embedding_kind="st", min_sim=cfg.relations.min_sim,
        )
        path = find_path_signal_candidates(
            conn, source_asset_id=sid, limit=cfg.relations.path_top_k,
        )
        candidates = union_candidates(emb, path)
        prompt = build_relation_proposal_prompt(
            source_summary=summary,
            source_media_type=str(src.get("modality") or ""),
            candidates=candidates,
            relation_kinds_catalog=kinds,
        )
        out.append(
            {
                "asset_id": sid,
                "fs_path": str(src.get("fs_path") or ""),
                "n_candidates": len(candidates),
                "prompt": prompt,
            }
        )
    return out


def _closed_topic_set() -> set[str]:
    """닫힌 27+기타 topic_ko 집합(taxonomy_seed.json 단일 출처·프롬프트와 동일 로더)."""
    from src.relations.prompt import _load_taxonomy_topics

    return {ko for ko, _ in _load_taxonomy_topics()}


def run_sample(*, env: str, limit: int, spread: bool = True) -> dict[str, Any]:
    """dry 샘플 실행 → 리포트 dict. **persist 없음**(읽기 전용 DB + LLM 제안 관측만)."""
    from src.database.postgres_util import PostgresUtil
    from src.relations.llm_propose import (
        parse_and_normalize_edges,
        propose_edges_json,
    )
    from src.relations.schema import parse_llm_edges

    cfg = _bootstrap(env)
    db = PostgresUtil()

    # 1) 읽기 전용 트랜잭션에서 프롬프트만 조립(느린 LLM 호출은 커넥션 밖에서).
    prompts = db.execute_in_transaction(
        lambda conn: _read_prompts(conn, cfg, limit=limit, spread=spread), idempotent=True
    )

    closed = _closed_topic_set()
    topic_counter: Counter[str] = Counter()          # 정규화된 topic_ko(저장형)
    raw_topic_counter: Counter[str] = Counter()       # LLM 원문 topic_ko(literal)
    kind_counter: Counter[str] = Counter()
    per_asset: list[dict[str, Any]] = []
    n_edges = 0

    # 2) 표본마다 LLM 제안 수신(단일 seam·temp0)·비저장 파싱.
    for item in prompts:
        raw = propose_edges_json(item["prompt"])
        raw_edges = parse_llm_edges(raw)
        norm_edges = parse_and_normalize_edges(raw)
        for e in raw_edges:
            rt = str(e.get("topic_ko") or "").strip()
            if rt:
                raw_topic_counter[rt] += 1
        asset_topics: list[str] = []
        for e in norm_edges:
            n_edges += 1
            tk = str(e.get("topic_ko") or "").strip()
            if tk:
                topic_counter[tk] += 1
                asset_topics.append(tk)
            kind_counter[str(e.get("relation_type_code") or "").strip() or "(none)"] += 1
        per_asset.append(
            {
                "asset_id": item["asset_id"],
                "fs_path": item["fs_path"],
                "n_candidates": item["n_candidates"],
                "n_edges": len(norm_edges),
                "topics": asset_topics,
                "kinds": [str(e.get("relation_type_code") or "") for e in norm_edges],
            }
        )

    # 3) in-list rate·목록 밖 값 집계(정규화된 저장형 기준).
    in_list = sum(c for t, c in topic_counter.items() if t in closed)
    total_topics = sum(topic_counter.values())
    off_list = {t: c for t, c in topic_counter.items() if t not in closed}
    in_list_rate = (in_list / total_topics) if total_topics else 0.0

    return {
        "env": env,
        "sample_assets": len(prompts),
        "closed_topic_count": len(closed),
        "n_edges_proposed": n_edges,
        "n_topic_labels": total_topics,
        "in_list_count": in_list,
        "in_list_rate": round(in_list_rate, 4),
        "off_list_labels": dict(sorted(off_list.items(), key=lambda kv: (-kv[1], kv[0]))),
        "topic_distribution": dict(sorted(topic_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "raw_topic_literals": dict(sorted(raw_topic_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "kind_distribution": dict(sorted(kind_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "per_asset": per_asset,
    }


def _print_report(rep: dict[str, Any]) -> None:
    """표본 결과를 사람이 읽을 형태로 출력한다(후보·주제·프롬프트 발췌)."""
    print("=" * 72)
    print(f"[058 T1102] 관계 제안 topic 품질 dry 샘플 (env={rep['env']}·비저장)")
    print("=" * 72)
    print(f"표본 자산 수      : {rep['sample_assets']}")
    print(f"닫힌 topic 수     : {rep['closed_topic_count']} (27+기타)")
    print(f"제안 엣지 수      : {rep['n_edges_proposed']}")
    print(f"topic 라벨 수     : {rep['n_topic_labels']}")
    print(f"목록 내(in-list)  : {rep['in_list_count']}  →  in-list rate = {rep['in_list_rate']:.1%}")
    if rep["off_list_labels"]:
        print(f"목록 밖(환각)     : {rep['off_list_labels']}")
    else:
        print("목록 밖(환각)     : 없음 ✅")
    print("-" * 72)
    print("topic 분포(정규화·저장형):")
    for t, c in rep["topic_distribution"].items():
        mark = "" if t in _closed_topic_set() else "  ⚠️목록밖"
        print(f"  {c:>3}  {t}{mark}")
    print("-" * 72)
    print("kind(relation_type_code) 분포:")
    for k, c in rep["kind_distribution"].items():
        print(f"  {c:>3}  {k}")
    print("=" * 72)


def main() -> int:
    """실제 관계 후보·프롬프트를 표본으로 뽑아 눈으로 확인한다(진단용).

    운영과 **같은 함수**로 후보를 만들므로, 여기서 이상하면 운영에서도 이상하다.

    Returns:
        0=성공.
    """
    p = argparse.ArgumentParser(
        description="관계 제안 topic 품질 dry 샘플러(spec 058 T1102·비저장)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument("--limit", type=int, default=6, help="표본 registered 자산 수(기본 6)")
    p.add_argument(
        "--head", action="store_true",
        help="코퍼스 앞에서부터 표본(기본은 전체 균등 간격 spread — topic 다양성)",
    )
    p.add_argument("--json", dest="json_out", default=None, help="리포트 JSON 저장 경로(선택)")
    args = p.parse_args()

    rep = run_sample(env=args.env, limit=args.limit, spread=not args.head)
    _print_report(rep)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"리포트 JSON 저장: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
