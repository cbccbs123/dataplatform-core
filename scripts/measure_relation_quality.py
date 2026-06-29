"""관계 품질 측정 러너 (spec 031 — T007/T008/T009).

서브커맨드:
  curate   부트스트랩 후보(path_signal 쌍 + 고confidence graph_edge)를 surface 해 **검토 초안** 골든을 만든다.
           사람이 이 초안을 편집·검증(잘못된 쌍 제거·kind 확정·고립 추가)해야 골든이 된다(ADR 결정1 — silver 자동채택 금지).
  snapshot 골든 소스마다 후보(union) + LLM 제안 1회를 **동결 스냅샷**(JSON)으로 저장. graph_edge **미기록**(측정 전용·SC-004).
  measure  골든+스냅샷 → 후보recall·관계 P/R·kind·고립·임계스윕 리포트(LLM 0·결정적).

측정 전용 — 어떤 서브커맨드도 graph_edge/relation_kind 에 쓰지 않는다(읽기 + LLM 호출만). 실 DB/LLM 필요.

실행:
  python -m scripts.measure_relation_quality --env dev curate   --out tests/golden/relations/relation_golden.draft.json
  python -m scripts.measure_relation_quality --env dev snapshot --golden <golden.json> --out <snapshot.json>
  python -m scripts.measure_relation_quality --env dev measure  --golden <golden.json> --snapshot <snapshot.json>
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.database.postgres_util import PostgresUtil
from src.relations.quality.golden import Golden, parse_golden, resolve_asset_keys
from src.relations.quality.metrics import isolated_candidates
from src.relations.quality.report import build_report
from src.relations.quality.snapshot import (
    ProposedEdge,
    Snapshot,
    SourceSnapshot,
    dump_snapshot,
    load_snapshot,
)

LlmFn = Callable[[str], dict[str, Any]]


# ── 읽기 헬퍼(graph 무기록) ──────────────────────────────────────────────────
def _source_summary_modality(conn: Connection[Any], asset_id: str) -> tuple[str, str]:
    """소스 자산의 (요약, modality). 프롬프트 구성용 — 읽기 전용."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT a.modality, COALESCE(m.ext_meta->>'summary', '') AS summary "
            "FROM asset a LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id "
            "WHERE a.asset_id = %s LIMIT 1",
            (asset_id,),
        )
        r = cur.fetchone()
    if not r:
        return "", ""
    return str(r["summary"] or ""), str(r["modality"] or "")


def _read_candidates_prompt(conn: Connection[Any], sid: str, cfg: Any, config: dict) -> tuple[list, str]:
    """소스 sid 의 후보(union)와 LLM 프롬프트를 만든다(읽기 전용). LLM 호출은 호출자가 트랜잭션 밖에서."""
    from src.relations.asset_candidates import find_embedding_candidates
    from src.relations.asset_entry import union_candidates
    from src.relations.path_signal import find_path_signal_candidates
    from src.relations.prompt import build_relation_proposal_prompt
    from src.relations.relation_type_catalog import fetch_active_relation_kinds

    emb = find_embedding_candidates(
        conn, source_asset_id=sid, top_k=config["top_k"],
        embedding_kind=config["embedding_kind"], min_sim=config["min_sim"],
    )
    path = find_path_signal_candidates(conn, source_asset_id=sid, limit=cfg.relation_path_top_k)
    cands = union_candidates(emb, path)
    summary, modality = _source_summary_modality(conn, sid)
    kinds = fetch_active_relation_kinds(conn)
    prompt = build_relation_proposal_prompt(
        source_summary=summary, source_media_type=modality,
        candidates=cands, relation_kinds_catalog=kinds,
    )
    return cands, prompt


def build_snapshot(
    db: PostgresUtil, golden: Golden, *, config: dict, llm_fn: LlmFn | None = None
) -> tuple[Snapshot, dict[str, str], list[str]]:
    """골든 소스마다 후보 union + 제안(llm_fn 또는 실 LLM)을 모아 (Snapshot, key_to_id, missing) 반환.

    graph_edge/relation_kind 무기록(측정 전용·SC-004) — 후보 조회 + 제안 파싱만.
    ``llm_fn`` 주입 시 실 LLM 대신 그 함수로 프롬프트→JSON(테스트·e2e용). 미주입이면 단일 seam ``propose_edges_json``.
    LLM 호출은 **트랜잭션 밖**에서(헌법 ⑤ — 풀 점유 최소).
    """
    from src.relations.asset_entry import target_emb_score_map
    from src.relations.llm_propose import parse_and_normalize_edges, propose_edges_json

    fn: LlmFn = llm_fn if llm_fn is not None else propose_edges_json
    with db.transaction() as conn:
        mapping, missing = resolve_asset_keys(conn, golden)
    source_keys = {k for p in golden.pairs for k in (p.a, p.b)} | set(golden.isolated)
    source_ids = sorted({mapping[k] for k in source_keys if k in mapping})

    sources: dict[str, SourceSnapshot] = {}
    for sid in source_ids:
        with db.transaction() as conn:  # 짧은 읽기 트랜잭션 — 후보·프롬프트만
            cands, prompt = _read_candidates_prompt(conn, sid, _settings(), config)
        # 033 FR-006: 후보의 {id: emb_score} 맵을 동결해 제안 엣지에 부착(2D 자동승인 스윕/AND 게이트가 참조).
        # path-only 후보·후보 맵 밖 타깃(LLM 환각)은 0.0 sentinel — target_emb_score_map 이 union 후보 그대로 보존.
        emb_map = target_emb_score_map(cands)
        raw = fn(prompt)  # ★ LLM(또는 주입) — 트랜잭션 밖
        edges = parse_and_normalize_edges(raw)
        sources[sid] = SourceSnapshot(
            # 033 FR-004: 후보를 (id, emb_score) 로 동결 → N1 min_sim 스윕이 후보 단계 recall 을
            # 점수 임계로 재측정(전체 후보 기준 — proposed 부분집합 아님). path-only=0.0 그대로.
            candidates=tuple((str(c["id"]), float(c["emb_score"])) for c in cands),
            proposed=tuple(
                ProposedEdge(
                    target=str(e["target_media_item_id"]),
                    kind=str(e.get("relation_type_code") or ""),
                    confidence=float(e.get("confidence") or 0.0),
                    topic_ko=str(e.get("topic_ko") or ""),
                    emb_score=emb_map.get(str(e["target_media_item_id"]), 0.0),
                )
                for e in edges
            ),
        )
    return Snapshot(config=config, sources=sources), mapping, missing


def _settings() -> Any:
    from src.config.settings import get_current_settings
    return get_current_settings()


# ── curate: 부트스트랩 후보 surface → 검토 초안 골든 ──────────────────────────
def _bootstrap_candidate_pairs(conn: Connection[Any], *, edge_conf_min: float) -> list[dict]:
    """검토용 후보 쌍: ① 동일 폴더·stem path_signal 쌍 ② confidence≥임계 graph_edge 쌍. (asset_id·fs_path·_source·_suggest_kind)"""
    from src.relations.path_signal import find_path_signal_candidates

    seen: set[frozenset] = set()
    out: list[dict] = []
    with conn.cursor(row_factory=dict_row) as cur:
        # ① 고confidence graph_edge → kind 제안과 함께(엣지 자체가 kind 보유).
        cur.execute(
            "SELECT na.asset_id AS a, nb.asset_id AS b, rk.kind_code AS kind, ge.confidence AS conf "
            "FROM graph_edge ge "
            "JOIN node na ON na.node_id = ge.src_node AND na.node_kind = 'asset' "
            "JOIN node nb ON nb.node_id = ge.dst_node AND nb.node_kind = 'asset' "
            "JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id "
            "WHERE ge.confidence >= %s ORDER BY ge.confidence DESC, ge.edge_id ASC",
            (edge_conf_min,),
        )
        for r in cur.fetchall():
            key = frozenset((str(r["a"]), str(r["b"])))
            if key in seen:
                continue
            seen.add(key)
            out.append({"a": str(r["a"]), "b": str(r["b"]), "_source": "edge",
                        "_suggest_kind": str(r["kind"]), "_conf": float(r["conf"] or 0.0)})
        # ② path_signal 쌍(registered 자산 순회·동일폴더+stem). kind 는 사람이 확정(제안 미상).
        cur.execute("SELECT asset_id FROM asset WHERE status = 'registered' ORDER BY asset_id")
        reg_ids = [str(r["asset_id"]) for r in cur.fetchall()]
    for sid in reg_ids:
        for c in find_path_signal_candidates(conn, source_asset_id=sid, limit=10):
            key = frozenset((sid, str(c["id"])))
            if key in seen:
                continue
            seen.add(key)
            out.append({"a": sid, "b": str(c["id"]), "_source": "path_signal", "_suggest_kind": ""})
    return out


def _asset_fs_path(conn: Connection[Any], ids: set[str]) -> dict[str, str]:
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT asset_id, fs_path FROM asset WHERE asset_id = ANY(%s)", (list(ids),))
        return {str(a): str(p) for a, p in cur.fetchall()}


def _registered_asset_ids(conn: Connection[Any]) -> list[str]:
    """registered 자산 id 전체(정렬) — 고립 후보 모집단."""
    with conn.cursor() as cur:
        cur.execute("SELECT asset_id FROM asset WHERE status = 'registered' ORDER BY asset_id")
        return [str(r[0]) for r in cur.fetchall()]


def _has_any_candidate(conn: Connection[Any], sid: str, *, cfg: Any, config: dict) -> bool:
    """소스 sid 가 임베딩(≥min_sim) 또는 path-signal 후보를 1개라도 갖는가(C2 고립 판정)."""
    from src.relations.asset_candidates import find_embedding_candidates
    from src.relations.path_signal import find_path_signal_candidates

    emb = find_embedding_candidates(
        conn, source_asset_id=sid, top_k=config["top_k"],
        embedding_kind=config["embedding_kind"], min_sim=config["min_sim"])
    if emb:
        return True
    return bool(find_path_signal_candidates(conn, source_asset_id=sid, limit=cfg.relation_path_top_k))


def cmd_curate(db: PostgresUtil, out_path: str, *, edge_conf_min: float) -> dict:
    """검토 초안 골든을 만든다 — 후보 쌍을 fs_path 키로, `_review:true`·제안 kind 와 함께 출력.

    ★ 이 산출물은 **골든이 아니다** — 사람이 편집(잘못된 쌍 제거·kind 확정·`_review` 제거·고립 추가)해야 골든이 된다.
    """
    cfg = _settings()
    config = {"top_k": cfg.relation_top_k, "min_sim": cfg.relation_min_sim,
              "embedding_kind": "both"}  # snapshot/measure 기본과 동일(--embedding-kind default="both")
    with db.transaction() as conn:
        raw_pairs = _bootstrap_candidate_pairs(conn, edge_conf_min=edge_conf_min)
        ids = {p["a"] for p in raw_pairs} | {p["b"] for p in raw_pairs}
        # C2: registered 전체에서 임베딩(≥min_sim) ∨ path-signal 후보가 1개라도 있는 자산을 추림.
        #     코사인·동일폴더 신호는 대칭이라 소스 측 유무 검사로 고립을 판정한다(FR-101).
        reg_ids = _registered_asset_ids(conn)
        cand_ids = {sid for sid in reg_ids if _has_any_candidate(conn, sid, cfg=cfg, config=config)}
        iso_ids = isolated_candidates(set(reg_ids), cand_ids)
        id2path = _asset_fs_path(conn, ids | set(iso_ids))
    draft_pairs = []
    for p in raw_pairs:
        a, b = id2path.get(p["a"]), id2path.get(p["b"])
        if not a or not b or a == b:
            continue
        draft_pairs.append({"a": a, "b": b, "kind": p.get("_suggest_kind") or "REVIEW",
                            "note": f"{p['_source']}", "_review": True})
    draft = {"version": 1, "key_type": "fs_path", "pairs": draft_pairs,
             "isolated": sorted(id2path[i] for i in iso_ids if i in id2path),
             "_NOTE": "검토 초안 — 사람이 잘못된 쌍 제거·kind 확정·_review/_NOTE 제거·고립 검증 후에야 골든."}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(draft, f, ensure_ascii=False, indent=2)
    return {"draft_pairs": len(draft_pairs), "isolated": len(draft["isolated"]), "out": out_path}


# ── snapshot / measure ───────────────────────────────────────────────────────
def _make_config(args: Any, cfg: Any) -> dict:
    return {"top_k": args.top_k or cfg.relation_top_k, "min_sim": cfg.relation_min_sim,
            "embedding_kind": args.embedding_kind}


def cmd_snapshot(db: PostgresUtil, golden: Golden, *, config: dict, out_path: str) -> dict:
    snap, mapping, missing = build_snapshot(db, golden, config=config)
    payload = {"snapshot": dump_snapshot(snap), "key_to_id": mapping, "missing_keys": missing}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"sources": len(snap.sources), "missing_keys": len(missing), "out": out_path}


# 033 FR-004·005: measure 가 출력하는 임계 스윕 격자.
#   N1(min_sim): 후보 코사인 유사도 하한 후보 — recall/통과 후보 수.
#   #3(auto_approve 2D): LLM conf × 후보 emb_score 격자 — 자동승인 precision/승인 수.
_MIN_SIM_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
_AA_CONF_GRID = [0.8, 0.85, 0.9, 0.95]
_AA_EMB_GRID = [0.0, 0.3, 0.4, 0.5, 0.6]


def _resolve_golden_pairs(golden: Golden, key_to_id: dict[str, str]) -> list[tuple[str, str]]:
    """골든 쌍을 asset_id 공간으로 정합(양쪽 해소된 쌍만). 스윕 입력용·결정적."""
    pairs: list[tuple[str, str]] = []
    for p in golden.pairs:
        a, b = key_to_id.get(p.a), key_to_id.get(p.b)
        if a is not None and b is not None:
            pairs.append((a, b))
    return pairs


def cmd_measure(golden: Golden, snapshot_path: str) -> dict:
    """골든+스냅샷 → 리포트. LLM 0·DB 0(스냅샷에 key_to_id 포함 — 결정적·SC-002).

    033 FR-004·005: 동결 스냅샷 위에서 min_sim 스윕(N1)·2D 자동승인 스윕(#3) 표를 더해 출력한다.
    **읽기 전용** — graph_edge/relation_kind 미기록(measure 의 측정 전용 성질 보존·SC-004).
    - N1 스윕 후보 신호 = 동결된 SourceSnapshot.candidates 의 (id, emb_score)(전체 후보 — FR-004).
    - #3 스윕 신호 = 제안 엣지의 emb_score(자동승인 대상은 제안 엣지이므로).
    ※ 스윕 하한 탐색범위는 스냅샷 생성 시 후보 조회 min_sim(현 0.2) 이상 — 그 아래를 보려면 더 낮은
      floor 로 스냅샷 재생성. N1 감사 목표는 "0.2 과느슨 → 상향"이라 상향 스윕으로 충분.
    """
    with open(snapshot_path, encoding="utf-8") as f:
        payload = json.load(f)
    snap = load_snapshot(payload["snapshot"])
    key_to_id = {str(k): str(v) for k, v in payload.get("key_to_id", {}).items()}
    report = build_report(golden, snap, key_to_id)

    # 033 스윕 — 동결 스냅샷(asset_id 공간) 위 결정적 재측정. 골든은 key_to_id 로 정합.
    from src.relations.quality.metrics import auto_approve_sweep, min_sim_sweep

    gpairs = _resolve_golden_pairs(golden, key_to_id)
    proposed = {sid: list(ss.proposed) for sid, ss in snap.sources.items()}
    # N1: 동결된 후보 (id, emb_score) 전체를 신호로 — 후보 단계 recall 을 점수 임계로 재측정(FR-004).
    cand_by_src = {sid: list(ss.candidates) for sid, ss in snap.sources.items()}
    report["min_sim_sweep"] = min_sim_sweep(gpairs, cand_by_src, thresholds=_MIN_SIM_GRID)
    report["auto_approve_sweep"] = auto_approve_sweep(
        gpairs, proposed, conf_thresholds=_AA_CONF_GRID, emb_thresholds=_AA_EMB_GRID)
    return report


def _load_golden(path: str) -> Golden:
    with open(path, encoding="utf-8") as f:
        return parse_golden(json.load(f))


def main() -> int:
    import argparse
    from pathlib import Path

    from dotenv import load_dotenv

    from src.config.settings import init_settings

    p = argparse.ArgumentParser(description="관계 품질 측정 러너 (spec 031)")
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("curate", help="부트스트랩 후보 → 검토 초안 골든")
    pc.add_argument("--out", required=True)
    pc.add_argument("--edge-conf-min", type=float, default=0.8)

    ps = sub.add_parser("snapshot", help="골든 소스 LLM 제안 동결")
    ps.add_argument("--golden", required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--top-k", dest="top_k", type=int, default=None)
    ps.add_argument("--embedding-kind", dest="embedding_kind", choices=["st", "clip", "both"], default="both")

    pm = sub.add_parser("measure", help="골든+스냅샷 → 리포트(LLM 0)")
    pm.add_argument("--golden", required=True)
    pm.add_argument("--snapshot", required=True)

    args = p.parse_args()
    dotenv_path = Path(__file__).resolve().parents[1] / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    if args.cmd == "measure":  # DB/LLM 불요(스냅샷 기반)
        print(json.dumps(cmd_measure(_load_golden(args.golden), args.snapshot), ensure_ascii=False, indent=2))
        return 0

    db = PostgresUtil()
    with db:
        if args.cmd == "curate":
            out = cmd_curate(db, args.out, edge_conf_min=args.edge_conf_min)
        else:  # snapshot
            cfg = _settings()
            out = cmd_snapshot(db, _load_golden(args.golden), config=_make_config(args, cfg), out_path=args.out)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
