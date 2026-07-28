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
  # shadow A/B(079) — 골든 대신 active 엣지 보유 자산 표본으로 두 변형을 각각 동결한다.
  python -m scripts.measure_relation_quality --env dev snapshot \
      --sample-active 200 --seed 20260728 --prompt-variant no-circular-hint --out <snapshot_B.json>
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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

# 🔴 스냅샷 생성은 **반드시 순차**다(1). 병렬로 올리지 마라 — 헌법 3조(결정 재현성) 위반이다.
#
# 2026-07-28 실측: 같은 시드·같은 프롬프트로 20자산을 두 번 돌렸을 때
#   · 순차(1) → 제안 20/20 완전 동일
#   · 동시(6) → 1/20 소스의 제안이 달라짐(duplicate_near→same_domain · confidence 0.8→0.85)
# 원인은 LLM 서버의 연속 배칭(continuous batching)으로 보인다 — 같은 프롬프트가 다른 배치
# 구성에 실리면 배치 행렬곱의 부동소수점 결합 순서가 달라져 로짓이 미세하게 흔들리고, 긴 생성에서
# 그 편차가 누적돼 토큰 선택이 갈린다. temperature=0 으로는 막을 수 없다(샘플링이 아니라
# 로짓 자체가 다르다).
#
# ⚠️ 판정(`scripts/judge_relations.py`)은 동시 6 이어도 결정적이다(39건 × 4회 실행 전부 동일).
#    출력이 `verdict`·`why` 두 필드로 짧아 미세 편차가 토큰 선택을 뒤집지 못한다. 즉 이 제약은
#    **긴 구조적 출력을 생성하는 호출에만** 적용된다.
#
# 대가: 자산당 ~13초라 1,000자산에 약 3.5시간이 걸린다. 그래도 재현 불가능한 측정보다는 낫다.
_SNAPSHOT_CONCURRENCY = 1

# shadow A/B 변형 — **운영 프롬프트는 바꾸지 않는다.** 여기 테이블만 갈아끼워 비교하고,
# 통과한 변형만 나중에 운영 상수로 옮긴다(spec 폐기 기준 4항).
PROMPT_VARIANTS: dict[str, dict] = {
    # 대조군 — 현행 운영 프롬프트 그대로.
    "baseline": {},
    # 순환 지시 제거. 현행 두 문장은 함께 읽으면 "전 후보에 duplicate_near 를 붙여라"가 된다 —
    # 모든 후보가 정의상 임베딩 유사도로 온 것이기 때문이다(active 83% 쏠림의 유력 원인).
    "no-circular-hint": {
        "kind_hints_override": {
            "duplicate_near": "**같은 구체적 대상**을 거의 같은 형식으로 담은 사실상 중복본일 때",
            "same_domain": "대상은 다르지만 같은 분야로 묶일 때",
        },
        "anti_dup_override": (
            "\n\n**구분:** 주제·세부주제가 같아도 **다루는 대상이 다르면** "
            "``duplicate_near`` 가 아니다. 대상이 다르고 분야만 같으면 ``same_domain`` 이다."
        ),
    },
}


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


def sample_active_sources(db: PostgresUtil, *, n: int, seed: int) -> list[str]:
    """active 엣지를 가진 자산에서 시드 고정으로 ``n`` 건을 뽑는다(읽기 전용).

    골든이 아니라 표본으로 스냅샷을 뜨는 이유: A/B 는 구·신 **상대 비교**라 정답셋이 필요 없다.
    골든 재스냅샷은 머지의 선행조건이지 실험의 선행조건이 아니다(ADR 결정 7).

    Args:
        db: DB 핸들.
        n: 뽑을 자산 수.
        seed: 난수 시드.

    Returns:
        정렬된 자산 id 목록. 풀이 ``n`` 보다 작으면 전수.

    Raises:
        ValueError: active 엣지를 가진 자산이 하나도 없을 때. 조용히 빈 스냅샷을 만들면
            A/B 가 "차이 없음"으로 보이는데 실은 아무것도 재지 않은 것이다.
    """
    # node_kind='asset' 가드는 레포 관례(graph_query·review·asset_topic_query 동일) —
    # entity 노드는 asset_id 가 NULL 이라 빼지 않으면 None 이 소스 id 로 섞인다.
    sql = """
        SELECT DISTINCT nd.asset_id::text AS asset_id
        FROM graph_edge ge
        JOIN node nd ON nd.node_id IN (ge.src_node, ge.dst_node) AND nd.node_kind = 'asset'
        WHERE ge.status = 'active'
        ORDER BY 1
    """  # 대칭 엣지는 행이 하나라 양끝을 IN 으로 함께 본다(한쪽만 보면 절반이 빠진다).
    with db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql)
        ids = [r["asset_id"] for r in cur.fetchall()]
    if not ids:
        raise ValueError(
            "active 엣지를 가진 자산이 없다 — 표본을 뽑을 수 없다. "
            "관계 생성이 한 번도 돌지 않았거나 잘못된 DB 를 보고 있는지 확인하라.")
    if len(ids) < n:
        print(f"⚠️ 표본 풀 {len(ids)}건 < 요청 {n}건 — 전수를 쓴다. "
              f"비율만 보고하지 말고 실제 n 을 함께 적어라.", flush=True)
    # 정렬된 모집단 + 고정 시드 = 같은 시드면 항상 같은 표본(SC-002 재현성).
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def _read_candidates_prompt(
    conn: Connection[Any], sid: str, cfg: Any, config: dict,
    *, prompt_variant: dict | None = None,
) -> tuple[list, str]:
    """소스 sid 의 후보(union)와 LLM 프롬프트를 만든다(읽기 전용). LLM 호출은 호출자가 트랜잭션 밖에서.

    Args:
        sid: 소스 자산 id — 이 자산을 기준으로 후보를 모은다.
        prompt_variant: shadow A/B 프롬프트 변형 정의(``PROMPT_VARIANTS`` 의 한 항목).
            ``build_relation_proposal_prompt`` 의 override 인자로 **그대로 풀어 넣는다**.
            ``None`` 이나 빈 dict(=``baseline``)이면 운영과 바이트 동일한 프롬프트다.

    Returns:
        ``(후보 목록, LLM 프롬프트)``.
    """
    from src.relations.asset_candidates import find_embedding_candidates
    from src.relations.asset_entry import union_candidates
    from src.relations.path_signal import find_path_signal_candidates
    from src.relations.prompt import build_relation_proposal_prompt
    from src.relations.relation_type_catalog import fetch_active_relation_kinds

    emb = find_embedding_candidates(
        conn, source_asset_id=sid, top_k=config["top_k"],
        embedding_kind=config["embedding_kind"], min_sim=config["min_sim"],
    )
    path = find_path_signal_candidates(conn, source_asset_id=sid, limit=cfg.relations.path_top_k)
    cands = union_candidates(emb, path)
    summary, modality = _source_summary_modality(conn, sid)
    kinds = fetch_active_relation_kinds(conn)
    prompt = build_relation_proposal_prompt(
        source_summary=summary, source_media_type=modality,
        candidates=cands, relation_kinds_catalog=kinds,
        **(prompt_variant or {}))
    return cands, prompt


def build_snapshot(
    db: PostgresUtil, golden: Golden | None = None, *, config: dict,
    llm_fn: LlmFn | None = None, source_ids: list[str] | None = None,
    prompt_variant: dict | None = None,
) -> tuple[Snapshot, dict[str, str], list[str]]:
    """소스마다 후보 union + 제안(llm_fn 또는 실 LLM)을 모아 (Snapshot, key_to_id, missing) 반환.

    ⚠️ **아무것도 저장하지 않는다** — 측정 전용이라 엣지·관계 어휘를 기록하지 않는다.
    ⚠️ LLM 호출은 **트랜잭션 밖**에서 한다 — 느린 호출이 커넥션을 붙잡으면 다른 작업이 밀린다.

    Args:
        db: DB 핸들.
        golden: 정답 묶음. 주면 **여기서 소스를 도출한다**(쌍 양끝 + 고립). ``None`` 이면
            ``source_ids`` 로 소스를 받는다.
        config: 후보 조회 설정(임계·상한 등).
        llm_fn: 제안 함수. **바꿔 끼울 수 있게 열어 뒀다** — 실제 LLM 없이 고정 응답으로
            측정 배선을 검증한다. ``None``(기본)이면 실 LLM(``propose_edges_json``).
        source_ids: 소스 자산 id 목록. **골든 없이** 표본으로 스냅샷을 뜰 때 쓴다
            (구·신 프롬프트 A/B 는 상대 비교라 정답셋이 필요 없다). ``None`` 이면 ``golden`` 경로.
        prompt_variant: 프롬프트 변형 정의. ``None``(기본)이면 운영과 동일한 프롬프트를 쓴다.

    Returns:
        ``(스냅샷, 골든 키→자산 id 매핑, 해소 못 한 키 목록)``.
        ``source_ids`` 경로에서는 골든이 없으므로 매핑·미해소가 빈 값이다.

    Raises:
        ValueError: ``golden`` 과 ``source_ids`` 를 둘 다 주거나 둘 다 안 줬을 때.
    """
    from src.relations.asset_entry import target_emb_score_map
    from src.relations.llm_propose import parse_and_normalize_edges, propose_edges_json

    # 소스가 어디서 오는지 모호하면 "무엇을 측정했는가"가 흐려진다 — DB 를 건드리기 전에 막는다.
    if (golden is None) == (source_ids is None):
        raise ValueError("golden 과 source_ids 중 **정확히 하나**를 준다")

    fn: LlmFn = llm_fn if llm_fn is not None else propose_edges_json
    if golden is not None:
        with db.transaction() as conn:
            mapping, missing = resolve_asset_keys(conn, golden)
        source_keys = {k for p in golden.pairs for k in (p.a, p.b)} | set(golden.isolated)
        sids = sorted({mapping[k] for k in source_keys if k in mapping})
    else:
        # 표본 경로 — 정답셋이 없으니 해소할 키도 없다(매핑·미해소는 빈 값).
        mapping, missing, sids = {}, [], sorted(source_ids or [])

    def _one(sid: str) -> tuple[str, SourceSnapshot]:
        """소스 하나를 동결한다 — 다른 소스와 완전히 독립이라 병렬 실행이 안전하다."""
        with db.transaction() as conn:  # 짧은 읽기 트랜잭션 — 후보·프롬프트만
            cands, prompt = _read_candidates_prompt(
                conn, sid, _settings(), config, prompt_variant=prompt_variant)
        # 033 FR-006: 후보의 {id: emb_score} 맵을 동결해 제안 엣지에 부착(2D 자동승인 스윕/AND 게이트가 참조).
        # path-only 후보·후보 맵 밖 타깃(LLM 환각)은 0.0 sentinel — target_emb_score_map 이 union 후보 그대로 보존.
        emb_map = target_emb_score_map(cands)
        raw = fn(prompt)  # ★ LLM(또는 주입) — 트랜잭션 밖
        edges = parse_and_normalize_edges(raw)
        return sid, SourceSnapshot(
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

    # 소스별로 병렬 처리한다 — LLM 이 자산당 ~13초라 순차로는 1,000자산에 3시간 넘는다.
    # **결정성은 유지된다**: ① 소스마다 결과가 독립이라 서로 영향이 없고 ② `ex.map` 이 입력
    # 순서대로 결과를 돌려주며 ③ `sids` 가 이미 정렬돼 있어 dict 삽입 순서까지 같다.
    # ⚠️ 동시성 상한은 DB 풀(max_pool_size 기본 10)보다 작게 유지한다 — 넘으면 커넥션 대기로
    #    오히려 느려진다. 각 워커가 짧은 트랜잭션을 하나씩만 쓴다.
    workers = min(_SNAPSHOT_CONCURRENCY, max(1, len(sids)))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            sources = dict(ex.map(_one, sids))
    else:
        sources = dict(_one(sid) for sid in sids)
    return Snapshot(config=config, sources=sources), mapping, missing


def _settings() -> Any:
    """설정을 초기화해 돌려준다(측정 스크립트는 운영과 같은 설정을 써야 한다)."""
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
    """자산 id 를 경로로 바꿀 매핑을 **한 번에** 조회한다(리포트에 이름을 붙이려고).

    Args:
        conn: DB 연결.
        ids: 조회할 자산 id 집합. **비어 있으면 DB 를 건드리지 않는다**.

    Returns:
        ``{asset_id: 경로}``.
    """
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


def cmd_curate(db: PostgresUtil, out_path: str, *, edge_conf_min: float) -> dict:
    """검토 초안 골든을 만든다 — 후보 쌍을 fs_path 키로, `_review:true`·제안 kind 와 함께 출력.

    ★ 이 산출물은 **골든이 아니다** — 사람이 편집(잘못된 쌍 제거·kind 확정·`_review` 제거·고립 검증)해야 골든이 된다.
    """
    with db.transaction() as conn:
        raw_pairs = _bootstrap_candidate_pairs(conn, edge_conf_min=edge_conf_min)
        # 부트스트랩 쌍(고conf graph_edge + path_signal)에 등장한 자산 = 관계/경로 후보 보유.
        ids = {p["a"] for p in raw_pairs} | {p["b"] for p in raw_pairs}
        # C2(051): registered 중 그 집합에 없는 자산 = 관계 0 ∧ path 0 = 고립 후보(관계 단계·FR-101).
        #   035 isolation 의미(평가완료·엣지 0)와 일치. min_sim 이 낮아 임베딩 후보는 거의 모두 존재하므로
        #   "임베딩 후보 0" 대신 "관계/경로 후보 0"으로 고립을 정의한다(임베딩 전수 스캔 불요·결정적).
        reg_ids = _registered_asset_ids(conn)
        iso_ids = isolated_candidates(set(reg_ids), ids)
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
    """측정 조건을 dict 로 굳힌다 — 스냅샷에 함께 저장돼 "어떤 조건의 수치인지" 남는다.

    ``prompt_variant`` 를 함께 굳히는 이유: shadow A/B 는 같은 표본을 두 번 떠서 비교하는데,
    파일명(``snap_A``/``snap_B``)으로만 구분하면 **보고서에서 두 팔을 뒤바꿔 적는 사고**를
    막을 방법이 없다. 스냅샷 자체가 자기가 어느 팔인지 알고 있어야 한다.
    측정 로직은 이 키를 읽지 않으므로 수치에는 영향이 없다.

    Args:
        args: 파싱된 CLI 인자. ``top_k``·``embedding_kind``·``prompt_variant`` 를 읽는다.
        cfg: 활성 설정. ``args`` 가 비운 값의 기본값 출처다.

    Returns:
        스냅샷 ``config`` 에 그대로 실릴 dict.
    """
    return {"top_k": args.top_k or cfg.relations.top_k, "min_sim": cfg.relations.min_sim,
            "embedding_kind": args.embedding_kind,
            # 골든 경로(변형 개념이 없는 호출)에서도 "baseline" 이 박힌다 — 사후에
            # "이 스냅샷은 운영 프롬프트였다"를 단언할 수 있으니 그편이 낫다.
            "prompt_variant": getattr(args, "prompt_variant", None) or "baseline"}


def assert_same_candidates(a: Snapshot, b: Snapshot) -> None:
    """두 스냅샷의 후보 집합이 같은지 단언한다(A/B 오염 검출).

    프롬프트만 바꾼 A/B 에서 후보가 달라졌다면 후보 단계가 함께 흔들렸다는 뜻이고, 그러면
    "프롬프트 때문에 좋아졌다"고 말할 수 없다. **실험을 계속하기 전에 멈춘다.**

    Args:
        a: 대조군 스냅샷.
        b: 실험군 스냅샷.

    Raises:
        AssertionError: 소스 집합이나 어느 소스의 후보 목록이 다를 때.
    """
    if set(a.sources) != set(b.sources):
        raise AssertionError(
            f"소스 집합이 다르다: A만 {sorted(set(a.sources) - set(b.sources))[:5]} / "
            f"B만 {sorted(set(b.sources) - set(a.sources))[:5]}")
    for sid in sorted(a.sources):
        ca, cb = a.sources[sid].candidates, b.sources[sid].candidates
        if ca != cb:
            raise AssertionError(f"후보가 다르다(source={sid}): A={ca[:3]} B={cb[:3]}")


def cmd_snapshot(
    db: PostgresUtil, golden: Golden | None = None, *, config: dict, out_path: str,
    source_ids: list[str] | None = None, prompt_variant: dict | None = None,
) -> dict:
    """골든(또는 표본) 소스마다 후보·LLM 제안을 받아 **파일로 동결**한다.

    이후 임계를 바꿔 가며 재측정할 때 LLM 을 다시 부르지 않기 위한 단계다.

    Args:
        db: DB 핸들.
        golden: 정답 묶음. 주면 여기서 소스를 도출한다. ``None`` 이면 ``source_ids`` 경로.
        config: 후보 조회 설정(임계·상한 등) — 스냅샷에 함께 저장된다.
        out_path: 동결 JSON 을 쓸 경로.
        source_ids: 골든 없이 쓸 소스 자산 id 목록(shadow A/B 표본). ``None`` 이면 ``golden`` 경로.
            ``golden`` 과 **정확히 하나만** 준다(둘 다/둘 다 아님이면 ``build_snapshot`` 이 거부).
        prompt_variant: 프롬프트 변형 정의(``PROMPT_VARIANTS`` 의 한 항목).
            ``None``·빈 dict(=baseline)이면 운영과 바이트 동일한 프롬프트다.

    Returns:
        동결 결과 요약 dict(파일 경로·소스 수·미해소 키 등).
    """
    snap, mapping, missing = build_snapshot(
        db, golden, config=config, source_ids=source_ids, prompt_variant=prompt_variant)
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
    """골든 쌍을 자산 id 공간으로 옮긴다.

    Args:
        golden: 정답 묶음(사람이 읽는 키로 적혀 있다).
        key_to_id: 키 → 자산 id 매핑.

    Returns:
        자산 id 쌍 목록. **양쪽이 모두 해소된 쌍만** 담는다 — 한쪽만 아는 쌍을 넣으면
        재현율 분모가 부풀어 지표가 실제보다 나빠 보인다.
    """
    pairs: list[tuple[str, str]] = []
    for p in golden.pairs:
        a, b = key_to_id.get(p.a), key_to_id.get(p.b)
        if a is not None and b is not None:
            pairs.append((a, b))
    return pairs


def cmd_measure(golden: Golden, snapshot_path: str, *, confidence_min: float = 0.0) -> dict:
    """골든+스냅샷 → 리포트. LLM 0·DB 0(스냅샷에 key_to_id 포함 — 결정적·SC-002).

    ``confidence_min``: 제안 엣지 accepted 판정 임계. **프로덕션 자동승인(RELATION_AUTO_APPROVE_MIN=0.9)
    으로 측정해야 precision/recall/isolation 이 실제 동작을 반영**한다(051 — 0.0 이면 저신뢰 제안까지
    accepted 로 세어 isolation_accuracy 가 항상 0). 비회귀 게이트는 baseline 의 confidence_min 을 재사용한다.

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
    report = build_report(golden, snap, key_to_id, confidence_min=confidence_min)

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


def _dump_report(report: dict, path: str) -> None:
    """측정 리포트를 비교 기준 파일로 동결한다.

    Args:
        report: 측정 결과.
        path: 저장 경로.

    **키를 정렬해 쓴다** — 그래야 내용이 같을 때 파일 diff 가 비고, 무엇이 실제로 달라졌는지
    바로 보인다.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)


def _load_golden(path: str) -> Golden:
    """골든 파일을 읽어 검증까지 마친 객체로 돌려준다(형식 오류면 예외)."""
    with open(path, encoding="utf-8") as f:
        return parse_golden(json.load(f))


def main() -> int:
    """관계 품질 측정 CLI 진입점 — 하위 명령(스냅샷 동결·측정·큐레이션)을 분기한다.

    Returns:
        0=성공, 그 외=실패(셸 종료 코드).
    """
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

    ps = sub.add_parser("snapshot", help="골든(또는 active 표본) 소스 LLM 제안 동결")
    ps.add_argument("--golden", default=None, help="골든 파일 경로(--sample-active 와 택일)")
    ps.add_argument("--out", required=True)
    ps.add_argument("--top-k", dest="top_k", type=int, default=None)
    ps.add_argument("--embedding-kind", dest="embedding_kind", choices=["st", "clip", "both"], default="both")
    ps.add_argument("--sample-active", dest="sample_active", type=int, default=None,
                    help="골든 대신 active 엣지 보유 자산 N건을 시드 고정 표본으로 쓴다(A/B용)")
    ps.add_argument("--seed", type=int, default=None, help="--sample-active 의 표본 시드")
    ps.add_argument("--prompt-variant", dest="prompt_variant",
                    choices=sorted(PROMPT_VARIANTS), default="baseline",
                    help="프롬프트 변형(baseline=운영과 동일). shadow A/B 전용")

    pm = sub.add_parser("measure", help="골든+스냅샷 → 리포트(LLM 0)")
    pm.add_argument("--golden", required=True)
    pm.add_argument("--snapshot", required=True)
    pm.add_argument("--out", default=None, help="리포트를 baseline_report.json 로 동결(선택)")
    pm.add_argument("--confidence-min", dest="confidence_min", type=float, default=None,
                    help="accepted 판정 임계(미지정 시 RELATION_AUTO_APPROVE_MIN — 프로덕션 자동승인)")

    args = p.parse_args()
    if args.cmd == "snapshot":
        # 소스가 골든인지 표본인지 모호하면 "무엇을 측정했는가"가 흐려진다 — DB 를 열기 전에 막는다.
        if (args.golden is None) == (args.sample_active is None):
            ps.error("--golden 과 --sample-active 중 정확히 하나를 지정한다")
        # 시드가 없으면 매 실행 표본이 달라져 A/B 두 팔의 소스가 어긋난다(재현성 SC-002).
        if args.sample_active is not None and args.seed is None:
            ps.error("--sample-active 를 쓸 때는 --seed 를 함께 준다(같은 시드 = 같은 표본)")

    dotenv_path = Path(__file__).resolve().parents[1] / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    if args.cmd == "measure":  # DB/LLM 불요(스냅샷 기반)
        # confidence_min 미지정이면 프로덕션 자동승인 임계(RELATION_AUTO_APPROVE_MIN)로 측정.
        cmin = args.confidence_min if args.confidence_min is not None else _settings().relations.auto_approve_min
        report = cmd_measure(_load_golden(args.golden), args.snapshot, confidence_min=cmin)
        if args.out:
            _dump_report(report, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    db = PostgresUtil()
    with db:
        if args.cmd == "curate":
            out = cmd_curate(db, args.out, edge_conf_min=args.edge_conf_min)
        else:  # snapshot — 소스는 골든 또는 active 표본 중 하나(위에서 상호배타 검증됨)
            cfg = _settings()
            sids = (sample_active_sources(db, n=args.sample_active, seed=args.seed)
                    if args.sample_active is not None else None)
            out = cmd_snapshot(
                db, _load_golden(args.golden) if args.golden else None,
                config=_make_config(args, cfg), out_path=args.out, source_ids=sids,
                prompt_variant=PROMPT_VARIANTS[args.prompt_variant])
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
