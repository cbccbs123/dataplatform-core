"""리포트 조립 — 골든+스냅샷+키매핑 → 메트릭 리포트 (spec 031 T008 순수부).

골든은 fs_path/content_hash **키 공간**, 스냅샷은 **asset_id 공간**이다.
`build_report`는 `key_to_id`(키→asset_id, DB 해소는 러너/사람 몫 T006)로 골든을
asset_id 공간으로 정합한 뒤 후보 recall·관계 P/R·임계 스윕을 한 번에 조립한다(SC-001).

이 함수는 **순수**(입력=Golden+Snapshot+resolved map) — LLM/DB 0·결정적이라
단위테스트로 검증한다. CLI 러너(scripts/)는 이 함수를 호출만 한다(DB/LLM 글루는 별도·사람 몫).
"""
from __future__ import annotations

from src.relations.quality.golden import Golden
from src.relations.quality.metrics import (
    candidate_recall,
    relation_metrics,
    threshold_sweep,
)
from src.relations.quality.snapshot import Snapshot

# 임계 스윕 기본 격자 — auto_approve 후보 범위를 균등히 훑는다(FR-006).
_DEFAULT_THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def build_report(
    golden: Golden,
    snapshot: Snapshot,
    key_to_id: dict[str, str],
    *,
    confidence_min: float = 0.0,
    thresholds: list[float] | None = None,
) -> dict:
    """골든+스냅샷+키매핑을 조립해 관계 품질 리포트(dict)를 반환한다.

    - 골든 키를 `key_to_id`로 asset_id 공간으로 정합. 한쪽이라도 미해소인 쌍·
      미해소 고립 키는 제외하고 `unresolved_keys`로 보고(결정적·정렬).
    - `candidate_recall`(스냅샷 candidates)·`relation_metrics`(confidence_min)·
      `threshold_sweep`(thresholds)를 호출해 결과를 한 리포트에 담는다(SC-001).
    반환 dict는 JSON 직렬화 가능(리포트 출력용).
    """
    thresholds = list(_DEFAULT_THRESHOLDS if thresholds is None else thresholds)

    # 골든 키를 asset_id로 정합. 미해소 키는 모아서 보고.
    unresolved: set[str] = set()

    def _resolve(key: str) -> str | None:
        aid = key_to_id.get(key)
        if aid is None:
            unresolved.add(key)
        return aid

    # 해소된 쌍만 채택(양쪽 모두 asset_id로 해소되어야 측정 가능).
    pairs: list[tuple[str, str]] = []
    triples: list[tuple[str, str, str]] = []
    for p in golden.pairs:
        a_id, b_id = _resolve(p.a), _resolve(p.b)
        if a_id is None or b_id is None:
            continue
        pairs.append((a_id, b_id))
        triples.append((a_id, b_id, p.kind))

    isolated: set[str] = set()
    for key in golden.isolated:
        aid = _resolve(key)
        if aid is not None:
            isolated.add(aid)

    # 스냅샷(asset_id 공간)에서 메트릭 입력 추출.
    # ss.candidates 는 (target_id, emb_score) 튜플들 — candidate_recall 은 id 집합을 기대하므로
    # id 만 추출한다(033 이 candidates 를 (id,score)로 바꾼 뒤 set(candidates)가 튜플 집합이 돼
    # candidate_recall 이 항상 0 이던 잠복 버그 수정 — 051).
    source_candidates = {
        sid: {tid for tid, _ in ss.candidates} for sid, ss in snapshot.sources.items()}
    proposed = {sid: list(ss.proposed) for sid, ss in snapshot.sources.items()}

    rm = relation_metrics(
        triples=triples, isolated=isolated, proposed=proposed,
        confidence_min=confidence_min)
    sweep = threshold_sweep(
        triples=triples, isolated=isolated, proposed=proposed, thresholds=thresholds)

    return {
        "config": dict(snapshot.config),
        "confidence_min": confidence_min,
        "n_golden_pairs": len(pairs),
        "n_isolated": len(isolated),
        "unresolved_keys": sorted(unresolved),
        "candidate_recall": candidate_recall(pairs, source_candidates),
        "relation_metrics": rm,
        "sweep": sweep,
    }
