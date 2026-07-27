"""골든 + 동결 스냅샷 + 키 매핑을 합쳐 품질 리포트를 만든다(순수 함수).

골든은 **파일 키**로, 스냅샷은 **asset_id** 로 자산을 가리킨다 — 그 둘을 맞추는 매핑을 받아
같은 공간으로 정렬한 뒤 후보 회수율·관계 정확도·임계 곡선을 한 번에 계산한다.

DB·LLM 을 건드리지 않으므로 단위 테스트로 검증된다(실행 스크립트가 이 함수를 부르기만 한다).
"""
from __future__ import annotations

from src.relations.quality.golden import Golden
from src.relations.quality.metrics import (
    candidate_recall,
    relation_metrics,
    threshold_sweep,
)
from src.relations.quality.snapshot import Snapshot

# 임계 곡선을 그릴 기본 격자 — 자동 승인 후보 구간을 고르게 훑는다.
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

    한쪽이라도 id 로 바꾸지 못한 쌍은 측정에서 빼고 `unresolved_keys` 로 보고한다 — 조용히
    0점 처리하면 골든이 낡은 것인지 검색이 나쁜 것인지 구분할 수 없다.

    Args:
        golden: 정답 관계셋(키 공간).
        snapshot: 동결된 후보·제안(asset_id 공간).
        key_to_id: 골든 키 → ``asset_id`` 매핑. **여기 없는 키는 측정에서 빠지고**
            ``unresolved_keys`` 로 보고된다(조용히 0점 처리하지 않는다).
        confidence_min: 본 측정에 쓸 신뢰도 하한.
        thresholds: 스윕할 임계 목록. ``None`` 이면 기본 격자(0.0~0.95)를 쓴다.

    Returns:
        JSON 직렬화 가능한 리포트 dict — ``config``·``n_golden_pairs``·``unresolved_keys``·
        ``candidate_recall``·``relation_metrics``·``sweep``.
    """
    thresholds = list(_DEFAULT_THRESHOLDS if thresholds is None else thresholds)

    # 골든 키를 asset_id로 정합. 미해소 키는 모아서 보고.
    unresolved: set[str] = set()

    def _resolve(key: str) -> str | None:
        """골든 키를 asset_id 로 바꾸고, 실패하면 ``unresolved`` 에 모아 둔다(부수효과 있음)."""
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
    # ⚠️ **id 만 뽑아야 한다** — 후보가 (id, 점수) 쌍이라 그대로 집합으로 만들면 튜플 집합이
    #    되어 id 비교가 전부 실패한다(회수율이 늘 0으로 나오던 원인).
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
