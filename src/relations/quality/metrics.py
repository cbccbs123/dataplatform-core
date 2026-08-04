"""관계 품질 지표 — 전부 순수 함수(LLM·DB 없이 결정적).

단계를 나눠 재는 이유: 품질이 나쁠 때 **후보 검색이 못 찾은 것**과 **LLM 이 잘못 판단한 것**을
구분해야 고칠 곳을 안다.
- `candidate_recall`: 후보 단계가 정답 상대를 회수했는가.
- `relation_metrics`: 제안된 엣지의 정확도·회수율·종류 일치·고립 판정.
- `threshold_sweep`: 임계를 훑어 그린 곡선(동결 스냅샷 위에서 재측정 — LLM 재호출 0).
- `coverage_curve`: 임계별 커버리지(엣지·자산 수)와 strong 비율을 함께 — 품질만 보면 화면이
  비는 트레이드오프를 놓친다.
- `wilson_interval`·`cohen_kappa`: 소표본에서 점추정을 단정하지 않기 위한 구간·일치도.
"""
from __future__ import annotations

import math

from src.relations.quality.snapshot import ProposedEdge


def isolated_candidates(
    registered_ids: set[str],
    candidate_ids: set[str],
) -> list[str]:
    """후보로 한 번도 등장하지 않은 자산 = 고립 후보.

    단순 집합 차이다 — **무엇을 후보로 볼지는 호출자가 정한다**(신뢰도 높은 기존 엣지에 등장한
    자산으로 볼 수도, 경로 신호까지 포함할 수도 있다).

    Args:
        registered_ids: 측정 모집단(등록된 전체 자산 id).
        candidate_ids: 후보로 한 번이라도 등장한 자산 id. **무엇을 후보로 볼지는 호출자가 정한다.**

    Returns:
        고립 자산 id 리스트(정렬 — 결정적).
    """
    return sorted(registered_ids - candidate_ids)


def candidate_recall(
    pairs: list[tuple[str, str]],
    source_candidates: dict[str, set[str]],
) -> float:
    """정답 쌍 중 후보 단계가 상대를 회수한 비율.

    쌍 (a,b)는 `b∈cand[a]` 또는 `a∈cand[b]`면 회수(대칭 kind 양방향 인정).

    Args:
        pairs: 정답 자산 쌍 목록.
        source_candidates: ``{소스 자산: 후보 자산 집합}``. 키가 없으면 후보 없음으로 본다.

    Returns:
        회수 비율 0.0~1.0. **골든이 비면 0.0**(1.0 아님 — 측정 불가를 만점으로 오해하지 않도록).
    """
    if not pairs:
        return 0.0
    hit = 0
    for a, b in pairs:
        if b in source_candidates.get(a, set()) or a in source_candidates.get(b, set()):
            hit += 1
    return hit / len(pairs)


def _accepted_edges(proposed: dict[str, list[ProposedEdge]], confidence_min: float):
    """confidence_min 이상(accepted)인 ``(소스, 엣지)`` 쌍만 골라 흘려보낸다(제너레이터).

    Args:
        proposed: ``{소스 자산: 제안 엣지 목록}``.
        confidence_min: 신뢰도 하한(**이상**이면 통과).

    Yields:
        ``(소스 자산 id, ProposedEdge)``.
    """
    for src, edges in proposed.items():
        for e in edges:
            if e.confidence >= confidence_min:
                yield src, e


def relation_metrics(
    *,
    triples: list[tuple[str, str, str]],
    isolated: set[str],
    proposed: dict[str, list[ProposedEdge]],
    confidence_min: float,
) -> dict:
    """제안된 엣지를 정답과 대조해 정확도·회수율·종류 일치·고립 정확도를 낸다.

    - precision: accepted 엣지 중 골든 쌍과 일치하는 비율(무순 쌍).
    - recall: 골든 쌍 중 accepted 엣지가 덮은 비율(대칭 양방향 인정).
    - kind_accuracy: 일치한 쌍 중 제안 kind가 정답과 같은 비율.
    - isolation_accuracy: 고립 자산 중 accepted 엣지 0인 비율(감사 #2 진단).

    Args:
        triples: 정답 ``(a, b, kind)`` 목록. 쌍은 **무순**으로 비교한다.
        isolated: 관계가 없어야 하는 자산 집합.
        proposed: ``{소스 자산: 제안 엣지 목록}``(측정 대상).
        confidence_min: 이 값 **이상**인 제안만 채택(accepted)으로 본다.

    Returns:
        precision·recall·kind_accuracy·isolation_accuracy + 모수(``n_pairs``·``n_isolated``·
        ``n_accepted``) dict. 분모가 0인 지표는 0.0 이다.
    """
    golden_pairs = {frozenset((a, b)): kind for a, b, kind in triples}
    accepted = list(_accepted_edges(proposed, confidence_min))
    tp_edges = matched_kind = 0
    covered: set[frozenset] = set()
    for src, e in accepted:
        key = frozenset((src, e.target))
        if key in golden_pairs:
            tp_edges += 1
            covered.add(key)
            if e.kind == golden_pairs[key]:
                matched_kind += 1
    precision = (tp_edges / len(accepted)) if accepted else 0.0
    recall = (len(covered) / len(golden_pairs)) if golden_pairs else 0.0
    kind_accuracy = (matched_kind / tp_edges) if tp_edges else 0.0
    iso_clean = sum(
        1 for a in isolated
        if not any(e.confidence >= confidence_min for e in proposed.get(a, [])))
    isolation_accuracy = (iso_clean / len(isolated)) if isolated else 0.0
    return {
        "precision": precision, "recall": recall, "kind_accuracy": kind_accuracy,
        "isolation_accuracy": isolation_accuracy,
        "n_pairs": len(golden_pairs), "n_isolated": len(isolated),
        "n_accepted": len(accepted)}


def min_sim_sweep(
    golden_pairs: list[tuple[str, str]],
    candidates_by_source: dict[str, list[tuple[str, float]]],
    *,
    thresholds: list[float],
) -> list[dict]:
    """유사도 하한을 훑으며 각 값에서의 회수율과 통과 후보 수를 잰다.

    각 하한 t 에서 `emb_score >= t` 후보만 남기고(소스별 (id, emb_score) 리스트), 남은 후보를
    `dict[str, set[str]]` 로 접어 `candidate_recall`(대칭 인정·DRY) 로 recall 을 잰다.

    Args:
        golden_pairs: 정답 자산 쌍 목록.
        candidates_by_source: ``{소스: [(후보 id, emb_score), ...]}``.
        thresholds: 시험할 하한 값들(0~1). 순서대로 결과 행이 나온다.

    Returns:
        ``[{min_sim, recall, candidates}]`` — 하한을 올릴수록 recall 은 떨어지고 후보 수는 준다.
        이 곡선에서 운영 하한을 고른다.
    """
    rows = []
    for t in thresholds:
        kept: dict[str, set[str]] = {}
        n_cand = 0
        for src, cands in candidates_by_source.items():
            survivors = {tid for tid, score in cands if score >= t}
            kept[src] = survivors
            n_cand += len(survivors)
        rows.append({
            "min_sim": t,
            "recall": candidate_recall(golden_pairs, kept),
            "candidates": n_cand,
        })
    return rows


def auto_approve_sweep(
    golden_pairs: list[tuple[str, str]],
    proposed: dict[str, list[ProposedEdge]],
    *,
    conf_thresholds: list[float],
    emb_thresholds: list[float],
) -> list[dict]:
    """신뢰도 × 유사도 격자를 훑으며 각 조합의 자동 승인 정확도와 승인 수를 잰다.

    자동승인 집합 = `confidence >= conf_min AND emb_score >= emb_min` 인 제안 엣지.
    precision = 골든 무순 쌍과 일치한 승인 수 / 전체 승인 수(승인 0 이면 0.0).

    Args:
        golden_pairs: 정답 자산 쌍 목록.
        proposed: ``{소스: 제안 엣지 목록}``.
        conf_thresholds: 시험할 LLM 신뢰도 하한 값들.
        emb_thresholds: 시험할 임베딩 유사도 하한 값들.

    Returns:
        격자점마다 ``{conf_min, emb_min, precision, approved}`` 한 행. 행 수는 두 목록 길이의 곱.
    """
    golden_keys = {frozenset((a, b)) for a, b in golden_pairs}
    rows = []
    for c in conf_thresholds:
        for e in emb_thresholds:
            approved = 0
            hit = 0
            for src, edges in proposed.items():
                for edge in edges:
                    if edge.confidence >= c and edge.emb_score >= e:
                        approved += 1
                        if frozenset((src, edge.target)) in golden_keys:
                            hit += 1
            precision = (hit / approved) if approved else 0.0
            rows.append({
                "conf_min": c, "emb_min": e,
                "precision": precision, "approved": approved,
            })
    return rows


def threshold_sweep(
    *,
    triples: list[tuple[str, str, str]],
    isolated: set[str],
    proposed: dict[str, list[ProposedEdge]],
    thresholds: list[float],
) -> list[dict]:
    """각 임계에서 `relation_metrics` 를 다시 돌려 곡선을 만든다.

    동결 스냅샷을 입력하면 LLM 을 다시 부르지 않고도 임계만 바꿔 재측정할 수 있다.

    Args:
        triples: 정답 ``(a, b, kind)`` 목록.
        isolated: 관계가 없어야 하는 자산 집합.
        proposed: ``{소스: 제안 엣지 목록}``.
        thresholds: 시험할 신뢰도 하한 값들.

    Returns:
        임계마다 ``{confidence_min, ...relation_metrics 결과}`` 한 행.
    """
    return [
        {"confidence_min": t,
         **relation_metrics(triples=triples, isolated=isolated,
                            proposed=proposed, confidence_min=t)}
        for t in thresholds]


def coverage_curve(
    *,
    edges: list[tuple[str, float, str, str]],
    verdicts: dict[str, str],
    thresholds: list[float],
) -> list[dict]:
    """자동승인 임계를 훑으며 **(엣지 수, strong 절대수, strong율, 관계 보유 자산 수)** 를 낸다.

    **왜 자산 수까지 세는가**: ``strong율`` 은 엣지를 전부 지우면 100%가 되는 지표다. 실제로
    자산의 31.7%가 이미 관계 0건이고 중앙값 차수가 1이라, 비율만 보고 임계를 올리면 "품질이
    좋아졌는데 화면은 비는" 결과가 된다. 세 값을 함께 봐야 트레이드오프가 보인다.

    Args:
        edges: ``(edge_id, confidence, src_asset_id, dst_asset_id)`` 목록.
        verdicts: ``{edge_id: 판정값}``. 없는 키는 미판정으로 보고 비율 분모에서 뺀다.
        thresholds: 훑을 confidence 임계들. 순서와 무관하게 **오름차순으로 정렬해** 반환한다.

    Returns:
        임계별 dict 목록. ``rated_count`` 는 실제로 판정된 건수(=비율 분모)로,
        ``edge_count`` 와 크게 벌어지면 그 측정은 표본이 부족하다는 뜻이다.
    """
    out: list[dict] = []
    for t in sorted(thresholds):
        kept = [e for e in edges if e[1] >= t]
        # ``error``(호출 실패)와 미판정은 분모에서 뺀다 — error 를 weak 로 세면 품질이
        # 좋아 보이는 사고가 난다.
        rated = [verdicts[e[0]] for e in kept
                 if verdicts.get(e[0]) in ("strong", "weak", "none")]
        strong = sum(1 for v in rated if v == "strong")
        assets = {a for e in kept for a in (e[2], e[3])}
        out.append({
            "threshold": t,
            "edge_count": len(kept),
            "rated_count": len(rated),
            "strong_count": strong,
            "strong_rate": (strong / len(rated)) if rated else 0.0,
            "assets_with_edge": len(assets),
        })
    return out


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson 점수 신뢰구간.

    **왜 필요한가**: 표본이 작을 때 "0건이니까 0%" 라고 단정하면 큰 손실을 0으로 장부에 올리게
    된다. 예컨대 ``conf<0.75`` 1,913건에서 69건을 봐 strong 0건이었다면 상한은 5.3%이고,
    모집단으로 환산하면 **최대 101건**이 걸려 있다. 정규근사(Wald)는 성공 0에서 폭이 0이 돼
    쓸 수 없으므로 Wilson 을 쓴다.

    Args:
        successes: 성공 건수.
        n: 표본 크기.
        z: 표준정규 분위수. 기본 1.96 = 95% 양측.

    Returns:
        ``(하한, 상한)``. ``n<=0`` 이면 정보가 없다는 뜻으로 ``(0.0, 1.0)``.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """두 판정자의 일치도(우연 일치를 보정한 값).

    LLM 이 LLM 을 채점한 순환성의 상한을 재는 데 쓴다 — 사람 판정과 얼마나 맞는지가
    "이 측정을 얼마나 믿을 수 있는가"의 답이다.

    ⚠️ n=30·3범주에서 신뢰구간이 넓다. **점추정만 보고 단정하지 않는다** — 합격 게이트가
    아니라 경보로 쓴다(spec 폐기 기준).

    Args:
        pairs: ``(판정자A 라벨, 판정자B 라벨)`` 목록.

    Returns:
        κ. 완전일치 1.0 · 우연 수준 0.0 · 우연보다 나쁘면 음수. 입력이 비면 0.0.
        한쪽 라벨만 등장해 기대일치가 1이면 κ 가 정의되지 않으므로 관례대로 완전일치는 1.0,
        아니면 0.0 을 준다.
    """
    n = len(pairs)
    if n == 0:
        return 0.0
    labels = {a for a, _ in pairs} | {b for _, b in pairs}
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((sum(1 for a, _ in pairs if a == c) / n)
             * (sum(1 for _, b in pairs if b == c) / n)
             for c in labels)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)
