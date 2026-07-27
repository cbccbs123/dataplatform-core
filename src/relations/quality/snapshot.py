"""LLM 동결 스냅샷 모델 + 순수 JSON I/O (spec 031 T002).

골든 소스마다 LLM 1회 제안(target·kind·confidence·topic)을 스냅샷으로 동결한다(FR-005).
이후 측정·임계 스윕은 스냅샷 위에서 결정적으로 돌아 **재측정 LLM 호출 0**(SC-002, 헌법 3조).
`dump_snapshot`/`load_snapshot`은 LLM/DB 불요 순수 함수다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProposedEdge:
    """LLM이 제안한 관계 엣지 — 타깃 자산·관계 kind·신뢰도(+주제·후보 임베딩 유사도)."""
    target: str
    kind: str
    confidence: float
    topic_ko: str = ""
    # 033 FR-006: 후보 단계의 코사인 유사도(emb_score) 동결값. 기본 0.0 =
    # path-only 후보(임베딩 신호 없음)·구 스냅샷 호환. 2D 자동승인 스윕/AND 게이트가 참조한다.
    emb_score: float = 0.0


@dataclass(frozen=True)
class SourceSnapshot:
    """한 소스 자산의 동결 결과 — union 후보 집합 + LLM 제안 엣지.

    033 FR-004: candidates 는 ``(target_id, emb_score)`` 쌍으로 동결한다 — N1 min_sim 스윕이
    후보 단계 recall(=골든 양성이 후보로 회수되는지)을 점수 임계로 재측정하려면 후보 코사인
    유사도가 필요하기 때문(proposed 엣지는 LLM 이 고른 부분집합이라 후보 recall 을 과소측정).
    """
    candidates: tuple[tuple[str, float], ...]
    proposed: tuple[ProposedEdge, ...]


@dataclass(frozen=True)
class Snapshot:
    """전체 스냅샷 — 측정 설정 config + 소스별 동결 결과."""
    config: dict
    sources: dict[str, SourceSnapshot]


def dump_snapshot(s: Snapshot) -> dict:
    """스냅샷을 JSON 직렬화 가능한 dict 로 변환한다(순수 함수).

    Args:
        s: 동결할 스냅샷.

    Returns:
        ``{version, config, sources}`` dict. ``candidates`` 는 ``[id, score]`` 2원소 배열로 쓴다.
    """
    return {
        "version": 1,
        "config": s.config,
        "sources": {
            sid: {
                "candidates": [[cid, score] for cid, score in ss.candidates],
                "proposed": [
                    {"target": e.target, "kind": e.kind,
                     "confidence": e.confidence, "topic_ko": e.topic_ko,
                     "emb_score": e.emb_score}
                    for e in ss.proposed],
            }
            for sid, ss in s.sources.items()},
    }


def load_snapshot(d: dict) -> Snapshot:
    """dict 를 검증해 ``Snapshot`` 으로 복원한다(순수 함수).

    구 형식 스냅샷도 읽는다 — ``candidates`` 가 문자열만 있으면 ``emb_score=0.0`` 으로,
    ``emb_score`` 키가 없으면 0.0 으로 채운다(옛 스냅샷으로도 재측정이 가능하도록).

    Args:
        d: ``dump_snapshot`` 이 만든 dict(또는 그 구 버전).

    Returns:
        복원된 ``Snapshot``.

    Raises:
        ValueError: ``version`` 이 1이 아닐 때.
    """
    if d.get("version") != 1:
        raise ValueError(f"snapshot version must be 1: {d.get('version')!r}")
    sources = {
        sid: SourceSnapshot(
            # (id, emb_score) 쌍으로 복원. 문자열만 있는 구 형식은 emb_score=0.0 으로 흡수.
            candidates=tuple(
                (str(c), 0.0) if isinstance(c, str) else (str(c[0]), float(c[1]))
                for c in v.get("candidates", [])),
            proposed=tuple(
                ProposedEdge(e["target"], e["kind"], float(e["confidence"]),
                             str(e.get("topic_ko") or ""),
                             float(e.get("emb_score") or 0.0))  # 키 없는 구 스냅샷은 0.0
                for e in v.get("proposed", [])))
        for sid, v in d.get("sources", {}).items()}
    return Snapshot(config=dict(d.get("config", {})), sources=sources)
