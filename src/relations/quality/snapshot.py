"""LLM 동결 스냅샷 모델 + 순수 JSON I/O (spec 031 T002).

골든 소스마다 LLM 1회 제안(target·kind·confidence·topic)을 스냅샷으로 동결한다(FR-005).
이후 측정·임계 스윕은 스냅샷 위에서 결정적으로 돌아 **재측정 LLM 호출 0**(SC-002, 헌법 3조).
`dump_snapshot`/`load_snapshot`은 LLM/DB 불요 순수 함수다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProposedEdge:
    """LLM이 제안한 관계 엣지 — 타깃 자산·관계 kind·신뢰도(+주제)."""
    target: str
    kind: str
    confidence: float
    topic_ko: str = ""


@dataclass(frozen=True)
class SourceSnapshot:
    """한 소스 자산의 동결 결과 — union 후보 집합 + LLM 제안 엣지."""
    candidates: tuple[str, ...]
    proposed: tuple[ProposedEdge, ...]


@dataclass(frozen=True)
class Snapshot:
    """전체 스냅샷 — 측정 설정 config + 소스별 동결 결과."""
    config: dict
    sources: dict[str, SourceSnapshot]


def dump_snapshot(s: Snapshot) -> dict:
    """스냅샷을 JSON 직렬화 가능한 dict로 변환한다."""
    return {
        "version": 1,
        "config": s.config,
        "sources": {
            sid: {
                "candidates": list(ss.candidates),
                "proposed": [
                    {"target": e.target, "kind": e.kind,
                     "confidence": e.confidence, "topic_ko": e.topic_ko}
                    for e in ss.proposed],
            }
            for sid, ss in s.sources.items()},
    }


def load_snapshot(d: dict) -> Snapshot:
    """dict를 검증해 `Snapshot`으로 복원한다. 버전 불일치는 `ValueError`."""
    if d.get("version") != 1:
        raise ValueError(f"snapshot version must be 1: {d.get('version')!r}")
    sources = {
        sid: SourceSnapshot(
            candidates=tuple(v.get("candidates", [])),
            proposed=tuple(
                ProposedEdge(e["target"], e["kind"], float(e["confidence"]),
                             str(e.get("topic_ko") or ""))
                for e in v.get("proposed", [])))
        for sid, v in d.get("sources", {}).items()}
    return Snapshot(config=dict(d.get("config", {})), sources=sources)
