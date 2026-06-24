"""045 Phase A — 게이트 이후 버킷 정책 오케스트레이션(cut·rerank·rescue·clean).

``search_assets_os`` modality 루프에서 게이트 신호(``gate_signal``) 산출 **이후** 분기를
한곳으로 모은다. 융합·게이트 수학은 ``opensearch_search`` 순수 함수에 그대로 두고,
본 모듈은 027·029·044 rescue 경로의 **결정적 오케스트레이션**만 담당한다(헌법 3조).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.search.query_evidence import evidence_score, lexical_rescue_keep, strong_evidence_score
from src.search.query_plan import SearchPolicy


@dataclass(frozen=True, slots=True)
class BucketPolicyOutcome:
    """``apply_bucket_policy`` 결과 — 정제 행 + gate_meta 부분 필드."""

    rows: list[dict[str, Any]]
    gate_passed: bool
    lexical_evidence: bool
    cut_count: int
    rerank: dict[str, Any] | None


def apply_bucket_policy(
    fused: list[dict[str, Any]],
    *,
    query: str,
    top: float,
    baseline: float,
    k: int,
    cutoff_enabled: bool,
    cutoff_eps: float,
    cutoff_floor: float,
    result_floor: float,
    bm25_operator: str,
    rerank_enabled: bool,
    rerank_fn: Callable[..., list[float]] | None,
    rerank_top_r: int,
    rerank_tau: float,
    rerank_model: str,
    policy: SearchPolicy,
    evidence_rescue_enabled: bool,
    evidence_debug: bool,
    passes_cutoff_fn: Callable[..., bool],
    cut_rows_fn: Callable[..., list[dict[str, Any]]],
    rerank_reorder_fn: Callable[..., tuple[list[dict[str, Any]], list[float]]],
) -> BucketPolicyOutcome:
    """융합 행에 게이트·컷·rerank·lexical rescue·응답 정제를 적용한다.

    ``passes_cutoff_fn``·``cut_rows_fn``·``rerank_reorder_fn`` 은 ``opensearch_search`` 의
    순수 함수를 주입받아 단위 테스트에서 mock·동일 구현을 공유한다(순환 import 방지).
  """
    has_lexical = any(r.get("_bm25") for r in fused)
    rerank_info: dict[str, Any] | None = None

    if not cutoff_enabled:
        kept = fused
        gate_passed = True
        cut_count = 0
    else:
        gate_passed = passes_cutoff_fn(top, baseline, eps=cutoff_eps, floor=cutoff_floor)
        if gate_passed:
            kept = cut_rows_fn(fused, result_floor=result_floor)
            cut_count = len(fused) - len(kept)
            if rerank_enabled and kept:
                kept, scores = rerank_reorder_fn(
                    kept,
                    query,
                    rerank_fn,
                    top_r=rerank_top_r,
                    tau=rerank_tau,
                    model_name=rerank_model,
                )
                cut_count = len(fused) - len(kept)
                rerank_info = {
                    "enabled": True,
                    "scored": len(scores),
                    "kept": len(kept),
                    "top": (max(scores) if scores else 0.0),
                }
        elif has_lexical and bm25_operator == "and":
            kept = []
            for row in fused:
                if not row.get("_bm25"):
                    continue
                keep, reason = lexical_rescue_keep(
                    row.get("matched_queries"),
                    policy=policy,
                    rescue_enabled=evidence_rescue_enabled,
                )
                if keep:
                    tagged = dict(row)
                    tagged["_keep_reason"] = reason
                    kept.append(tagged)
            cut_count = len(fused) - len(kept)
        else:
            kept = []
            cut_count = len(fused) - len(kept)

    clean: list[dict[str, Any]] = []
    for row in kept[: int(k)]:
        out_row = {
            key: val
            for key, val in row.items()
            if key not in ("_cos", "_bm25", "_rrtext", "_keep_reason")
        }
        if evidence_debug:
            mq = out_row.get("matched_queries")
            out_row["evidence_score"] = evidence_score(mq)
            out_row["strong_evidence_score"] = strong_evidence_score(mq)
            out_row["gate_passed"] = gate_passed
            out_row["keep_reason"] = row.get("_keep_reason") or (
                "gate_passed" if gate_passed else "unknown"
            )
        clean.append(out_row)

    return BucketPolicyOutcome(
        rows=clean,
        gate_passed=gate_passed,
        lexical_evidence=has_lexical,
        cut_count=cut_count,
        rerank=rerank_info,
    )


__all__ = ["BucketPolicyOutcome", "apply_bucket_policy"]
