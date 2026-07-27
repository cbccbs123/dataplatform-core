"""045 Phase A — 게이트 이후 버킷 정책 오케스트레이션(cut·rerank·rescue·clean).

``search_assets_os`` modality 루프에서 게이트 신호(``gate_signal``) 산출 **이후** 분기를
한곳으로 모은다. 융합·게이트 수학은 ``opensearch_search`` 순수 함수에 그대로 두고,
본 모듈은 027·029·044 rescue 경로의 **결정적 오케스트레이션**만 담당한다(헌법 3조).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.search.about_filter import about_or_filter
from src.search.query_evidence import evidence_score, lexical_rescue_keep, strong_evidence_score
from src.search.query_plan import SearchPolicy
from src.search.search_tuning import SearchTuning


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
    tuning: SearchTuning,
    rerank_fn: Callable[..., list[float]] | None,
    policy: SearchPolicy,
    passes_cutoff_fn: Callable[..., bool],
    cut_rows_fn: Callable[..., list[dict[str, Any]]],
    rerank_reorder_fn: Callable[..., tuple[list[dict[str, Any]], list[float]]],
) -> BucketPolicyOutcome:
    """융합 행에 게이트·컷·rerank·lexical rescue·aboutness 필터·응답 정제를 적용한다.

    069 US-E(FR-E5②): 게이트·컷·rerank·evidence·about·bm25 튜닝 12종을 ``tuning: SearchTuning`` 한
    묶음으로 받는다(인자 21→11·getattr 릴레이 제거). per-request 값(top·baseline·k·query·policy·
    rerank_fn)과 주입 seam(passes_cutoff_fn·cut_rows_fn·rerank_reorder_fn)은 그대로 개별 인자.

    분기 5종(``tuning.cutoff_enabled``·gate_passed·``tuning.bm25_operator`` 로 택1):
      1. 게이트 off(``tuning.cutoff_enabled=False``): 컷 없이 융합 전체 통과(gate_passed=True·cut_count=0·디버그).
      2. 게이트 통과: ``cut_rows_fn`` per-result 컷 후, ``tuning.rerank_enabled`` 면 상위 head 만 재정렬(순서만).
      3. 게이트 실패 + ``tuning.bm25_operator=='and'`` + lexical 증거: BM25 매칭 행만 ``lexical_rescue_keep``
         으로 선별 회수(AND 질의 구제).
      4. 그 외 게이트 실패: 전멸(빈 버킷 — no-match).
    (+) 위 결과에 ``tuning.about_filter_enabled`` on 이면 aboutness OR-증거 필터를 추가 적용한다.

    ``cut_count`` 는 분기에 따라 **최대 3회 재계산**된다(컷 후·rerank 후·about 필터 후) — 매번
    ``len(fused) - len(kept)`` 로 다시 구하므로 최종값은 게이트·컷·rescue·about 로 제거된 **총량**이다
    (요청 k 상한 절삭은 제외 — 컷 효과만 관측).

    Args:
        fused: 그 버킷의 융합 완료 행들(``fuse_hybrid`` 결과).
        query: 리랭크·증거 판정에 쓸 질의(정규화 적용된 값).
        top: 이 버킷 kNN 표본의 최고 코사인(``gate_signal``).
        baseline: 배경 수준(``gate_signal``).
        k: 응답 상한. 마지막에 이 개수로 자른다.
        tuning: 게이트·컷·리랭크·증거·about 튜닝 12종 묶음. per-request 값이 아니라 설정 파생이다.
        rerank_fn: 리랭커 채점 함수. ``None`` 이면 기본 seam 을 지연 로드한다.
        policy: 검색 정책(``mode`` 가 rescue 임계를 가른다).
        passes_cutoff_fn: 게이트 판정 함수 **주입** — 순환 import 를 피하려고 인자로 받는다.
        cut_rows_fn: per-result 컷 함수 주입.
        rerank_reorder_fn: 재정렬 함수 주입.

    Returns:
        ``BucketPolicyOutcome`` — 정제된 행 + 게이트 통과 여부 + 어휘 증거 유무 + 제거 총량 +
        리랭크 관측치. 게이트 실패이고 구제도 안 되면 ``rows`` 는 빈 리스트다(무관 결과를
        보여주느니 "없음"을 반환하는 설계).
    """
    has_lexical = any(r.get("_bm25") for r in fused)
    rerank_info: dict[str, Any] | None = None

    if not tuning.cutoff_enabled:
        kept = fused
        gate_passed = True
        cut_count = 0
    else:
        gate_passed = passes_cutoff_fn(top, baseline, eps=tuning.cutoff_eps, floor=tuning.cutoff_floor)
        if gate_passed:
            kept = cut_rows_fn(fused, result_floor=tuning.result_floor)
            cut_count = len(fused) - len(kept)
            if tuning.rerank_enabled and kept:
                kept, scores = rerank_reorder_fn(
                    kept,
                    query,
                    rerank_fn,
                    top_r=tuning.rerank_top_r,
                    tau=tuning.rerank_tau,
                    model_name=tuning.rerank_model,
                )
                cut_count = len(fused) - len(kept)
                rerank_info = {
                    "enabled": True,
                    "scored": len(scores),
                    "kept": len(kept),
                    "top": (max(scores) if scores else 0.0),
                }
        elif has_lexical and tuning.bm25_operator == "and":
            kept = []
            for row in fused:
                if not row.get("_bm25"):
                    continue
                keep, reason = lexical_rescue_keep(
                    row.get("matched_queries"),
                    policy=policy,
                    rescue_enabled=tuning.evidence_rescue_enabled,
                )
                if keep:
                    tagged = dict(row)
                    tagged["_keep_reason"] = reason
                    kept.append(tagged)
            cut_count = len(fused) - len(kept)
        else:
            kept = []
            cut_count = len(fused) - len(kept)

    # 073: aboutness OR-증거 필터 — 게이트·컷·rescue 생존 행에서 질의 개체와 증거(about∪keywords)가
    # 전혀 없는 행을 걸러낸다(드롭만·순서 보존·fail-safe 는 about_or_filter 내부: 전멸 시 원행 유지).
    # cut_count 를 재계산해 관측(gate_meta.cut_count)이 실제 제거 총량을 반영한다. 토글 off 면 무접촉.
    if tuning.about_filter_enabled and kept:
        kept = about_or_filter(kept, query)
        cut_count = len(fused) - len(kept)

    clean: list[dict[str, Any]] = []
    for row in kept[: int(k)]:
        out_row = {
            key: val
            for key, val in row.items()
            if key not in ("_cos", "_bm25", "_rrtext", "_keep_reason", "_about", "_kwtext")
        }
        if tuning.evidence_debug:
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
