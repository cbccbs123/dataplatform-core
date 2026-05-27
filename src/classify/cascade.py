"""도메인-불가지 분류 cascade — Stage 1(시그니처) → 2(어휘) → 3(온프레미스 LLM).

도메인 지식은 ``DomainProfileProvider`` 가 주는 ``DomainProfile`` 목록에 산다(코드 분기 없음).
앞 단계가 확정하면 즉시 반환. general 은 '어느 도메인도 충분치 않음'의 엔진 내장 폴백.
라우팅 직후·추출 전에 호출(per-asset 두 번째 스테이지, 도메인 팩 선택 분기점).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.classify import stage2_zeroshot, stage3_gemma
from src.classify.profiles import DomainProfileProvider, RegistryProvider
from src.classify.text_probe import extract_text_for_classification
from src.classify.types import (
    DOMAIN_GENERAL,
    POLICY_VERSION,
    ClassificationResult,
)
from src.classify import domains as _domains  # noqa: F401 — 기본 도메인 프로파일 등록

_HEAD_BYTES = 8192
_HIT_THRESHOLD = 2  # top 도메인 확정 최소 hit
_MARGIN = 1         # top 과 2위의 최소 격차


def _read_head(file_path: str, n: int = _HEAD_BYTES) -> bytes:
    try:
        with open(file_path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def _build_scan_text(file_path: str, modality: str) -> str:
    """Stage 2 어휘 스캔 입력: 파일명 + 결정적 추출 텍스트(LLM 없음)."""
    parts = [Path(file_path).name, extract_text_for_classification(file_path, modality)]
    return "\n".join(p for p in parts if p)


def classify(
    file_path: str,
    modality: str,
    *,
    provider: DomainProfileProvider | None = None,
    _llm_classify: Callable = stage3_gemma.classify,
) -> ClassificationResult:
    """프로파일 기반 도메인-불가지 3-stage cascade."""
    provider = provider or RegistryProvider()
    profiles = provider.all_profiles()

    # Stage 1 — 시그니처: 정확히 한 도메인만 매칭하면 확정, 0/충돌은 Stage 2 로.
    head = _read_head(file_path)
    s1_scores: dict[str, dict] = {}
    for p in profiles:
        for rule in p.signatures:
            hit = rule(head)
            if hit is not None:
                s1_scores[p.domain] = {"signature": hit.signature, **hit.detail}
                break
    if len(s1_scores) == 1:
        domain = next(iter(s1_scores))
        return ClassificationResult(
            final_label=domain, confidence=1.0, decided_stage=1, stage1_scores=s1_scores
        )

    # Stage 2 — 어휘: 도메인별 hit, top hit≥임계 & margin → 확정 / 0 → general / 그 외 → Stage 3.
    scan = _build_scan_text(file_path, modality)
    s2_scores: dict[str, dict] = {}
    for p in profiles:
        n, terms = stage2_zeroshot.count_hits(scan, p.lexicon)
        s2_scores[p.domain] = {"hits": n, "terms": terms}

    ranked = sorted(profiles, key=lambda p: s2_scores[p.domain]["hits"], reverse=True)
    top_hits = s2_scores[ranked[0].domain]["hits"] if ranked else 0
    second_hits = s2_scores[ranked[1].domain]["hits"] if len(ranked) > 1 else 0

    if top_hits == 0:
        return ClassificationResult(
            final_label=DOMAIN_GENERAL, confidence=0.7, decided_stage=2,
            stage1_scores=s1_scores, stage2_scores=s2_scores,
        )
    if top_hits >= _HIT_THRESHOLD and (top_hits - second_hits) >= _MARGIN:
        return ClassificationResult(
            final_label=ranked[0].domain, confidence=0.9, decided_stage=2,
            stage1_scores=s1_scores, stage2_scores=s2_scores,
        )

    # Stage 3 — 온프레미스 LLM(모호 케이스만), 라벨 = 등록 도메인 + general.
    labels = [p.domain for p in profiles] + [DOMAIN_GENERAL]
    s3_label, s3 = _llm_classify(scan, labels)
    conf = 0.6 if s3_label != "review" else 0.0
    return ClassificationResult(
        final_label=s3_label, confidence=conf, decided_stage=3,
        stage1_scores=s1_scores, stage2_scores=s2_scores, stage3_scores=s3,
        policy_version=POLICY_VERSION,
    )
