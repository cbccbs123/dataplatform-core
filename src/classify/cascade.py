"""F-5.1 의료 도메인 분류 — 3-stage cascade.

Stage 1(시그니처) → Stage 2(어휘 zero-shot) → Stage 3(온프레미스 LLM).
앞 단계가 확정하면 즉시 반환(단축). Stage 3 는 모호 케이스에만 호출(≤10% 게이트).
결과 ``ClassificationResult`` 는 asset_classification 적재 + asset.domain_label 갱신에 쓰인다.
라우팅 직후·추출 전에 호출한다.
"""

from __future__ import annotations

from pathlib import Path

from src.classify import stage1_magic, stage2_zeroshot, stage3_gemma
from src.classify.text_probe import extract_text_for_classification
from src.classify.types import (
    DOMAIN_MEDICAL,
    POLICY_VERSION,
    ClassificationResult,
)


def _build_scan_text(file_path: str, modality: str) -> str:
    """Stage 2 어휘 스캔 입력: 파일명 + 결정적 추출 텍스트(문서=텍스트레이어, 이미지=OCR, LLM 없음)."""
    parts = [Path(file_path).name, extract_text_for_classification(file_path, modality)]
    return "\n".join(p for p in parts if p)


def classify(file_path: str, modality: str) -> ClassificationResult:
    """3-stage cascade 로 도메인 분류."""
    # Stage 1 — 시그니처
    s1_label, s1 = stage1_magic.detect(file_path)
    if s1_label == DOMAIN_MEDICAL:
        return ClassificationResult(
            final_label=DOMAIN_MEDICAL, confidence=1.0, decided_stage=1, stage1_scores=s1
        )

    # Stage 2 — 어휘 zero-shot
    scan = _build_scan_text(file_path, modality)
    s2_label, s2 = stage2_zeroshot.score(scan)
    if s2_label is not None:
        conf = 0.9 if s2_label == DOMAIN_MEDICAL else 0.7
        return ClassificationResult(
            final_label=s2_label, confidence=conf, decided_stage=2,
            stage1_scores=s1, stage2_scores=s2,
        )

    # Stage 3 — 온프레미스 LLM(모호 케이스만)
    s3_label, s3 = stage3_gemma.classify(scan)
    conf = 0.6 if s3_label in (DOMAIN_MEDICAL, "general") else 0.0
    return ClassificationResult(
        final_label=s3_label, confidence=conf, decided_stage=3,
        stage1_scores=s1, stage2_scores=s2, stage3_scores=s3, policy_version=POLICY_VERSION,
    )
