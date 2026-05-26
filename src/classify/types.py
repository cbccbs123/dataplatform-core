"""F-5.1 도메인 분류 결과 데이터클래스."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# asset.domain_label / asset_classification.final_label 과 동일 값.
DOMAIN_MEDICAL = "medical"
DOMAIN_GENERAL = "general"
DOMAIN_REVIEW = "review"

POLICY_VERSION = "classify.v1"


@dataclass
class ClassificationResult:
    """3-stage cascade 분류 결과. asset_classification 한 행에 대응."""

    final_label: str  # medical | general | review
    confidence: float
    decided_stage: int  # 1 | 2 | 3
    stage1_scores: dict[str, Any] = field(default_factory=dict)
    stage2_scores: dict[str, Any] = field(default_factory=dict)
    stage3_scores: dict[str, Any] = field(default_factory=dict)
    policy_version: str = POLICY_VERSION
