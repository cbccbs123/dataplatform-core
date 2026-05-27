"""의료 도메인 프로파일 — 시그니처 능력(DICOM/HL7/FHIR) + 어휘 + 등록.

(구 stage1_magic 의 시그니처 판별을 능력 모듈로 이관. 어휘는 medical_terms 에서 재사용.)
"""
from __future__ import annotations

import re

from src.classify.medical_terms import MEDICAL_TERMS
from src.classify.profiles import DomainProfile, SigHit, register_profile
from src.classify.types import DOMAIN_MEDICAL

# FHIR R4 주요 리소스 타입(부분 집합).
_FHIR_RESOURCES = frozenset(
    {
        "Patient", "Encounter", "Observation", "ImagingStudy", "DiagnosticReport",
        "Condition", "Procedure", "MedicationRequest", "Bundle", "DocumentReference",
        "ServiceRequest", "Specimen",
    }
)
_FHIR_RE = re.compile(r'"resourceType"\s*:\s*"([A-Za-z]+)"')


def _decode(head: bytes) -> str:
    return head.decode("utf-8", errors="ignore").lstrip("﻿").lstrip()


def _dicom(head: bytes) -> SigHit | None:
    """DICOM: 128바이트 프리앰블 뒤 'DICM'."""
    if len(head) >= 132 and head[128:132] == b"DICM":
        return SigHit("dicom")
    return None


def _hl7v2(head: bytes) -> SigHit | None:
    """HL7 v2: 'MSH|' 로 시작."""
    if _decode(head).startswith("MSH|"):
        return SigHit("hl7v2")
    return None


def _fhir(head: bytes) -> SigHit | None:
    """FHIR: resourceType 이 알려진 리소스."""
    text = _decode(head)
    if '"resourceType"' in text:
        m = _FHIR_RE.search(text)
        if m and m.group(1) in _FHIR_RESOURCES:
            return SigHit("fhir", {"resourceType": m.group(1)})
    return None


MEDICAL_PROFILE = DomainProfile(
    domain=DOMAIN_MEDICAL,
    lexicon=MEDICAL_TERMS,
    llm_label=DOMAIN_MEDICAL,
    signatures=(_dicom, _hl7v2, _fhir),
)
register_profile(MEDICAL_PROFILE)
