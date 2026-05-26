"""F-5.1 Stage 1 — magic-byte/시그니처 기반 의료 표준 포맷 확정.

DICOM(@byte128 'DICM') / HL7 v2('MSH|') / FHIR(JSON resourceType)를 빠르게 판별한다.
매칭되면 의료로 확정(고신뢰), 아니면 None 으로 Stage 2 에 넘긴다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.classify.types import DOMAIN_MEDICAL

_HEAD_BYTES = 8192

# FHIR R4 주요 리소스 타입(부분 집합).
_FHIR_RESOURCES = frozenset(
    {
        "Patient", "Encounter", "Observation", "ImagingStudy", "DiagnosticReport",
        "Condition", "Procedure", "MedicationRequest", "Bundle", "DocumentReference",
        "ServiceRequest", "Specimen",
    }
)

_FHIR_RE = re.compile(r'"resourceType"\s*:\s*"([A-Za-z]+)"')


def detect(file_path: str) -> tuple[str | None, dict[str, Any]]:
    """시그니처로 의료 포맷 판별. (label, signal). label 은 'medical' 또는 None."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(_HEAD_BYTES)
    except OSError:
        return None, {}

    # DICOM: 128바이트 프리앰블 뒤 'DICM'
    if len(head) >= 132 and head[128:132] == b"DICM":
        return DOMAIN_MEDICAL, {"signature": "dicom"}

    text = head.decode("utf-8", errors="ignore").lstrip("﻿").lstrip()

    # HL7 v2: 'MSH|' 로 시작
    if text.startswith("MSH|"):
        return DOMAIN_MEDICAL, {"signature": "hl7v2"}

    # FHIR: resourceType 이 알려진 리소스
    if '"resourceType"' in text:
        m = _FHIR_RE.search(text)
        if m and m.group(1) in _FHIR_RESOURCES:
            return DOMAIN_MEDICAL, {"signature": "fhir", "resourceType": m.group(1)}

    return None, {}
