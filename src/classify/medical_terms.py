"""의료 도메인 어휘 사전(zero-shot 보조용). 학습 없이 규칙 매칭에 사용.

영문은 소문자로, 한글은 그대로 저장(매칭 시 텍스트를 lower() 한 뒤 부분 문자열 검사).
"""

from __future__ import annotations

MEDICAL_TERMS: frozenset[str] = frozenset(
    {
        # 한국어
        "환자", "진단", "진료", "처방", "병원", "의원", "내원", "외래", "입원", "퇴원",
        "수술", "마취", "소견", "판독", "검사결과", "혈액검사", "영상의학", "방사선",
        "초음파", "골절", "종양", "악성", "양성", "전이", "병변", "투약", "항생제",
        "진단서", "의무기록", "병력", "증상", "처치", "재활", "물리치료", "주치의",
        "간호", "응급", "중환자", "조직검사", "생검", "심전도", "혈압", "당뇨", "고혈압",
        # 의료 표준/모달리티 약어
        "dicom", "hl7", "fhir", "pacs", "emr", "ehr", "ct", "mri", "x-ray", "xray",
        "ultrasound", "radiology", "pathology", "modality", "accession", "study uid",
        # 영문 임상 용어
        "patient", "diagnosis", "diagnostic", "clinical", "hospital", "prescription",
        "surgery", "anesthesia", "biopsy", "tumor", "malignant", "lesion", "symptom",
        "treatment", "medication", "antibiotic", "icd-10", "icd10",
    }
)
