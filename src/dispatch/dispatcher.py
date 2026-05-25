"""F-3.4 디스패처 — 단순 if/elif 분기.

레지스트리·Protocol·데코레이터·훅 없이 ``modality`` 기준으로 적절한 추출 함수를 호출한다.
출력은 항상 ``AssetRecord`` 로 통일되어 영속화가 단일 경로로 처리한다.
새 modality 는 분기 한 줄 추가. 확장이 커지면 그때 레지스트리로 리팩터한다.

의료 도메인(``ctx.domain == 'medical'``)도 6월 단계에서는 modality 기준으로 동일 분기한다.
의료 표준 포맷(DICOM/HL7/FHIR) 전용 추출은 후속(F-5.2/5.3)에서 분기를 추가한다.
"""

from __future__ import annotations

from src.dispatch.types import AssetRecord, ExtractContext
from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind
from src.skills.audio_skill import extract_audio
from src.skills.image_skill import extract_image
from src.skills.text_skill import extract_text
from src.skills.video_skill import extract_video


class UnsupportedModalityError(ValueError):
    """디스패처가 처리할 수 없는 modality."""


def dispatch_extract(ctx: ExtractContext) -> AssetRecord:
    """``ctx.modality`` 에 맞는 추출 함수를 호출해 ``AssetRecord`` 를 반환한다."""
    modality = ctx.modality
    if modality in ALLOWED_TEXT_META_FILE_KINDS:  # txt/pdf/json/word/excel/powerpoint
        return extract_text(ctx)
    if modality == MediaKind.IMAGE.value:
        return extract_image(ctx)
    if modality == MediaKind.VIDEO.value:
        return extract_video(ctx)
    if modality == MediaKind.AUDIO.value:
        return extract_audio(ctx)
    raise UnsupportedModalityError(f"지원하지 않는 modality: {modality!r}")
