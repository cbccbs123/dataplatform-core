"""파일 모달리티·오피스 종류 enum 과 검색 능력별 media_type 집합의 중앙 정의.

``detect_file_kind`` 의 판정값(``MediaKind``/``OfficeKind``)과 ``data_loader`` 가 받는
``file_kind`` 허용 집합이 모두 여기 한곳에서 나온다(단일 출처). 검색측은 어떤 media_type 이
어떤 종류의 임베딩 청크를 갖는지를 아래 ``MEDIA_TYPES_*_CHUNK_SEARCH`` 로 참조한다.
"""

from __future__ import annotations

from enum import StrEnum


class MediaKind(StrEnum):
    IMAGE = "image"
    TEXT = "txt"
    VIDEO = "video"
    AUDIO = "audio"
    PDF = "pdf"
    JSON = "json"
    UNKNOWN = "unknown"


class OfficeKind(StrEnum):
    WORD = "word"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    UNKNOWN = "unknown"


OFFICE_FILE_KINDS = {k.value for k in OfficeKind if k is not OfficeKind.UNKNOWN}
TEXT_FILE_KINDS = {MediaKind.TEXT.value, MediaKind.JSON.value}
ALLOWED_TEXT_META_FILE_KINDS = {
    *OFFICE_FILE_KINDS,
    *{k.value for k in MediaKind if k in {MediaKind.TEXT, MediaKind.PDF, MediaKind.JSON}},
}

# ``asset_embedding.channel=st`` 임베딩이 있는 media_type(문서·오디오 STT·영상 VLM 텍스트 등).
# 벡터 검색과 별도로 풀텍스트(BM25)는 OpenSearch 색인(메타 기반)에서 처리한다(037 OS 전용; PG FTS 폐기).
MEDIA_TYPES_ST_CHUNK_SEARCH = frozenset(
    {*ALLOWED_TEXT_META_FILE_KINDS, MediaKind.AUDIO.value, MediaKind.VIDEO.value}
)


# asset.modality 저장 허용값(canonical). v292 CHECK 와 일치. unknown 은 모달리티가 아닌 격리표식.
CANONICAL_MODALITIES = ("text", "image", "video", "audio", "unknown")


def modality_of(file_kind: str) -> str:
    """세부 파일 종류를 저장용 **큰 갈래**로 좁힌다(순수 함수).

    ⚠️ **저장 경계에서만 쓴다.** 추출·청크 분할은 세부 종류를 그대로 봐야 알맞은 파서를 고를 수
    있다. 저장값만 좁히는 이유는 집계 축이 무한정 늘어나는 것을 막기 위해서고, 세부 종류가
    필요하면 경로의 확장자에서 다시 구할 수 있다.

    Args:
        file_kind: 파일 종류 판정값.

    Returns:
        큰 갈래 이름. 지원 밖·판별 불가는 **모달리티가 아닌 격리 표식**을 돌려준다 —
        "모른다"를 텍스트 같은 실제 갈래에 섞어 넣지 않기 위해서다.
    """
    if file_kind in ALLOWED_TEXT_META_FILE_KINDS:      # txt,pdf,json,word,excel,powerpoint
        return "text"
    if file_kind in (MediaKind.IMAGE.value, MediaKind.VIDEO.value, MediaKind.AUDIO.value):
        return file_kind
    return MediaKind.UNKNOWN.value                      # 'unknown' (미지원·판별불가 격리)


__all__ = [
    "MediaKind",
    "OfficeKind",
    "OFFICE_FILE_KINDS",
    "TEXT_FILE_KINDS",
    "ALLOWED_TEXT_META_FILE_KINDS",
    "MEDIA_TYPES_ST_CHUNK_SEARCH",
    "CANONICAL_MODALITIES",
    "modality_of",
]
