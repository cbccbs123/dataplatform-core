"""파일 모달리티·오피스 종류 enum 과 검색 능력별 media_type 집합의 중앙 정의.

``detect_file_kind`` 의 판정값(``MediaKind``/``OfficeKind``)과 ``data_loader`` 가 받는
``file_kind`` 허용 집합이 모두 여기 한곳에서 나온다(단일 출처). 검색측은 어떤 media_type 이
어떤 종류의 임베딩 청크를 갖는지를 아래 ``MEDIA_TYPES_*_CHUNK_SEARCH`` 로 참조한다.
"""

from __future__ import annotations

from enum import Enum


class MediaKind(str, Enum):
    IMAGE = "image"
    TEXT = "txt"
    VIDEO = "video"
    AUDIO = "audio"
    PDF = "pdf"
    JSON = "json"
    UNKNOWN = "unknown"


class OfficeKind(str, Enum):
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

# ``media_chunks`` 에 ``embedding_kind=st`` 청크가 있는 media_type(문서·오디오 STT·영상 VLM 텍스트 등).
# 벡터 검색과 별도로 풀텍스트는 ``media_items.search_vector``(메타 기반)에서 처리한다.
MEDIA_TYPES_ST_CHUNK_SEARCH = frozenset(
    {*ALLOWED_TEXT_META_FILE_KINDS, MediaKind.AUDIO.value, MediaKind.VIDEO.value}
)

# ``media_chunks`` 의 CLIP(``embedding_kind=clip``)으로 시각·키프레임 검색에 쓸 media_type
MEDIA_TYPES_CLIP_CHUNK_SEARCH = frozenset(
    {MediaKind.IMAGE.value, MediaKind.VIDEO.value}
)


__all__ = [
    "MediaKind",
    "OfficeKind",
    "OFFICE_FILE_KINDS",
    "TEXT_FILE_KINDS",
    "ALLOWED_TEXT_META_FILE_KINDS",
    "MEDIA_TYPES_ST_CHUNK_SEARCH",
    "MEDIA_TYPES_CLIP_CHUNK_SEARCH",
]
