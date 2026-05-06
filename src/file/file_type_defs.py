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

# ``media_chunks``에 SentenceTransformer 청크가 있어 텍스트 쿼리 검색에 포함할 media_type
MEDIA_TYPES_ST_CHUNK_SEARCH = frozenset(
    {*ALLOWED_TEXT_META_FILE_KINDS, MediaKind.AUDIO.value, MediaKind.VIDEO.value}
)

# ``media_chunks``의 CLIP(``embedding_kind=clip``)으로 시각·키프레임 검색에 쓸 media_type
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
