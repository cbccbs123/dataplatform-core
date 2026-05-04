"""
파일이 이미지 / 텍스트 / 영상 / 음성 / PDF / JSON 인지, 오피스면 Word·Excel·PPT 인지 판별한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import magic
from src.file.file_type_defs import MediaKind, OfficeKind

_READ_CHUNK = 262_144

_WORD_MIMES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
        "application/vnd.ms-word.document.macroenabled.12",
        "application/vnd.ms-word.template.macroenabled.12",
        "application/msword",
        "application/vnd.ms-word",
    }
)
_EXCEL_MIMES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.template",
        "application/vnd.ms-excel.sheet.macroenabled.12",
        "application/vnd.ms-excel.template.macroenabled.12",
        "application/vnd.ms-excel.sheet.binary.macroenabled.12",
        "application/vnd.ms-excel",
        "application/excel",
    }
)
_PPT_MIMES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.presentationml.template",
        "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
        "application/vnd.ms-powerpoint.presentation.macroenabled.12",
        "application/vnd.ms-powerpoint.slideshow.macroenabled.12",
        "application/vnd.ms-powerpoint",
        "application/vnd.ms-powerpoint.presentation",
    }
)


def _read_head(path: Path, max_bytes: int = _READ_CHUNK) -> bytes:
    with path.open("rb") as f:
        return f.read(max_bytes)


def _magic_mime(path: Path) -> str:
    mime = magic.from_file(str(path), mime=True)
    if isinstance(mime, bytes):
        mime = mime.decode("utf-8", errors="replace")
    return str(mime).lower().strip()


def _kind_from_mime(mime: str) -> Optional[MediaKind]:
    if not mime or mime == "application/octet-stream":
        return None
    if mime == "application/pdf":
        return MediaKind.PDF
    if mime in {"application/json", "application/x-ndjson", "application/ld+json", "application/problem+json"}:
        return MediaKind.JSON
    if mime.startswith("image/"):
        return MediaKind.IMAGE
    if mime.startswith("video/"):
        return MediaKind.VIDEO
    if mime.startswith("audio/"):
        return MediaKind.AUDIO
    if mime.startswith("text/"):
        return MediaKind.TEXT
    if mime in {"application/xml", "application/xhtml+xml", "application/javascript", "application/x-sh"}:
        return MediaKind.TEXT
    return None


def detect_file_kind(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    mime = _magic_mime(path)
    if mime in _WORD_MIMES:
        return OfficeKind.WORD.value
    if mime in _EXCEL_MIMES:
        return OfficeKind.EXCEL.value
    if mime in _PPT_MIMES:
        return OfficeKind.POWERPOINT.value

    media = _kind_from_mime(mime)
    if media is not None:
        return media.value

    return MediaKind.UNKNOWN.value


__all__ = ["MediaKind", "OfficeKind", "detect_file_kind"]
