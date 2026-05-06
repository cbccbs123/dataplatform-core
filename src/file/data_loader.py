"""
로컬 Ollama(gemma)로 텍스트·오피스·PDF·JSON 파일 요약/키워드 추출.

- 일반 텍스트: ``.txt`` (``office_kind`` 생략)
- PDF: ``.pdf`` (``office_kind`` 생략, ``pypdf``)
- JSON: ``.json`` (``office_kind`` 생략, 파싱 후 들여쓴 문자열로 모델에 전달)
- 오피스: ``office_kind`` 에 ``word`` / ``excel`` / ``powerpoint`` 지정 후 각각 ``.docx`` / ``.xlsx`` / ``.pptx``
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, TypedDict

from src.file.file_type_defs import (
    ALLOWED_TEXT_META_FILE_KINDS,
    OFFICE_FILE_KINDS,
    TEXT_FILE_KINDS,
    MediaKind,
    OfficeKind,
)


class SummaryKeywords(TypedDict):
    summary: str
    keywords: list[str]
    language: str
    encoding: str
    num_sentences: int
    num_tokens: int
    length: int


_OFFICE_EXTENSIONS: dict[str, frozenset[str]] = {
    OfficeKind.WORD.value: frozenset({".docx"}),
    OfficeKind.EXCEL.value: frozenset({".xlsx"}), 
    OfficeKind.POWERPOINT.value: frozenset({".pptx"}),
}
_MAX_INPUT_CHARS = 20_000_000
_OFFICE_FILE_KINDS = OFFICE_FILE_KINDS
_TEXT_FILE_KINDS = TEXT_FILE_KINDS
_ALLOWED_FILE_KINDS = ALLOWED_TEXT_META_FILE_KINDS


def _normalize_file_kind(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s not in _ALLOWED_FILE_KINDS:
        raise ValueError(f"file_kind는 {sorted(_ALLOWED_FILE_KINDS)} 중 하나여야 합니다.")
    return s


def _choose_encoding(path: Path, preferred_encoding: str) -> str:
    # 파일 스트리밍 읽기를 위해 샘플 바이트로 인코딩을 추정한다.
    sample = path.read_bytes()[:65536]
    for enc in (preferred_encoding, "utf-8-sig", "cp949", "euc-kr"):
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _chunk_from_segments(segments: Iterable[str], chunk_size: int) -> Iterator[str]:
    buf = ""
    for seg in segments:
        if not seg:
            continue
        s = seg.strip()
        if not s:
            continue
        if len(s) > chunk_size:
            if buf:
                yield buf
                buf = ""
            for i in range(0, len(s), chunk_size):
                part = s[i : i + chunk_size]
                if part:
                    yield part
            continue
        if not buf:
            buf = s
            continue
        if len(buf) + 1 + len(s) <= chunk_size:
            buf += "\n" + s
        else:
            yield buf
            buf = s
    if buf:
        yield buf


def _limit_segments(segments: Iterable[str], max_chars: int) -> Iterator[str]:
    """세그먼트 스트림을 최대 max_chars 문자까지만 통과시킨다."""
    if max_chars <= 0:
        return
    remaining = max_chars
    for seg in segments:
        if remaining <= 0:
            break
        if not seg:
            continue
        if len(seg) <= remaining:
            yield seg
            remaining -= len(seg)
        else:
            yield seg[:remaining]
            break


def _apply_overlap(chunks: Iterable[str], overlap_size: int) -> Iterator[str]:
    """이미 생성된 청크들에 문자 단위 오버랩을 부여한다."""
    prev: str | None = None
    for chunk in chunks:
        if prev is None:
            prev = chunk
            yield chunk
            continue
        if overlap_size <= 0:
            yield chunk
        else:
            prefix = prev[-overlap_size:] if len(prev) > overlap_size else prev
            yield (prefix + "\n" + chunk).strip()
        prev = chunk


def _iter_document_chunks(
    path: Path,
    *,
    file_kind: str | None,
    encoding: str,
    chunk_size: int,
    overlap_size: int,
    max_input_chars: int,
) -> Iterator[str]:
    if overlap_size < 0:
        raise ValueError("overlap_size는 0 이상이어야 합니다.")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size는 chunk_size보다 작아야 합니다.")

    ext = path.suffix.lower()
    if file_kind is None:
        raise ValueError("file_kind는 필수입니다. txt/pdf/json/word/excel/powerpoint 중 하나를 전달하세요.")

    if file_kind in _OFFICE_FILE_KINDS:
        allowed = _OFFICE_EXTENSIONS[file_kind]
        if ext not in allowed:
            raise ValueError(
                f"file_kind={file_kind!r} 일 때 허용 확장자는 {sorted(allowed)} 입니다. 현재: {ext!r}"
            )
        if file_kind == OfficeKind.WORD.value:
            from docx import Document  # type: ignore

            doc = Document(str(path))
            lines = (p.text for p in doc.paragraphs)
            limited = _limit_segments(lines, max_input_chars)
            yield from _apply_overlap(_chunk_from_segments(limited, chunk_size), overlap_size)
            return
        if file_kind == OfficeKind.EXCEL.value:
            from openpyxl import load_workbook  # type: ignore

            wb = load_workbook(filename=str(path), read_only=True, data_only=True)
            try:
                def _rows() -> Iterator[str]:
                    for sheet in wb.worksheets:
                        yield f"[시트] {sheet.title}"
                        for row in sheet.iter_rows(values_only=True):
                            cells = [str(v).strip() for v in row if v is not None and str(v).strip()]
                            if cells:
                                yield " | ".join(cells)

                limited = _limit_segments(_rows(), max_input_chars)
                yield from _apply_overlap(_chunk_from_segments(limited, chunk_size), overlap_size)
            finally:
                wb.close()
            return
        if file_kind == OfficeKind.POWERPOINT.value:
            from pptx import Presentation  # type: ignore

            prs = Presentation(str(path))

            def _slides() -> Iterator[str]:
                for i, slide in enumerate(prs.slides, start=1):
                    yield f"[슬라이드] {i}"
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text:
                            yield str(shape.text)

            limited = _limit_segments(_slides(), max_input_chars)
            yield from _apply_overlap(_chunk_from_segments(limited, chunk_size), overlap_size)
            return
        raise ValueError("지원하지 않는 오피스 file_kind 입니다.")

    if file_kind in _TEXT_FILE_KINDS:
        if file_kind == MediaKind.TEXT.value and ext != ".txt":
            raise ValueError(f"file_kind='{MediaKind.TEXT.value}' 일 때는 .txt 파일이어야 합니다.")
        if file_kind == MediaKind.JSON.value and ext != ".json":
            raise ValueError(f"file_kind='{MediaKind.JSON.value}' 일 때는 .json 파일이어야 합니다.")
        enc = _choose_encoding(path, encoding)
        with path.open("r", encoding=enc, errors="replace") as f:
            def _parts() -> Iterator[str]:
                while True:
                    part = f.read(4096)
                    if not part:
                        break
                    yield part

            limited = _limit_segments(_parts(), max_input_chars)
            yield from _apply_overlap(_chunk_from_segments(limited, chunk_size), overlap_size)
        return

    if file_kind == MediaKind.PDF.value:
        if ext != ".pdf":
            raise ValueError(f"file_kind='{MediaKind.PDF.value}' 일 때는 .pdf 파일이어야 합니다.")
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages = ((page.extract_text() or "") for page in reader.pages)
        limited = _limit_segments(pages, max_input_chars)
        yield from _apply_overlap(_chunk_from_segments(limited, chunk_size), overlap_size)
        return

    raise ValueError(
        "지원 확장자: .txt, .pdf, .json 또는 오피스(.docx/.xlsx/.pptx + file_kind=word|excel|powerpoint)."
    )


def iter_plain_text_chunks(
    text: str,
    *,
    chunk_size: int,
    overlap_size: int = 0,
    max_input_chars: int = _MAX_INPUT_CHARS,
) -> Iterator[str]:
    """
    파일 없이 문자열만 받아 ``_chunk_from_segments`` 규칙으로 나눈다.

    STT 전체 텍스트 등을 ``iter_document_chunks``와 같은 크기·오버랩 정책으로
    잘라 임베딩할 때 사용한다.
    """
    if overlap_size < 0:
        raise ValueError("overlap_size는 0 이상이어야 합니다.")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size는 chunk_size보다 작아야 합니다.")
    if max_input_chars <= 0:
        return
    if text is None or not str(text).strip():
        return
    limited = _limit_segments(iter([str(text)]), max_input_chars)
    yield from _apply_overlap(_chunk_from_segments(limited, chunk_size), overlap_size)


# Public API (recommended)
normalize_file_kind = _normalize_file_kind
choose_encoding = _choose_encoding
iter_document_chunks = _iter_document_chunks
MAX_INPUT_CHARS = _MAX_INPUT_CHARS

# Backward-compatible private aliases
_normalize_file_kind = normalize_file_kind
_choose_encoding = choose_encoding
_iter_document_chunks = iter_document_chunks
_MAX_INPUT_CHARS = MAX_INPUT_CHARS

__all__ = [
    "MAX_INPUT_CHARS",
    "choose_encoding",
    "iter_document_chunks",
    "iter_plain_text_chunks",
    "normalize_file_kind",
]

