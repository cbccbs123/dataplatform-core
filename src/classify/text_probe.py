"""F-5.1 분류용 결정적 텍스트 추출 (LLM 없음).

분류 단계는 무거운 LLM(캡션/요약) 없이, 어휘 판정에 쓸 텍스트만 싸게 뽑는다.
- txt/json: 평문 읽기
- pdf/office(word/excel/powerpoint): 텍스트 레이어(`data_loader`)
- image: OCR(Tesseract, kor+eng)
- video/audio/unknown: 분류 단계 텍스트 없음(빈 문자열) → 시그니처/경로/stage3 가 처리
"""

from __future__ import annotations

from pathlib import Path

from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind

_MAX_CHARS = 4000  # stage2 어휘 스캔용 — stage3 LLM 입력(_MAX_TEXT=4000)과 별도 관리
_PLAIN_KINDS = frozenset({MediaKind.TEXT.value, MediaKind.JSON.value})  # txt, json


def extract_text_for_classification(file_path: str, modality: str, *, max_chars: int = _MAX_CHARS) -> str:
    """분류용 텍스트(LLM 없음). 실패는 빈 문자열로 흡수.

    video/audio/unknown 은 빈 문자열을 반환 — stage1 시그니처·파일명·stage3 LLM 이 처리.
    분류 단계에서 LLM 캡션/요약을 쓰지 않는 이유: 비용·지연 없이 결정론적 판정 우선.
    """
    if modality in ALLOWED_TEXT_META_FILE_KINDS:
        return _document_text(file_path, modality, max_chars)
    if modality == MediaKind.IMAGE.value:
        return _ocr_image(file_path, max_chars)
    return ""


def _document_text(file_path: str, modality: str, max_chars: int) -> str:
    if modality in _PLAIN_KINDS:
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                return f.read(max_chars)
        except OSError:
            return ""
    # pdf / office → 텍스트 레이어(data_loader 지연 임포트: 무거운 패키지를 분류 불필요 경로에서 로드 방지)
    from src.file.data_loader import iter_document_chunks, normalize_file_kind

    kind = normalize_file_kind(modality)
    if kind is None:
        return ""
    try:
        out: list[str] = []
        total = 0
        for ch in iter_document_chunks(
            Path(file_path), file_kind=kind, encoding="utf-8",
            chunk_size=max_chars, overlap_size=0, max_input_chars=max_chars,
        ):
            out.append(ch)
            total += len(ch)
            if total >= max_chars:
                break
        return "".join(out)[:max_chars]
    except Exception:  # noqa: BLE001 — 분류용이므로 실패는 빈 텍스트로 흡수
        return ""


def _ocr_image(file_path: str, max_chars: int) -> str:
    try:
        import pytesseract
        from PIL import Image

        with Image.open(file_path) as img:
            text = pytesseract.image_to_string(img, lang="kor+eng")
        return (text or "").strip()[:max_chars]
    except Exception:  # noqa: BLE001 — OCR 실패는 빈 텍스트
        return ""
