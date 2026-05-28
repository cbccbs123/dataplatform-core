'''
이 이미지에 있는 모든 객체를 빠짐없이 나열해줘.
일반 명사 형태로 리스트만 반환해.
중복 제거해서 출력해.
'''

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TypedDict

from PIL import Image

from src.config.settings import get_current_settings
from src.llm.client import complete_vision_json


class ImageSummaryResult(TypedDict):
    """이미지 요약 문장(summary), 키워드, 객체 목록."""

    summary: str
    keywords: list[str]
    objects: list[str]


def _encode_image_as_jpeg_data_url(
    image_path: Path,
    *,
    max_side: int = 1024,
    jpeg_quality: int = 85,
) -> str:
    """비전 API 호환을 위해 RGB JPEG로 인코딩한 data URL을 반환한다."""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _encode_jpeg_bytes_as_data_url_resized(
    jpeg_bytes: bytes,
    *,
    max_side: int = 1024,
    jpeg_quality: int = 85,
) -> str:
    """OpenCV 등에서 온 고해상도 JPEG도 비전 API용으로 max_side 기준으로 축소한다."""
    with Image.open(BytesIO(jpeg_bytes)) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def summarize_image_caption_keywords_objects(
    file_path: str | Path,
    *,
    max_side: int = 1024,
    jpeg_quality: int = 85,
) -> ImageSummaryResult:
    """
    이미지 한 장에 대해 캡션, 키워드, 객체 목록을 LLM으로 추출한다.

    설정은 ``get_current_settings()``(``init_settings`` 이후)를 사용한다.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    image_url = _encode_image_as_jpeg_data_url(path, max_side=max_side, jpeg_quality=jpeg_quality)
    return _summarize_image_caption_keywords_objects_from_data_url(image_data_url=image_url)


def summarize_image_caption_keywords_objects_from_jpeg_bytes(
    jpeg_bytes: bytes,
    *,
    max_side: int = 1024,
    jpeg_quality: int = 85,
) -> ImageSummaryResult:
    """메모리의 JPEG bytes를 바로 받아 요약/키워드/객체를 추출한다. 파일 경로 API와 동일하게 리사이즈한다."""
    image_url = _encode_jpeg_bytes_as_data_url_resized(
        jpeg_bytes, max_side=max_side, jpeg_quality=jpeg_quality
    )
    return _summarize_image_caption_keywords_objects_from_data_url(image_data_url=image_url)


def _summarize_image_caption_keywords_objects_from_data_url(
    *,
    image_data_url: str,
) -> ImageSummaryResult:
    cfg = get_current_settings()

    prompt = (
        "이 이미지를 분석해서 반드시 JSON만 출력해.\n"
        "형식:\n"
        '{ "summary": "한국어로 이미지 전체를 설명하는 문장", '
        '"keywords": ["키워드1", "키워드2"], '
        '"objects": ["객체1", "객체2"] }\n'
        f"- summary은 {cfg.summary_max_chars}자 이내\n"
        f"- keywords는 핵심 키워드 최대 {cfg.top_k_keywords}개 (한국어)\n"
        "- objects는 이미지에 보이는 모든 주요 객체를 일반 명사 형태의 한국어로 나열 (중복 제거)\n"
        "개수/비율/합계 같은 통계 표현은 summary/keywords에는 쓰지 말 것."
    )

    data = complete_vision_json(text=prompt, image_data_url=image_data_url)

    summary = str(data.get("summary", "")).strip()[: cfg.summary_max_chars]

    keywords_raw = data.get("keywords", [])
    if not isinstance(keywords_raw, list):
        keywords_raw = []
    keywords: list[str] = []
    for kw in keywords_raw:
        k = str(kw).strip()
        if k and k not in keywords:
            keywords.append(k)
        if len(keywords) >= cfg.top_k_keywords:
            break

    objects_raw = data.get("objects", [])
    if not isinstance(objects_raw, list):
        objects_raw = []
    objects: list[str] = []
    for obj in objects_raw:
        o = str(obj).strip()
        if o and o not in objects:
            objects.append(o)

    return {
        "summary": summary,
        "keywords": keywords,
        "objects": objects,
    }