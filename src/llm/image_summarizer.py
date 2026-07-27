"""이미지 1장 → LLM 비전 요약(summary·keywords·objects).

per-asset 파이프라인의 extract 단계에서 ``image_skill`` 이, 영상 키프레임 처리에서 ``video_skill`` 이
호출한다. LLM 호출은 전부 단일 seam ``src.llm.client.complete_vision_json`` 경유(temperature=0·온프레미스).

산출 ``summary`` 는 임베딩 입력(vlm_text_for_embedding)·BM25(search_text)의 **단일 원천**이라,
프롬프트에서 매체 문형('~이미지입니다')을 금지하고 내용·주제 중심으로 강제한다(spec 026 FR-001 토픽화).
``objects`` 는 CLIP 제로샷 라벨 후보로만 쓰이고 최종 메타에서는 제거된다(→ ``image_skill`` 참고).
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TypedDict

from PIL import Image

from src.config.settings import get_current_settings
from src.llm.client import complete_vision_json
from src.llm.summary_postprocess import promote_objects_to_keywords


class ImageSummaryResult(TypedDict):
    """이미지 요약 문장(summary), 키워드, 객체 목록."""

    summary: str
    keywords: list[str]
    objects: list[str]


# 요약 토픽화 지시(spec 026 FR-001). summary 가 **매체 자체**가 아니라 **담긴 내용·주제**를 쓰게 강제한다.
# summary 는 임베딩 입력(vlm_text_for_embedding)·BM25(search_text)의 단일 원천이라, "~이미지입니다/
# 썸네일" 같은 매체 문형이 들어가면 토픽이 아닌 '매체 포맷'으로 군집된다(F1 — '스마트폰' 질의에 수영·
# 자전거 영상이 상위로 끼는 실증 원인). 산출 JSON 키·파싱은 불변이고, **이 지시 문구만** 추가한다.
_SUMMARY_TOPIC_INSTRUCTION = (
    "- summary 는 매체 자체를 언급하지 말 것 — '이미지입니다/사진입니다/영상입니다/썸네일' 같은 "
    "매체 단어·문형 금지\n"
    "- 담긴 내용·주제·개체를 명사구 중심으로 서술\n"
)


# 049 FR-201/202: v2 캡션 추가 지시(검색지향). v1 토픽화 지시 위에 **덧붙이기만** 하므로 v1 경로는
# 불변(FR-102 바이트 동일·회귀 안전판). 화면의 구체 개체·고유명사·화면 텍스트·인물 역할·배경을 명시
# 지시하고, keywords 는 검색에 쓸 구체 명사 위주로 유도한다('영상/장면/이미지' 같은 일반어 금지).
# 키워드 정규화·objects 승격은 summarize_* 의 후처리(summary_postprocess)가 v2 일 때만 추가로 한다.
_V2_CAPTION_EXTRA_INSTRUCTION = (
    "- 화면에 보이는 구체 개체·고유명사·화면 텍스트(슬라이드 제목·자막)·인물 역할·배경 장소를 명시\n"
    "- keywords 는 검색에 쓰일 구체 명사(제품·주제·장소·행위) 위주 — "
    "'영상'·'장면'·'이미지' 같은 일반어 금지\n"
)


def _build_image_caption_prompt(
    *, summary_max_chars: int, top_k_keywords: int, v2: bool = False
) -> str:
    """이미지 캡션·키워드·객체 추출용 비전 프롬프트를 만든다(순수 — LLM·settings 미접촉).

    프롬프트를 함수 안 f-string 에 묻어 두지 않고 **따로 빼 둔 이유**: 지시문이 바뀌면 요약
    결과가 통째로 달라지므로, 문구 자체를 단위 테스트로 못박아 둘 수 있어야 한다.

    Args:
        summary_max_chars: 요약 길이 상한(글자).
        top_k_keywords: 키워드 개수 상한.
        v2: 검색 지향 추가 지시를 끼워 넣을지. **끄면 기존 문구가 글자 그대로** 나온다 —
            문구를 바꿨을 때 되돌릴 수 있는 안전판이다.

    Returns:
        비전 모델에 그대로 넣을 프롬프트.
    """
    v2_extra = _V2_CAPTION_EXTRA_INSTRUCTION if v2 else ""
    return (
        "이 이미지를 분석해서 반드시 JSON만 출력해.\n"
        "형식:\n"
        '{ "summary": "한국어로 담긴 내용·주제를 서술한 문장", '
        '"keywords": ["키워드1", "키워드2"], '
        '"objects": ["객체1", "객체2"] }\n'
        f"- summary은 {summary_max_chars}자 이내\n"
        f"- keywords는 핵심 키워드 최대 {top_k_keywords}개 (한국어)\n"
        "- objects는 이미지에 보이는 모든 주요 객체를 일반 명사 형태의 한국어로 나열 (중복 제거)\n"
        + _SUMMARY_TOPIC_INSTRUCTION
        + v2_extra
        + "개수/비율/합계 같은 통계 표현은 summary/keywords에는 쓰지 말 것."
    )


def _encode_image_as_jpeg_data_url(
    image_path: Path,
    *,
    max_side: int = 1024,
    jpeg_quality: int = 85,
) -> str:
    """비전 API 호환을 위해 RGB JPEG로 인코딩한 data URL을 반환한다.

    ⚠️ **RGB 로 강제 변환한다** — 투명도가 있는 이미지는 JPEG 로 바로 저장할 수 없다.

    Args:
        image_path: 원본 이미지 경로.
        max_side: 긴 변의 최대 픽셀. **비율은 유지**하며 이미 작으면 아무 일도 하지 않는다.
            줄이는 이유는 고해상도 원본을 그대로 보내면 요청이 지나치게 커지기 때문이다.
        jpeg_quality: JPEG 품질.

    Returns:
        ``data:image/jpeg;base64,...`` 문자열.
    """
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
    """이미 JPEG 바이트로 들고 있는 이미지를 축소해 data URL 로 만든다.

    영상 키프레임처럼 **파일로 떨어뜨리지 않은** 이미지를 위한 경로다.

    Args:
        jpeg_bytes: 원본 JPEG 바이트.
        max_side: 긴 변의 최대 픽셀(비율 유지·이미 작으면 무동작).
        jpeg_quality: 재인코딩 품질.

    Returns:
        ``data:image/jpeg;base64,...`` 문자열.
    """
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
    """이미지 한 장에서 캡션·키워드·객체 목록을 뽑는다(LLM 호출).

    Args:
        file_path: 이미지 경로.
        max_side: 보낼 이미지의 긴 변 최대 픽셀(요청 크기 억제).
        jpeg_quality: 재인코딩 품질.

    Returns:
        요약 결과.

    Raises:
        FileNotFoundError: 파일이 없을 때.
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
    """메모리의 JPEG 바이트에서 바로 요약·키워드·객체를 뽑는다(파일 경로 판과 동일 처리).

    Args:
        jpeg_bytes: 원본 JPEG 바이트.
        max_side: 보낼 이미지의 긴 변 최대 픽셀.
        jpeg_quality: 재인코딩 품질.

    Returns:
        요약 결과.
    """
    image_url = _encode_jpeg_bytes_as_data_url_resized(
        jpeg_bytes, max_side=max_side, jpeg_quality=jpeg_quality
    )
    return _summarize_image_caption_keywords_objects_from_data_url(image_data_url=image_url)


def _summarize_image_caption_keywords_objects_from_data_url(
    *,
    image_data_url: str,
) -> ImageSummaryResult:
    """data URL 형태의 이미지에서 캡션·키워드·객체를 LLM 비전으로 추출한다.

    공개 API 인 ``summarize_image_caption_keywords_objects`` 와
    ``summarize_image_caption_keywords_objects_from_jpeg_bytes`` 가
    인코딩 방식만 다를 뿐 이 함수로 수렴한다 — 비전 프롬프트와 후처리 로직이 한 곳에만 존재한다.
    ``objects`` 는 CLIP 제로샷 라벨 후보로 쓰인 후 최종 메타에서는 제거된다(→ image_skill 참고).
    """
    cfg = get_current_settings()

    # 049: 토글(vlm_summary_prompt_v2)을 빌더에 v2= 로 전달한다. False(기본)면 v1 프롬프트·v1 키워드
    # 루프가 그대로 돌아 출력이 현행과 바이트 동일하다(FR-102 회귀 안전판). 빌더 자체는 settings 를
    # 모르는 순수 함수라, 토글 해소는 여기(summarize 함수)에서만 한다(plan P2).
    v2 = cfg.vlm.summary_prompt_v2
    prompt = _build_image_caption_prompt(
        summary_max_chars=cfg.summary_max_chars, top_k_keywords=cfg.top_k_keywords, v2=v2
    )

    data = complete_vision_json(text=prompt, image_data_url=image_data_url)

    summary = str(data.get("summary", "")).strip()[: cfg.summary_max_chars]

    keywords_raw = data.get("keywords", [])
    if not isinstance(keywords_raw, list):
        keywords_raw = []

    objects_raw = data.get("objects", [])
    if not isinstance(objects_raw, list):
        objects_raw = []
    objects: list[str] = []
    for obj in objects_raw:
        o = str(obj).strip()
        if o and o not in objects:
            objects.append(o)

    if v2:
        # v2: 결정적 후처리 — 키워드 정규화(generic 제거) + objects 를 검색 키워드로 승격(top_k cap).
        keywords = promote_objects_to_keywords(
            keywords_raw, objects, limit=cfg.top_k_keywords
        )
    else:
        # v1: 현행 inline 루프(dedup·top_k cap·objects 미승격) — 바이트 동일 보존.
        keywords = []
        for kw in keywords_raw:
            k = str(kw).strip()
            if k and k not in keywords:
                keywords.append(k)
            if len(keywords) >= cfg.top_k_keywords:
                break

    return {
        "summary": summary,
        "keywords": keywords,
        "objects": objects,
    }
