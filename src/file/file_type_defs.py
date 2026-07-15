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

# ``asset_embedding.channel=st`` 임베딩이 있는 media_type(문서·오디오 STT·영상 VLM 텍스트 등).
# 벡터 검색과 별도로 풀텍스트(BM25)는 OpenSearch 색인(메타 기반)에서 처리한다(037 OS 전용; PG FTS 폐기).
MEDIA_TYPES_ST_CHUNK_SEARCH = frozenset(
    {*ALLOWED_TEXT_META_FILE_KINDS, MediaKind.AUDIO.value, MediaKind.VIDEO.value}
)

# ``asset_embedding.channel=clip``(CLIP)으로 시각·키프레임 검색에 쓸 media_type
MEDIA_TYPES_CLIP_CHUNK_SEARCH = frozenset(
    {MediaKind.IMAGE.value, MediaKind.VIDEO.value}
)


# asset.modality 저장 허용값(canonical). v292 CHECK 와 일치. unknown 은 모달리티가 아닌 격리표식.
CANONICAL_MODALITIES = ("text", "image", "video", "audio", "unknown")


def modality_of(file_kind: str) -> str:
    """file_kind(``detect_file_kind`` 판정값) → canonical modality. 결정적·순수(헌법 3·6조·053).

    저장(asset.modality)만 canonical 5종으로 좁힌다 — 추출은 file_kind 를 그대로 쓰므로
    (dispatcher·data_loader) 이 매핑은 오직 ``create_asset`` 저장 경계에서만 적용한다(A안).
    세분류(txt vs json vs pdf…)는 fs_path 확장자로 재도출 가능(file_ext)이라 저장값 정규화가
    추출 정확성을 해치지 않는다.
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
    "MEDIA_TYPES_CLIP_CHUNK_SEARCH",
    "CANONICAL_MODALITIES",
    "modality_of",
]
