"""F-3.4 디스패처 입출력 데이터클래스.

``AssetRecord`` 는 모든 추출 함수(extract_text/image/video/audio)의 **통일 출력 계약**이다.
영속화(F-1.3 ``persist_asset``)가 이 한 가지 형태만 받아 ``asset``/``asset_metadata``/
``asset_embedding`` 에 단일 경로로 적재한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmbeddingItem:
    """``asset_embedding`` 한 행에 대응. 채널·청크별 1536D 벡터."""

    channel: str  # 'st' | 'clip' | (옵션 B) 'medclip' 등
    vector: list[float]
    model_name: str
    model_version: str | None = None
    chunk_index: int = 0


@dataclass
class ExtractContext:
    """추출 함수에 넘기는 입력 컨텍스트."""

    file_path: str
    modality: str  # MediaKind/OfficeKind 값 (txt/pdf/json/word/excel/powerpoint/image/video/audio)
    domain: str = "general"  # 'general' | 'medical' | 'review'
    settings: Any = None  # PipelineSettings (선택)
    db: Any = None  # PostgresUtil (선택)
    scratch: dict[str, Any] = field(default_factory=dict)  # 스테이지 간 중간 산출물 핸드오프(재계산 방지)


@dataclass
class AssetRecord:
    """추출 결과의 통일 표현. asset_metadata(core/ext/tags/fts) + asset_embedding(embeddings)."""

    core_meta: dict[str, Any] = field(default_factory=dict)  # 파일/시스템 메타
    ext_meta: dict[str, Any] = field(default_factory=dict)  # 도메인 신호(summary/keywords/labels 등)
    tags: list[str] = field(default_factory=list)
    fts_plain: str = ""  # FTS 평문 (to_tsvector 입력)
    embeddings: list[EmbeddingItem] = field(default_factory=list)
