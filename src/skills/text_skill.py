"""F-3.1 텍스트/문서 추출 함수(디스패처가 호출). 구현은 T1-6.

기존 ``extract_text_meta``·``summarize_and_extract_keywords``·``embedding_text_chunks``·
``build_media_item_fts_plain`` 를 재사용해 ``AssetRecord`` 로 매핑할 예정.
"""

from __future__ import annotations

from src.dispatch.types import AssetRecord, ExtractContext


def extract_text(ctx: ExtractContext) -> AssetRecord:
    raise NotImplementedError("F-3.1(T1-6)에서 구현 예정")
