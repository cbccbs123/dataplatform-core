"""F-3.1 텍스트/문서 추출 함수(디스패처가 호출).

기존 추출/요약/임베딩/FTS 함수를 재사용해 ``AssetRecord`` 로 매핑한다
(``run_extract_meta.py`` 의 텍스트 분기와 동일 로직, 출력만 AssetRecord).
"""

from __future__ import annotations

from src.config.settings import get_current_settings
from src.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from src.skills.meta_split import split_core_ext

_CHANNEL_ST = "st"


def _extract_text_meta(ctx: ExtractContext) -> AssetRecord:
    # 무거운 import(추출/요약)는 함수 내부 — 디스패처 import 시 미로딩.
    # 모든 LLM 은 설정된 단일 온프레미스 엔드포인트를 사용한다(외부 LLM 미사용).
    from src.extractors.text_meta_extractor import extract_text_meta
    from src.llm.text_summarizer import summarize_and_extract_keywords
    from src.preprocess.media_item_search_text import build_media_item_fts_plain

    cfg = ctx.settings or get_current_settings()
    file_kind = ctx.modality
    meta = extract_text_meta(
        file_path=ctx.file_path,
        file_kind=file_kind,
        encoding=cfg.encoding,
        chunk_size=cfg.text_embedding_chunk_size,
        embedding_model_name=cfg.text_embedding_model,
    )
    meta = meta | summarize_and_extract_keywords(file_path=ctx.file_path, file_kind=file_kind)
    fts_plain = build_media_item_fts_plain(file_uri=ctx.file_path, meta=meta)
    core_meta, ext_meta = split_core_ext(meta)
    return AssetRecord(core_meta=core_meta, ext_meta=ext_meta, tags=[], fts_plain=fts_plain, embeddings=[])


def _embed_text(ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]:
    from src.embedders.text_embedder import embedding_text_chunks

    cfg = ctx.settings or get_current_settings()
    chunks = embedding_text_chunks(
        ctx.file_path,
        file_kind=ctx.modality,
        encoding=cfg.encoding,
        chunk_size=cfg.text_embedding_chunk_size,
        embedding_model_name=cfg.text_embedding_model,
        normalize_embeddings=cfg.text_embedding_normalize,
    )
    return [
        EmbeddingItem(
            channel=_CHANNEL_ST,
            vector=c["embedding_vector"],
            model_name=cfg.text_embedding_model,
            chunk_index=int(c["chunk_index"]),
        )
        for c in chunks
    ]
