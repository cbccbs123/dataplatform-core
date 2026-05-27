"""F-3.2 오디오 추출 함수(디스패처가 호출).

``run_extract_meta.py`` 의 오디오 분기를 이식(STT → 요약 → 청크 임베딩). 출력만 ``AssetRecord``.
무거운 import(faster-whisper/임베더)는 함수 내부에 둔다.
"""

from __future__ import annotations

from src.config.settings import get_current_settings
from src.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from src.skills.meta_split import split_core_ext

_CHANNEL_ST = "st"


def _extract_audio_meta(ctx: ExtractContext) -> AssetRecord:
    from src.extractors.audio_meta_extractor import extract_audio_meta
    from src.llm.text_summarizer import summarize_and_extract_keywords_from_audio
    from src.preprocess.media_item_search_text import build_media_item_fts_plain
    from src.preprocess.stt import transcribe_audio_local

    file = ctx.file_path
    stt_result = transcribe_audio_local(file_path=file)
    meta = extract_audio_meta(file_path=file)
    meta = meta | summarize_and_extract_keywords_from_audio(text=stt_result["text"])
    fts_plain = build_media_item_fts_plain(file_uri=file, meta=meta)

    ctx.scratch["stt_text"] = stt_result["text"]  # 임베딩 슬롯 재사용(whisper 재실행 방지)

    core_meta, ext_meta = split_core_ext(meta)
    return AssetRecord(core_meta=core_meta, ext_meta=ext_meta, tags=[], fts_plain=fts_plain, embeddings=[])


def _embed_audio(ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]:
    from src.embedders.text_embedder import embedding_plain_text_chunks

    cfg = ctx.settings or get_current_settings()
    stt_text = ctx.scratch.get("stt_text")
    if stt_text is None:
        raise RuntimeError("_embed_audio: ctx.scratch['stt_text'] 없음 — _extract_audio_meta 를 같은 ctx 로 먼저 실행해야 합니다.")
    chunks = embedding_plain_text_chunks(
        stt_text,
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
