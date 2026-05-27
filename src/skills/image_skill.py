"""F-3.2 이미지 추출 함수(디스패처가 호출).

``run_extract_meta.py`` 의 이미지 분기를 이식. 출력만 ``AssetRecord``.
무거운 import(torch/CLIP/VLM)는 함수 내부에 둔다 — 텍스트 전용 실행 시 미로딩.
"""

from __future__ import annotations

from src.config.settings import get_current_settings
from src.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from src.skills.meta_split import split_core_ext

_CHANNEL_ST = "st"
_CHANNEL_CLIP = "clip"


def _extract_image_meta(ctx: ExtractContext) -> AssetRecord:
    from src.embedders.image_embedder import (
        clip_zero_shot_ko_meta_items,
        zero_shot_tag_image_korean_clip,
    )
    from src.extractors.image_meta_extractor import extract_image_meta
    from src.llm.image_summarizer import summarize_image_caption_keywords_objects
    from src.preprocess.media_item_search_text import build_media_item_fts_plain

    cfg = ctx.settings or get_current_settings()
    file = ctx.file_path

    # 1) 파일·이미지 속성 메타  2) VLM 캡션·키워드·객체
    meta = extract_image_meta(file_path=file)
    summary = summarize_image_caption_keywords_objects(file_path=file)
    objects = summary.get("objects") or []
    meta = meta | summary
    meta_for_fts = dict(meta)

    # 3) CLIP 이미지 임베딩 + 한글 라벨 제로샷
    obj_list = [str(o) for o in objects] if isinstance(objects, list) else []
    zs = zero_shot_tag_image_korean_clip(file_path=file, korean_labels=obj_list)
    if zs["label_scores"]:
        labels_all = clip_zero_shot_ko_meta_items(zs["label_scores"])
        labels = [
            it for it in labels_all if float(it.get("score") or 0.0) >= cfg.labels_score_min
        ][: cfg.image_labels_meta_top_k]
        meta = meta | {"labels": labels}
        meta_for_fts = meta_for_fts | {"labels": labels}

    fts_plain = build_media_item_fts_plain(file_uri=file, meta=meta_for_fts)
    meta.pop("objects", None)  # objects 는 CLIP 후보용일 뿐 최종 meta 에서 제외

    # 임베딩 슬롯이 재계산 없이 쓰도록 clip 이미지 벡터를 핸드오프
    ctx.scratch["clip_vec"] = zs["clip_image_embedding"]

    core_meta, ext_meta = split_core_ext(meta)
    return AssetRecord(core_meta=core_meta, ext_meta=ext_meta, tags=[], fts_plain=fts_plain, embeddings=[])


def _embed_image(ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]:
    from src.config.embedding_constants import DEFAULT_CLIP_MODEL_NAME
    from src.embedders.text_embedder import embed_texts, pad_embedding_to_storage_dim
    from src.preprocess.vlm_text_for_embedding import build_image_vlm_text_for_embedding

    cfg = ctx.settings or get_current_settings()
    meta = dict(rec.core_meta) | dict(rec.ext_meta)
    chunk_content = build_image_vlm_text_for_embedding(meta)
    if not chunk_content.strip():
        chunk_content = " "
    st_raw = embed_texts(
        [chunk_content],
        model_name=cfg.text_embedding_model,
        normalize_embeddings=cfg.text_embedding_normalize,
    )[0]
    st_vec = pad_embedding_to_storage_dim(st_raw)
    clip_vec = ctx.scratch["clip_vec"]
    return [
        EmbeddingItem(channel=_CHANNEL_ST, vector=st_vec, model_name=cfg.text_embedding_model, chunk_index=0),
        EmbeddingItem(channel=_CHANNEL_CLIP, vector=clip_vec, model_name=DEFAULT_CLIP_MODEL_NAME, chunk_index=0),
    ]


def extract_image(ctx: ExtractContext) -> AssetRecord:
    """기존 시그니처 보존 래퍼 = 메타 추출(+clip 벡터 stash) + 임베딩 합성(동작 불변)."""
    rec = _extract_image_meta(ctx)
    rec.embeddings = _embed_image(ctx, rec)
    return rec
