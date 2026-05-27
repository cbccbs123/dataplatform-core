"""F-3.2 영상 추출 함수(디스패처가 호출).

``run_extract_meta.py`` 의 영상 분기를 이식(키프레임 → VLM 요약 → 키프레임별 ST/CLIP 임베딩 쌍).
출력만 ``AssetRecord``. 무거운 import(scenedetect/CLIP/VLM)는 함수 내부에 둔다.
"""

from __future__ import annotations

from src.config.settings import get_current_settings
from src.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from src.skills.meta_split import split_core_ext

_CHANNEL_ST = "st"
_CHANNEL_CLIP = "clip"


def _extract_video_meta(ctx: ExtractContext) -> AssetRecord:
    from src.embedders.video_embedder import embed_video_keyframes_clip
    from src.llm.image_summarizer import summarize_image_caption_keywords_objects_from_jpeg_bytes
    from src.llm.video_summarizer import summarize_video_from_scene_results
    from src.preprocess.media_item_search_text import build_media_item_fts_plain
    from src.preprocess.video_keyframes import (
        extract_video_basic_meta,
        extract_video_representative_frame_bytes,
    )

    cfg = ctx.settings or get_current_settings()
    file = ctx.file_path

    frame_items = extract_video_representative_frame_bytes(
        video_path=file, max_frames=cfg.video_max_keyframes
    )
    korean_labels_per_frame: list[list[str]] = []
    result: list[dict] = []
    for item in frame_items:
        summ = summarize_image_caption_keywords_objects_from_jpeg_bytes(item["jpeg_bytes"])
        objects = summ.get("objects") or []
        obj_list = [str(o) for o in objects] if isinstance(objects, list) else []
        korean_labels_per_frame.append(obj_list)
        result.append(
            {
                "scene_index": item["scene_index"],
                "start_sec": item["start_sec"],
                "end_sec": item["end_sec"],
                "frame_sec": item["frame_sec"],
                "jpeg_bytes": item["jpeg_bytes"],
                "summary": summ,
            }
        )

    meta = extract_video_basic_meta(file_path=file)
    meta = meta | summarize_video_from_scene_results(result)
    clip_ve = embed_video_keyframes_clip(result, korean_labels_per_frame=korean_labels_per_frame)

    # 키프레임 라벨 score 하한 + top-k
    for kf in clip_ve.get("keyframes") or []:
        labels_all = kf.get("labels") or []
        kf["labels"] = [
            it for it in labels_all if float(it.get("score") or 0.0) >= cfg.labels_score_min
        ][: cfg.video_keyframe_labels_meta_top_k]
    for _it in result:
        _it.pop("jpeg_bytes", None)

    meta["keyframes"] = [
        {k: v for k, v in kf.items() if k != "clip_image_embedding"} for kf in clip_ve["keyframes"]
    ]
    fts_plain = build_media_item_fts_plain(file_uri=file, meta=meta)

    # 키프레임별 임베딩 입력을 stash(키프레임/CLIP/VLM 재실행 방지)
    stash: list[dict] = []
    for kf in clip_ve["keyframes"]:
        summ = kf.get("summary") if isinstance(kf.get("summary"), dict) else {}
        stash.append(
            {
                "clip_vec": kf["clip_image_embedding"],
                "summary": str(summ.get("summary", "") or ""),
                "keywords": summ["keywords"] if isinstance(summ.get("keywords"), list) else [],
                "labels": kf.get("labels") or [],
            }
        )
    ctx.scratch["keyframes"] = stash

    core_meta, ext_meta = split_core_ext(meta)
    return AssetRecord(core_meta=core_meta, ext_meta=ext_meta, tags=[], fts_plain=fts_plain, embeddings=[])


def _embed_video(ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]:
    from src.config.embedding_constants import DEFAULT_CLIP_MODEL_NAME
    from src.embedders.text_embedder import embed_texts, pad_embedding_to_storage_dim
    from src.preprocess.vlm_text_for_embedding import build_image_vlm_text_for_embedding

    cfg = ctx.settings or get_current_settings()
    embeddings: list[EmbeddingItem] = []
    for i, kf in enumerate(ctx.scratch["keyframes"]):
        frame_meta = {"summary": kf["summary"], "keywords": kf["keywords"], "labels": kf["labels"]}
        chunk_content = build_image_vlm_text_for_embedding(frame_meta)
        if not chunk_content.strip():
            chunk_content = " "
        st_raw = embed_texts(
            [chunk_content],
            model_name=cfg.text_embedding_model,
            normalize_embeddings=cfg.text_embedding_normalize,
        )[0]
        st_vec = pad_embedding_to_storage_dim(st_raw)
        # 키프레임당 ST/CLIP 한 쌍(같은 chunk_index, 채널로 구분)
        embeddings.append(EmbeddingItem(channel=_CHANNEL_ST, vector=st_vec, model_name=cfg.text_embedding_model, chunk_index=i))
        embeddings.append(EmbeddingItem(channel=_CHANNEL_CLIP, vector=kf["clip_vec"], model_name=DEFAULT_CLIP_MODEL_NAME, chunk_index=i))
    return embeddings


def extract_video(ctx: ExtractContext) -> AssetRecord:
    """기존 시그니처 보존 래퍼 = 키프레임/메타(+벡터 stash) + 임베딩 합성(동작 불변)."""
    rec = _extract_video_meta(ctx)
    rec.embeddings = _embed_video(ctx, rec)
    return rec
