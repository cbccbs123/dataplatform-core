"""영상 키프레임 CLIP 임베딩 seam — ``image_embedder`` 의 단일 이미지 경로를 프레임 배치로 확장.

``src/skills/video_skill.py`` 가 호출한다: scenedetect 로 뽑은 대표 키프레임 JPEG 마다
CLIP 이미지 벡터(1536)와, VLM objects 를 후보로 한 제로샷 라벨 점수를 붙인다.
프레임당 이미지 인코딩 1회 — 같은 임베딩으로 벡터화와 라벨링을 함께 처리한다.
새 CLIP 추론 로직은 두지 않고 ``image_embedder`` 의 primitive 만 재사용한다(공간 일치 보장).
"""

from __future__ import annotations

from io import BytesIO
from typing import TypedDict

from PIL import Image

from src.config.embedding_constants import DEFAULT_CLIP_MODEL_NAME
from src.embedders.frame_types import KeyframeBytesResult
from src.embedders.image_embedder import (
    clip_image_embedding_normalized,
    clip_image_row_to_embedding_1536,
    clip_zero_shot_ko_meta_items,
    clip_zero_shot_label_scores_from_image_emb,
    get_clip,
    normalize_korean_label_candidates,
)


class KeyframeClipEmbedding(TypedDict):
    scene_index: int
    start_sec: float
    end_sec: float
    frame_sec: float
    summary: dict[str, str | list[str]]
    labels: list[dict[str, float | str]] | None
    clip_image_embedding: list[float]


class VideoClipEmbeddingsResult(TypedDict):
    """대표 프레임별 CLIP 임베딩·라벨."""

    keyframes: list[KeyframeClipEmbedding]


def embed_video_keyframes_clip(
    frame_items: list[KeyframeBytesResult],
    *,
    model_name: str = DEFAULT_CLIP_MODEL_NAME,
    korean_labels_per_frame: list[list[str]] | None = None,
    text_template: str = "사진 속 {label}",
) -> VideoClipEmbeddingsResult:
    """
    각 키프레임 JPEG에 대해 CLIP 이미지 임베딩(1536)과,
    ``korean_labels_per_frame``(보통 VLM objects)가 있으면 제로샷 라벨 점수를 붙인다.
    이미지 인코딩은 프레임당 1회.
    """
    if not frame_items:
        return {"keyframes": []}

    processor, model = get_clip(model_name)
    keyframes: list[KeyframeClipEmbedding] = []

    for i, it in enumerate(frame_items):
        with Image.open(BytesIO(it["jpeg_bytes"])) as img:
            rgb = img.convert("RGB")
        image_emb = clip_image_embedding_normalized(processor, model, rgb)

        cleaned: list[str] = []
        if korean_labels_per_frame is not None and i < len(korean_labels_per_frame):
            cleaned = normalize_korean_label_candidates(korean_labels_per_frame[i])

        label_scores = clip_zero_shot_label_scores_from_image_emb(
            processor,
            model,
            image_emb,
            cleaned,
            text_template=text_template,
        )
        labels_meta: list[dict[str, float | str]] | None = (
            clip_zero_shot_ko_meta_items(label_scores) if label_scores else None
        )

        keyframes.append(
            {
                "scene_index": int(it["scene_index"]),
                "start_sec": float(it["start_sec"]),
                "end_sec": float(it["end_sec"]),
                "frame_sec": float(it["frame_sec"]),
                "summary": it["summary"],
                "labels": labels_meta,
                "clip_image_embedding": clip_image_row_to_embedding_1536(image_emb[0]),
            }
        )

    return {"keyframes": keyframes}
