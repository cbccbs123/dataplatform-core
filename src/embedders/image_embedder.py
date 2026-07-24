"""CLIP 시각/언어 임베딩 seam — CLIP 추론(학습 배제·inference only).

두 역할을 한 모델로 묶는다(같은 임베딩 공간이라 가능):
  - 인덱싱 측: 이미지/키프레임을 CLIP 이미지 인코더로 벡터화(``src/skills/image_skill.py``·
    ``video_embedder``). 동시에 VLM 이 준 한글 후보 라벨로 제로샷 점수도 매긴다(이미지 인코딩 1회 재사용).
    이 CLIP 이미지 벡터는 현재 관계 후보(``asset_candidates``)가 소비한다.
  - (제거됨·2026-07-20) 텍스트 질의를 CLIP **텍스트** 인코더로 벡터화하던
    ``embed_clip_text_query_for_image_search`` — 037 이후 검색 read path 는 OpenSearch
    하이브리드(활성 텍스트 채널 임베딩)뿐이라 소비처 0 이 되어 069 US-F 에서 제거했다.

따라서 CLIP 벡터를 쓰는 경로(인덱싱·관계)는 동일 ``model_name``(기본 ``DEFAULT_CLIP_MODEL_NAME``)을
같은 공간 비교 전제로 공유한다.
모든 출력 벡터는 ``clip_image_row_to_embedding_1536`` 으로 DB ``vector(1536)`` 에 맞춘다
(CLIP 기본 차원은 512라 뒤를 0 패딩 — 헌법 1536D 통일).

``coerce_clip_*_feature_tensor`` 군은 transformers 버전마다 ``get_*_features`` 반환형이
텐서/``BaseModelOutput``(pooler·last_hidden_state)로 갈리는 것을 흡수하는 호환 셈이다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from src.config.embedding_constants import DEFAULT_CLIP_MODEL_NAME, FIX_EMBEDDING_DIMENSION


class ZeroShotKoTagResult(TypedDict):
    """한글 라벨별 제로샷 점수 + 검색용 CLIP 이미지 벡터(1536차원, 부족분 0 패딩)."""

    label_scores: dict[str, float]
    clip_image_embedding: list[float]


@lru_cache(maxsize=2)
def get_clip(model_name: str) -> tuple[CLIPProcessor, CLIPModel]:
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def l2_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return x / n


def pad_or_truncate_1d(vec: np.ndarray, target_dim: int) -> np.ndarray:
    if vec.shape[0] == target_dim:
        return vec
    if vec.shape[0] > target_dim:
        return vec[:target_dim]
    return np.pad(vec, (0, target_dim - vec.shape[0]), mode="constant", constant_values=0.0)


def clip_image_row_to_embedding_1536(image_row: torch.Tensor | np.ndarray) -> list[float]:
    """CLIP 시각 행(배치 1행 또는 1차원 numpy)을 DB 저장 차원 벡터로 패딩/절단."""
    if isinstance(image_row, torch.Tensor):
        vec = image_row.detach().cpu().numpy().astype(np.float32)
    else:
        vec = np.asarray(image_row, dtype=np.float32).ravel()
    vec = pad_or_truncate_1d(vec, FIX_EMBEDDING_DIMENSION)
    return vec.tolist()


def apply_clip_projection_or_pass(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """
    CLIP 공유 임베딩으로 맞춘다. ``in_features`` 크기면 Linear 적용, ``out_features`` 면 이미 투영된 값.
    """
    in_d, out_d = int(linear.in_features), int(linear.out_features)
    d = int(x.shape[-1])
    if d == in_d:
        with torch.no_grad():
            return linear(x)
    if d == out_d:
        return x
    raise RuntimeError(
        f"CLIP projection 차원 불일치: got {d}, expected in={in_d} or out={out_d}"
    )


def coerce_clip_image_feature_tensor(model: CLIPModel, feat: object) -> torch.Tensor:
    if isinstance(feat, torch.Tensor):
        return feat
    proj = model.visual_projection
    pooler = getattr(feat, "pooler_output", None)
    if pooler is not None:
        return apply_clip_projection_or_pass(proj, pooler)
    last = getattr(feat, "last_hidden_state", None)
    if last is not None:
        return apply_clip_projection_or_pass(proj, last[:, 0, :])
    raise TypeError(f"CLIP 이미지 특징을 텐서로 바꿀 수 없습니다: {type(feat)!r}")


def coerce_clip_text_feature_tensor(
    model: CLIPModel,
    feat: object,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(feat, torch.Tensor):
        return feat
    proj = model.text_projection
    pooler = getattr(feat, "pooler_output", None)
    if pooler is not None:
        return apply_clip_projection_or_pass(proj, pooler)
    last = getattr(feat, "last_hidden_state", None)
    if last is not None:
        # CLIP 텍스트 풀링은 마지막 실제 토큰(EOT) 위치의 hidden state를 쓴다.
        # 패딩이 섞이므로 attention_mask 로 시퀀스 실길이를 구해 그 위치를 골라야 한다
        # (mask 없으면 [-1] 가정). pooler_output 이 없는 transformers 버전용 폴백 경로.
        if attention_mask is not None:
            seq_lens = (attention_mask.long().sum(dim=1) - 1).clamp(min=0)
            b = torch.arange(last.size(0), device=last.device, dtype=torch.long)
            pooled = last[b, seq_lens, :]
        else:
            pooled = last[:, -1, :]
        return apply_clip_projection_or_pass(proj, pooled)
    raise TypeError(f"CLIP 텍스트 특징을 텐서로 바꿀 수 없습니다: {type(feat)!r}")


def clip_image_embedding_normalized(
    processor: CLIPProcessor,
    model: CLIPModel,
    rgb_img: Image.Image,
) -> torch.Tensor:
    inputs = processor(images=rgb_img, return_tensors="pt")
    with torch.no_grad():
        raw = model.get_image_features(pixel_values=inputs["pixel_values"])
    feat = coerce_clip_image_feature_tensor(model, raw)
    return l2_normalize_rows(feat)


def clip_text_embeddings_normalized(
    processor: CLIPProcessor,
    model: CLIPModel,
    texts: list[str],
) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        raw = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
    feat = coerce_clip_text_feature_tensor(
        model,
        raw,
        attention_mask=inputs.get("attention_mask"),
    )
    return l2_normalize_rows(feat)


def clip_zero_shot_logits(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    model: CLIPModel,
) -> torch.Tensor:
    # CLIP 학습 온도(logit_scale=log(1/T), 로그로 저장)를 exp 로 복원해 코사인 유사도(-1~1)에 곱한다(≈100배).
    # 이 스케일이 있어야 뒤이은 softmax 가 라벨 간 확률을 분리한다 — 없으면 [-1,1] 코사인의 softmax 가 거의 균일해짐.
    logits = (image_emb @ text_emb.T) * model.logit_scale.exp()
    return logits[0]


def clip_zero_shot_label_scores_from_image_emb(
    processor: CLIPProcessor,
    model: CLIPModel,
    image_emb: torch.Tensor,
    cleaned: list[str],
    *,
    text_template: str = "사진 속 {label}",
) -> dict[str, float]:
    """이미 L2 정규화된 이미지 임베딩 배치(보통 1xD)와 정리된 한글 라벨로 소프트맥스 점수."""
    if not cleaned:
        return {}
    prompt_texts = [text_template.format(label=lab) for lab in cleaned]
    text_emb = clip_text_embeddings_normalized(processor, model, prompt_texts)
    logits = clip_zero_shot_logits(image_emb, text_emb, model)
    probs = logits.softmax(dim=0).detach().cpu().numpy()
    return {lab: float(p) for lab, p in zip(cleaned, probs, strict=False)}


def clip_zero_shot_ko_meta_items(label_scores: dict[str, float]) -> list[dict[str, float | str]]:
    """메타 JSON용: 각 한글 라벨과 CLIP 제로샷 소프트맥스 점수(높은 순)."""
    items: list[dict[str, float | str]] = [
        {"label": lab, "score": float(s)} for lab, s in label_scores.items()
    ]
    # 069 B1(P2-1): 동점 score 를 label 문자열 2차키로 깨서 top-k 컷을 결정화한다(헌법 3조).
    # 2차키 없으면 score 만으로는 동점 라벨의 상대 순서가 입력(dict) 순서에 좌우돼 재현성이 흔들린다.
    items.sort(key=lambda x: (-float(x["score"]), str(x["label"])))
    return items


def normalize_korean_label_candidates(korean_labels: list[str]) -> list[str]:
    """VLM 등에서 온 한글 후보 문자열을 중복 제거·trim 한 리스트로 만든다.

    069 B1(P2-1): ``set`` 은 해시 순서(비결정)라 동일 입력이라도 후보 순서가 흔들려 하위 CLIP
    제로샷 점수 계산·top-k 컷의 재현성을 저해했다. ``dict.fromkeys`` 로 바꿔 **첫 등장 순서를
    보존**하면서 중복만 제거한다(헌법 3조·결정성). score 는 순서와 무관하므로 값 자체는 불변.
    """
    return list(dict.fromkeys(lab for raw in korean_labels if (lab := str(raw).strip())))


def zero_shot_tag_rgb_korean_clip(
    rgb_img: Image.Image,
    korean_labels: list[str],
    *,
    model_name: str = DEFAULT_CLIP_MODEL_NAME,
    text_template: str = "사진 속 {label}",
) -> ZeroShotKoTagResult:
    """
    RGB ``PIL.Image`` 에 대해 VLM 후보 라벨로 CLIP 제로샷 점수(소프트맥스)와
    검색용 이미지 벡터(1536)를 반환한다. 이미지 인코딩은 1회.
    """
    cleaned = normalize_korean_label_candidates(korean_labels)
    processor, model = get_clip(model_name)
    rgb = rgb_img.convert("RGB")
    image_emb = clip_image_embedding_normalized(processor, model, rgb)
    clip_image_embedding = clip_image_row_to_embedding_1536(image_emb[0])
    label_scores = clip_zero_shot_label_scores_from_image_emb(
        processor, model, image_emb, cleaned, text_template=text_template
    )
    return {
        "label_scores": label_scores,
        "clip_image_embedding": clip_image_embedding,
    }


def zero_shot_tag_image_korean_clip(
    file_path: str | Path,
    korean_labels: list[str],
    *,
    model_name: str = DEFAULT_CLIP_MODEL_NAME,
    text_template: str = "사진 속 {label}",
) -> ZeroShotKoTagResult:
    """
    VLM이 준 한글 후보 라벨로 CLIP 제로샷 점수(소프트맥스)와,
    검색용 이미지 벡터(1536차원, 부족분 0 패딩)를 반환한다. 이미지 인코딩은 1회.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    with Image.open(path) as img:
        rgb_img = img.convert("RGB")
    return zero_shot_tag_rgb_korean_clip(
        rgb_img,
        korean_labels,
        model_name=model_name,
        text_template=text_template,
    )
