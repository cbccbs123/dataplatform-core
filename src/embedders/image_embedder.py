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
    """CLIP 전처리기·모델을 한 번만 로드해 재사용한다(추론 모드 고정).

    Args:
        model_name: 모델 이름. **캐시 키이기도 하다** — 이름이 다르면 별도 인스턴스가 뜬다.

    Returns:
        ``(processor, model)``. 모델은 평가 모드라 학습 관련 동작(드롭아웃 등)이 꺼져 있다.
    """
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def l2_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """행마다 길이를 1로 맞춘다 — 그래야 내적이 곧 코사인 유사도가 된다.

    Args:
        x: (행, 차원) 텐서.

    Returns:
        정규화된 텐서. 길이가 0인 행은 **아주 작은 값으로 나눠** 0으로 나누는 것을 피한다
        (그 행은 결과적으로 0 벡터에 가깝게 남는다).
    """
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return x / n


def pad_or_truncate_1d(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """벡터 길이를 저장 차원에 맞춘다 — 길면 자르고, 짧으면 0으로 채운다.

    모델마다 출력 차원이 달라도 **한 컬럼에 저장**하려면 길이를 통일해야 한다.

    Args:
        vec: 1차원 벡터.
        target_dim: 맞출 길이.

    Returns:
        길이가 ``target_dim`` 인 벡터(이미 맞으면 원본 그대로).
    """
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
    """CLIP 이미지 출력이 무엇으로 오든 **비교 가능한 텐서**로 맞춘다.

    라이브러리 버전에 따라 텐서가 그대로 오기도 하고, 풀링 결과나 은닉 상태 객체로 오기도
    한다. 후자는 투영을 거쳐야 텍스트 임베딩과 같은 공간이 된다.

    Args:
        model: 투영 계층을 꺼낼 모델.
        feat: 모델 출력(텐서 또는 출력 객체).

    Returns:
        이미지 임베딩 텐서.

    Raises:
        TypeError: 알아볼 수 없는 형태일 때 — 조용히 넘기면 엉뚱한 벡터가 저장된다.
    """
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
    """CLIP 텍스트 출력을 비교 가능한 텐서로 맞춘다(이미지판과 같은 취지).

    Args:
        model: 투영 계층을 꺼낼 모델.
        feat: 모델 출력(텐서 또는 출력 객체).
        attention_mask: 있으면 **패딩 토큰을 빼고** 평균 낸다. 없으면 전체를 평균 내므로
            길이가 다른 문장이 섞였을 때 값이 흐려진다.

    Returns:
        텍스트 임베딩 텐서.

    Raises:
        TypeError: 알아볼 수 없는 형태일 때.
    """
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
    """이미지 하나를 길이 1로 정규화된 임베딩으로 만든다(추론만·기울기 계산 없음).

    Args:
        processor: 이미지 전처리기.
        model: CLIP 모델.
        rgb_img: RGB 이미지. **모드 변환은 호출자 책임**이다(회색조·투명 이미지를 그대로
            넣으면 채널 수가 맞지 않는다).

    Returns:
        (1, 차원) 정규화 텐서 — 내적이 곧 코사인 유사도다.
    """
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
    """여러 문장을 한 번에 정규화된 임베딩으로 만든다(추론만).

    길이가 다른 문장을 함께 넣으므로 패딩이 붙는데, 그 패딩이 평균에 섞이지 않도록
    마스크를 함께 넘긴다.

    Args:
        processor: 텍스트 전처리기.
        model: CLIP 모델.
        texts: 문장 목록.

    Returns:
        (문장 수, 차원) 정규화 텐서.
    """
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
    """이미지 하나와 여러 라벨 사이의 점수를 낸다(제로샷 분류용).

    Args:
        image_emb: 정규화된 이미지 임베딩(1행).
        text_emb: 정규화된 라벨 임베딩(라벨 수만큼).
        model: 학습된 온도 계수를 꺼낼 모델.

    Returns:
        라벨 수만큼의 점수 벡터. 이후 softmax 를 씌워 확률로 쓴다.
    """
    # 코사인 유사도(-1~1)에 학습된 온도 계수를 곱한다(로그로 저장돼 있어 exp 로 복원·약 100배).
    # 이 스케일이 없으면 softmax 결과가 라벨 간에 거의 평평해져 분류가 되지 않는다.
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
