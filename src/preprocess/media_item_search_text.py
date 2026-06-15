"""``media_items.search_vector``(PostgreSQL ``to_tsvector``)용 평문만 생성."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from src.config.settings import get_current_settings

# ``INSERT ... to_tsvector('simple', ...)`` 와 맞출 것
MEDIA_ITEM_FTS_CONFIG = "simple"

_MAX_PLAIN_CHARS = 32_000
_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\uFEFF]")
_CTRL = re.compile(r"[\x00-\x1F\x7F]")
_WS = re.compile(r"\s+")
# 너무 일반적이라 FTS 변별력이 없는 CLIP 라벨 — search_vector 만 부풀려 정밀도를 떨어뜨리므로 제외.
_NOISE_LABELS = {
    "사람",
    "인물",
    "인간",
    "텍스트",
    "문자",
    "글자",
    "사람들"
}


def normalize_search_text(value: object) -> str:
    """FTS 입력 전 경량 정규화: NFKC + 제어문자 제거 + 공백 정리."""
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value))
    s = _ZERO_WIDTH.sub("", s)
    s = _CTRL.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _nonempty_str_list(val: Any) -> list[str]:
    if not isinstance(val, list):
        return []
    out: list[str] = []
    for x in val:
        n = normalize_search_text(x)
        if n:
            out.append(n)
    return out


def _clip_labels_to_strings_with_limits(
    labels: Any,
    *,
    top_k: int,
    score_min: float,
) -> list[str]:
    """
    CLIP labels(보통 {label, score} dict 리스트)을 문자열 리스트로 바꾼다.
    - score_min 미만 제외
    - 불용 라벨 제외
    - top_k만 유지 (보통 score 내림차순으로 들어오지만, 안전하게 정렬)
    """
    if top_k <= 0:
        return []

    items: list[tuple[str, float]] = []
    for item in labels or []:
        if isinstance(item, dict):
            lab = item.get("label")
            if not lab:
                continue
            s = normalize_search_text(lab)
            if not s:
                continue
            sc = item.get("score")
            try:
                score = float(sc) if sc is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            if score < score_min:
                continue
            if s.lower() in _NOISE_LABELS:
                continue
            items.append((s, score))
        elif isinstance(item, str):
            s = normalize_search_text(item)
            if not s:
                continue
            if s.lower() in _NOISE_LABELS:
                continue
            # score가 없는 경우는 포함한다.
            items.append((s, float("inf")))

    # score 기반 내림차순 정렬 후 상위 top_k
    items.sort(key=lambda x: -x[1])
    return [lab for (lab, _sc) in items[:top_k]]


def build_media_item_fts_plain(
    *,
    file_uri: str,
    meta: dict[str, Any],
    max_plain_chars: int = _MAX_PLAIN_CHARS,
) -> str:
    """
    ``search_vector`` 용 평문: 확장자 제거 파일명(stem), ``summary``, ``keywords``, ``objects``,
    설정 상한·점수·불용어가 적용된 CLIP ``labels``(이미지 및 영상 키프레임).

    ``description`` 은 요약과 겹치므로 넣지 않는다.
    """
    raw_file_name = Path(file_uri).stem if file_uri else ""
    file_name = normalize_search_text(raw_file_name)
    summary = normalize_search_text(meta.get("summary") or "")

    parts: list[str] = []
    if file_name:
        parts.append(file_name)
    if summary:
        parts.append(summary)
    parts.extend(_nonempty_str_list(meta.get("keywords")))
    parts.extend(_nonempty_str_list(meta.get("objects")))
    cfg = get_current_settings()
    parts.extend(
        _clip_labels_to_strings_with_limits(
            meta.get("labels"),
            top_k=cfg.image_labels_search_top_k,
            score_min=cfg.labels_score_min,
        )
    )
    for kf in meta.get("keyframes") or []:
        if isinstance(kf, dict):
            parts.extend(
                _clip_labels_to_strings_with_limits(
                    kf.get("labels"),
                    top_k=cfg.video_keyframe_labels_search_top_k,
                    score_min=cfg.labels_score_min,
                )
            )

    plain = normalize_search_text("\n".join(p for p in parts if p))
    if len(plain) > max_plain_chars:
        plain = plain[:max_plain_chars]
    return plain
