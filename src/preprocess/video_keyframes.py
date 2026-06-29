"""영상 → 장면 단위 대표 키프레임(메모리 JPEG) 추출.

PySceneDetect 의 ``ContentDetector`` 로 컷(장면 전환)을 찾아 각 장면의 중앙 시점 프레임을
OpenCV 로 읽어 JPEG bytes 로 인코딩한다. 파일로 저장하지 않고 메모리에서 바로 image_skill
(CLIP 라벨·VLM 요약)으로 넘겨 영상의 키프레임 임베딩·검색에 쓰기 위한 전처리다.
``extract_video_basic_meta`` 는 별도로 duration/fps/해상도만 뽑는다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict

import cv2
from scenedetect import ContentDetector, detect

if TYPE_CHECKING:
    from src.preprocess.keyframe_dedup import KeyframeDedupConfig

logger = logging.getLogger(__name__)


class KeyframeBytesResult(TypedDict):
    """장면별 대표 프레임(메모리 JPEG) 결과."""

    scene_index: int
    start_sec: float
    end_sec: float
    frame_sec: float
    jpeg_bytes: bytes
    summary: NotRequired[dict[str, str | list[str]]]


class VideoBasicMeta(TypedDict):
    duration: float
    frame_rate: float
    width: int
    height: int


def _to_seconds(timecode: object) -> float:
    if hasattr(timecode, "get_seconds"):
        return float(timecode.get_seconds())  # type: ignore[no-any-return]
    # scenedetect 버전 차이를 고려한 안전장치
    return float(timecode)  # type: ignore[arg-type]


def extract_video_basic_meta(
    file_path: str | Path,
) -> VideoBasicMeta:
    """영상 기본 메타(duration/fps/width/height)를 추출한다."""
    src = Path(file_path)
    if not src.is_file():
        raise FileNotFoundError(str(src))

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {src}")

    try:
        frame_rate = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()

    duration = (frame_count / frame_rate) if frame_rate > 0 else 0.0
    return {
        "duration": round(duration, 3),
        "frame_rate": round(frame_rate, 3),
        "width": width,
        "height": height,
    }


def _read_scene_mid_frame(
    cap: cv2.VideoCapture,
    *,
    start_sec: float,
    end_sec: float,
) -> tuple[float, object]:
    frame_sec = start_sec + max(0.0, (end_sec - start_sec) / 2.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("대표 프레임을 읽지 못했습니다.")
    return frame_sec, frame


def extract_video_representative_frame_bytes(
    video_path: str | Path,
    *,
    threshold: float = 30.0,
    min_scene_len: int = 15,
    jpeg_quality: int = 85,
    max_frames: int | None = None,
    dedup: KeyframeDedupConfig | None = None,
) -> list[KeyframeBytesResult]:
    """
    영상에서 장면(Scene) 단위 대표 프레임(중앙 시점) JPEG bytes를 반환한다.

    파일로 저장하지 않고 메모리에서 바로 후속 요약 파이프라인에 연결할 때 사용한다.

    장면이 하나도 안 잡히면(단일 컷·아주 짧은 영상 등) 영상 중앙 1프레임만 ``scene_index=1`` 로
    돌려준다. ``max_frames`` 는 장면 수 상한(앞에서부터 자름). 프레임을 못 읽거나 JPEG 인코딩에
    실패한 장면은 건너뛴다(예외로 중단하지 않음).

    048: ``dedup`` 가 주어지고 ``enabled`` 면 **VLM 직전 near-dup 제거**를 적용한다(FR-101). 이 경우
    다중 장면 경로에서 ``max_frames`` pre-cap 을 **건너뛰고** 전 장면을 추출한 뒤 ``dedup_keyframes`` 로
    중복을 제거하고, 그 결과를 앞에서부터 ``max_frames`` 로 trim 한다(순서 = dedup → cap·FR-104).
    ``dedup`` 가 ``None`` 이거나 ``enabled=False`` 면 **현행 코드 경로 그대로**(pre-cap 유지)라 추출
    결과가 기존과 바이트 동일하다(FR-103·완전 no-op). 단일 프레임(no-scene) 경로는 1장이라 무변경.
    """
    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(str(src))

    # 048: dedup 활성 여부 — enabled 일 때만 "전 장면 추출 → dedup → cap" 경로를 탄다.
    _dedup_on = dedup is not None and dedup.enabled

    scenes = detect(str(src), ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    if not scenes:
        cap0 = cv2.VideoCapture(str(src))
        if not cap0.isOpened():
            raise RuntimeError(f"비디오를 열 수 없습니다: {src}")
        try:
            frame_rate = float(cap0.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = float(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            duration = (frame_count / frame_rate) if frame_rate > 0 else 0.0
            frame_sec = (duration / 2.0) if duration > 0 else 0.0
            cap0.set(cv2.CAP_PROP_POS_MSEC, frame_sec * 1000.0)
            ok0, frame0 = cap0.read()
            if not ok0 or frame0 is None:
                return []
            enc_ok, encoded0 = cv2.imencode(
                ".jpg",
                frame0,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if not enc_ok:
                return []
            end_sec = duration if duration > 0 else 0.001
            return [
                {
                    "scene_index": 1,
                    "start_sec": 0.0,
                    "end_sec": round(end_sec, 3),
                    "frame_sec": round(frame_sec, 3),
                    "jpeg_bytes": encoded0.tobytes(),
                }
            ]
        finally:
            cap0.release()

    # 048: dedup off 면 현행 pre-cap(앞에서부터 자름) 유지 → 바이트 동일(FR-103). dedup on 이면
    # pre-cap 을 건너뛰고 전 장면을 추출한 뒤 아래에서 dedup → cap 순으로 trim 한다(FR-104).
    if not _dedup_on and max_frames is not None and max_frames > 0:
        scenes = scenes[:max_frames]

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {src}")

    results: list[KeyframeBytesResult] = []
    try:
        for i, (start_tc, end_tc) in enumerate(scenes, start=1):
            start_sec = _to_seconds(start_tc)
            end_sec = _to_seconds(end_tc)
            try:
                frame_sec, frame = _read_scene_mid_frame(cap, start_sec=start_sec, end_sec=end_sec)
            except RuntimeError:
                continue

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,  # type: ignore[arg-type]
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if not ok:
                continue

            results.append(
                {
                    "scene_index": i,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "frame_sec": round(frame_sec, 3),
                    "jpeg_bytes": encoded.tobytes(),
                }
            )
    finally:
        cap.release()

    # 048: dedup on 이면 전 장면 추출 결과에 near-dup 제거를 적용한 뒤(dedup) max_frames 로 trim(cap).
    # 순서 = dedup → cap(FR-104). off 면 위에서 이미 pre-cap 했으므로 results 를 그대로 반환(FR-103).
    if _dedup_on and dedup is not None:
        from src.preprocess.keyframe_dedup import dedup_keyframes

        kept, skips = dedup_keyframes(results, dedup)
        if skips:
            # FR-405·US4: skip 관측성 — 사유·개수(비용 절감 측정 SC-006 추적용). 디버그 레벨.
            logger.debug(
                "키프레임 dedup: %d/%d skip (mode=%s) — %s",
                len(skips),
                len(results),
                dedup.compare_mode,
                [{"scene": s["scene_index"], "reason": s["reason"]} for s in skips],
            )
        if max_frames is not None and max_frames > 0:
            kept = kept[:max_frames]
        return kept

    return results
