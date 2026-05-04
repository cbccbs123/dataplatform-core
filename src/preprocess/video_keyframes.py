from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict

import cv2
from scenedetect import ContentDetector, detect


class KeyframeResult(TypedDict):
    """장면별 대표 프레임 저장 결과."""

    scene_index: int
    start_sec: float
    end_sec: float
    frame_sec: float
    frame_path: str


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


def extract_video_representative_frames(
    video_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    threshold: float = 30.0,
    min_scene_len: int = 15,
    jpeg_quality: int = 90,
) -> list[KeyframeResult]:
    """
    영상에서 장면(Scene) 단위 대표 프레임(중앙 시점) 1장씩 추출한다.

    :param video_path: 입력 비디오 파일 경로.
    :param output_dir: 저장 디렉토리. None이면 `<영상명>_keyframes` 폴더 생성.
    :param threshold: PySceneDetect ContentDetector 임계값.
    :param min_scene_len: 최소 장면 길이(프레임 단위).
    :param jpeg_quality: 저장 JPEG 품질(0~100).
    :return: 장면별 대표 프레임 메타데이터 목록.
    """
    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(str(src))

    dst = Path(output_dir) if output_dir is not None else src.with_name(f"{src.stem}_keyframes")
    dst.mkdir(parents=True, exist_ok=True)

    scenes = detect(str(src), ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    if not scenes:
        return []

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {src}")

    results: list[KeyframeResult] = []
    try:
        for i, (start_tc, end_tc) in enumerate(scenes, start=1):
            start_sec = _to_seconds(start_tc)
            end_sec = _to_seconds(end_tc)
            frame_sec = start_sec + max(0.0, (end_sec - start_sec) / 2.0)

            cap.set(cv2.CAP_PROP_POS_MSEC, frame_sec * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame_path = dst / f"scene_{i:04d}_{frame_sec:.2f}s.jpg"
            write_ok = cv2.imwrite(
                str(frame_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if not write_ok:
                continue

            results.append(
                {
                    "scene_index": i,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "frame_sec": round(frame_sec, 3),
                    "frame_path": str(frame_path),
                }
            )
    finally:
        cap.release()

    return results


def extract_video_representative_frame_bytes(
    video_path: str | Path,
    *,
    threshold: float = 30.0,
    min_scene_len: int = 15,
    jpeg_quality: int = 85,
    max_frames: int | None = None,
) -> list[KeyframeBytesResult]:
    """
    영상에서 장면(Scene) 단위 대표 프레임(중앙 시점) JPEG bytes를 반환한다.

    파일로 저장하지 않고 메모리에서 바로 후속 요약 파이프라인에 연결할 때 사용한다.
    """
    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(str(src))

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

    if max_frames is not None and max_frames > 0:
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

    return results
