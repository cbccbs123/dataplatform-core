"""임베더 프레임 입력 타입(공유 계약) — preprocess 가 생산, embedders 가 소비.

078 코어 물리 분리: ``KeyframeBytesResult`` 는 preprocess(파이프라인 · video_keyframes)가 만들어
embedders(코어 · video_embedder)가 소비하는 **공유 데이터 계약**이다. 코어가 파이프라인을 역참조하지
않도록(코어→파이프라인 import 0), 이 계약 타입을 **코어 쪽(경량·표준 라이브러리만)**에 둔다 —
preprocess(생산)·embedders(소비) 양쪽이 여기서 import 한다(SSOT 유지·의존 방향 정상: 파이프라인→코어).
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class KeyframeBytesResult(TypedDict):
    """장면별 대표 프레임(메모리 JPEG) 결과."""

    scene_index: int
    start_sec: float
    end_sec: float
    frame_sec: float
    jpeg_bytes: bytes
    summary: NotRequired[dict[str, str | list[str]]]
