"""영상 키프레임 dedup 기본값 단일 출처(069 D2·P2-26 · 순수·의존 0).

``KeyframeDedupConfig`` dataclass 필드 기본값과 ``settings`` 의 ``VIDEO_KEYFRAME_DEDUP_*`` env
fallback 이 같은 리터럴을 각자 하드코딩하던 것을 여기 한 곳으로 모은다(SSOT — 한쪽만 바뀌어
드리프트하는 것을 차단). 값은 통합 전과 완전히 동일(동작 불변).

**왜 별도 경량 모듈인가**: 정본을 담는 ``keyframe_dedup.py`` 는 cv2·numpy(heavy)를 import 하는데,
``settings`` 는 어디서나 import 되는 경량 모듈이라 cv2 를 딸려오게 할 수 없다. 그래서 상수만 담은
의존 0 모듈을 두어 dataclass(무거워도 무방)와 settings(가벼워야 함) 양쪽이 안전하게 참조한다.
"""

from __future__ import annotations

DEFAULT_ENABLED = True
DEFAULT_HASH_MAX = 7  # dHash 해밍거리 임계(이하면 근접 프레임으로 간주)
DEFAULT_SSIM_MIN = 0.94  # 구조적 유사도 하한(이상이면 중복)
DEFAULT_SSIM_GRAY_LO = 0.90  # 저채도(그레이) 프레임의 완화된 SSIM 하한
DEFAULT_HIST_MIN = 0.97  # HSV 히스토그램 상관 하한
DEFAULT_COMPARE_MODE = "recent"  # "recent" | "last" | "global"
DEFAULT_RECENT_WINDOW = 4  # compare_mode="recent" 시 직전 N개와 비교

__all__ = [
    "DEFAULT_ENABLED",
    "DEFAULT_HASH_MAX",
    "DEFAULT_SSIM_MIN",
    "DEFAULT_SSIM_GRAY_LO",
    "DEFAULT_HIST_MIN",
    "DEFAULT_COMPARE_MODE",
    "DEFAULT_RECENT_WINDOW",
]
