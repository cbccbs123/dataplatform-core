"""자산 썸네일 생성 (057-후속) — 이미지·영상 시각 미리보기.

원본(``fs_path``)에서 축소 썸네일 JPEG 바이트를 만든다. **읽기 전용**(원본·DB 무수정, 헌법 6조)·
**결정적**(같은 파일 → 같은 바이트, 헌법 3조)·**학습/LLM 0**(단순 이미지 리사이즈·프레임 추출·헌법 1·2조).

- **이미지**: PIL 로 열어 EXIF 회전 반영 후 최대 변 ``THUMB_MAX_DIM`` 으로 축소.
- **영상**: cv2 로 **대표 프레임 1개**(1초 지점·검은 첫 프레임 회피·결정적) 추출 후 동일 축소.
- **오디오/텍스트/unknown**: 시각 표현이 없어 ``None`` → 엔드포인트 404 → 프론트가 모달리티 아이콘 폴백.

의존(cv2·PIL)은 **함수 내부 지연 import**(모듈 순수성 — 미사용 환경 import 부담 0). **의료(PHI) 배제는
엔드포인트 게이트**(``resolve_download_target`` registered·비의료)가 담당하므로 여기선 modality 만 본다.
서버측 캐시는 두지 않는다(브라우저 ``Cache-Control`` 로 완화; 디스크/LRU 캐시는 후속 최적화).
"""
from __future__ import annotations

import logging
from typing import Any

_LOG = logging.getLogger("meta_extract.thumbnail")

THUMB_MAX_DIM = 320  # 썸네일 최대 변(px) — 목록 카드용 소형
THUMBNAILABLE_MODALITIES = frozenset({"image", "video"})
_VIDEO_POS_MSEC = 1000.0  # 대표 프레임 위치(1초) — 0초는 검은 프레임이 흔해 회피(결정적)


def _encode_thumb(pil_img: Any) -> bytes:
    """PIL 이미지 → 축소 JPEG 바이트(순수·결정적). EXIF 회전 반영·RGB·LANCZOS 리샘플."""
    from io import BytesIO

    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(pil_img) or pil_img
    img = img.convert("RGB")
    img.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def generate_thumbnail(fs_path: str | None, modality: str | None) -> bytes | None:
    """원본 ``fs_path`` → 축소 JPEG 바이트(결정적). 비대상 modality·실패·손상 파일 → ``None``.

    어떤 예외도 전파하지 않는다(손상 파일·미지원 코덱 등은 썸네일 없음=404 로 격리). 결정성:
    이미지 리사이즈·영상 고정 위치 프레임·고정 JPEG 파라미터라 동일 입력 → 동일 출력.
    """
    if modality not in THUMBNAILABLE_MODALITIES or not fs_path:
        return None
    try:
        if modality == "image":
            from PIL import Image

            with Image.open(fs_path) as im:
                return _encode_thumb(im)
        # video: cv2 로 대표 프레임 1개(1초 지점·실패 시 첫 프레임)
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(fs_path)
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, _VIDEO_POS_MSEC)
            ok, frame = cap.read()
            if not ok or frame is None:  # 1초 지점 실패 → 첫 프레임 폴백
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or frame is None:
                return None
        finally:
            cap.release()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return _encode_thumb(Image.fromarray(rgb))
    except Exception as exc:  # noqa: BLE001 — 손상 파일·코덱 등은 썸네일 없음(404)으로 격리(best-effort)
        _LOG.warning("썸네일 생성 실패(무시): fs_path=%s modality=%s: %s", fs_path, modality, exc)
        return None
