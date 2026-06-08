"""포탈 다운로드 지원 — 단일 자산·관계 묶음(spec 010 D-4/D-5).

본 파일은 그룹별로 함수가 늘어난다:
    - **G2(이번)**: ``parse_range_header`` — HTTP Range 헤더 파싱(**완전 순수**, DB·IO 0).
    - G3(후속): ``resolve_download_target``/``collect_bundle_assets``/``build_bundle_zip``
      (conn·파일 IO). 같은 파일에 별도 task 로 추가 예정이므로 이번에는 순수 함수만 둔다.
"""

from __future__ import annotations

_BYTES_PREFIX = "bytes="


def parse_range_header(range_value: str | None, file_size: int) -> tuple[int, int] | None:
    """HTTP ``Range`` 헤더를 ``(start, end)`` 바이트 오프셋(둘 다 포함)으로 파싱한다(순수).

    지원 형식(단일 범위만):
        - ``bytes=start-end`` → ``(start, end)``
        - ``bytes=start-``    → ``(start, file_size-1)`` (열린 끝)
        - ``bytes=-suffix``   → ``(file_size-suffix, file_size-1)`` (마지막 suffix 바이트,
          suffix 가 파일보다 크면 전체로 클램프 — RFC 7233)
    헤더가 ``None`` 이면 ``None``(=전체 다운로드). 범위 위반(시작이 파일 크기 이상·끝이
    파일 크기 이상·역순)·형식 오류·다중 범위는 ``ValueError`` 로 거부한다(API 가 416 매핑,
    plan D-4: 범위 초과를 엄격히 거부).
    """
    if range_value is None:
        return None

    text = range_value.strip()
    if not text.startswith(_BYTES_PREFIX):
        raise ValueError(f"지원하지 않는 Range 단위: {range_value!r}")
    spec = text[len(_BYTES_PREFIX):].strip()

    # 다중 범위(콤마)는 본 MVP 미지원 — 단일 범위만 처리.
    if "," in spec:
        raise ValueError("다중 Range 는 미지원(단일 범위만)")
    if "-" not in spec:
        raise ValueError(f"Range 형식 오류: {range_value!r}")

    start_str, end_str = (part.strip() for part in spec.split("-", 1))
    try:
        if start_str == "":
            # 접미 형식 bytes=-suffix : 마지막 suffix 바이트.
            if end_str == "":
                raise ValueError("Range 형식 오류(빈 범위)")
            suffix = int(end_str)
            if suffix <= 0:
                raise ValueError("Range suffix 는 양수여야 함")
            start = max(0, file_size - suffix)
            end = file_size - 1
        else:
            start = int(start_str)
            end = int(end_str) if end_str != "" else file_size - 1
    except ValueError as exc:
        # int 변환 실패도 형식 오류로 통일(416 의미는 아래 범위 검증에서).
        raise ValueError(f"Range 형식 오류: {range_value!r} ({exc})") from exc

    if start < 0:
        raise ValueError(f"Range 시작이 음수: {range_value!r}")
    if start >= file_size:
        raise ValueError(f"Range 시작이 파일 크기 이상(416): {start} >= {file_size}")
    if end < start:
        raise ValueError(f"Range 역순(416): {start} > {end}")
    if end >= file_size:
        raise ValueError(f"Range 끝이 파일 크기 이상(416): {end} >= {file_size}")
    return (start, end)
