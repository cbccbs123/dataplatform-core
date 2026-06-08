"""일반 도메인 포탈 **백엔드 API** 패키지(spec 010, P1).

006 하이브리드 검색·005/008 관계 그래프 위에 검색→상세→다운로드(단일/관계묶음)
HTTP 백엔드를 올리는 조회 계층이다. 흐름은 순수 로직(검색 cursor 페이징·Range 파싱)과
conn 기반 조회 서비스(상세·묶음 수집/zip)로 나뉜다.

의존성
    FastAPI·psycopg 등 **무거운 의존성**은 진입점(``src/app/portal_api.py``)과 conn 기반
    조회 모듈에만 두고, ``from src.portal import X`` 는 ``__getattr__`` 로 **지연 import**
    한다(순환 참조·기동 비용 완화). ``src/relations/__init__.py`` 와 동형.

테스트
    순수 함수(``flatten_ranked``/``encode_cursor``/``decode_cursor``/``paginate``/
    ``parse_range_header``)는 DB 없이 ``from src.portal.search_page import ...`` 로 직접 import.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "flatten_ranked",
    "encode_cursor",
    "decode_cursor",
    "paginate",
    "parse_range_header",
]


def __getattr__(name: str) -> Any:
    """패키지 속성 지연 로드(``import src.portal as P; P.flatten_ranked`` 형태 지원)."""
    if name in ("flatten_ranked", "encode_cursor", "decode_cursor", "paginate"):
        from src.portal import search_page

        return getattr(search_page, name)
    if name == "parse_range_header":
        from src.portal.download import parse_range_header as fn

        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
