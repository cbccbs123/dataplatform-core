"""파일명(basename) 단일 출처 — 검색 색인·검색결과 표시·샘플 API 공용(069 D3·순수·표준 라이브러리만).

경로/URI 에서 파일명만 뽑는 **공통 코어**다. 통합 전에는 ``search_group``·``sample_search_api``·
``opensearch_sync`` 가 각자 ``_basename`` 사본을 두었는데, 코어 로직이 동일해 여기 한 곳으로 모은다
(SSOT). ``src/config/__init__`` 이 비어 있어 heavy 의존(torch 등)을 끌지 않으므로, "표준 라이브러리만
import" 순수 계약인 세 호출처가 모두 안전하게 import 할 수 있다(순환/무거운 import 없음).

**책임 경계**: 이 함수는 asset_id 프리픽스(``{asset_id}__``)를 **벗기지 않는다**. 아카이브 프리픽스
제거는 표시 전용 책임(065 T605)이라 호출처(``search_group``·``archiver.display_file_name``)가
이 코어 위에 별도로 합성한다 — 색인·샘플 경로는 원본 파일명(프리픽스 포함)이 그대로 필요하기 때문.
"""

from __future__ import annotations


def basename_of(uri: str) -> str:
    """경로/URI 에서 파일명만 추출한다(결정적·순수). 쿼리(``?``)·프래그먼트(``#``)는 제거.

    백슬래시를 ``/`` 로 정규화하고 후행 ``/`` 를 제거한 뒤 마지막 세그먼트를 취한다. 쿼리/프래그먼트를
    떼고 남는 게 없으면(스킴/쿼리만 있는 입력) 마지막 세그먼트로 폴백한다(기존 3벌 공통 ``or tail``).
    asset_id 프리픽스는 벗기지 않는다(표시용 strip 은 별도 책임 — 모듈 docstring 참조).
    """
    if not uri:
        return ""
    tail = uri.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail.split("?", 1)[0].split("#", 1)[0] or tail
