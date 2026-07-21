"""파일명(basename) 단일 출처 — 검색 색인·검색결과 표시·샘플 API 공용(069 D3·순수·표준 라이브러리만).

경로/URI 에서 파일명만 뽑는 **공통 코어**다. 통합 전에는 ``search_group``·``sample_search_api``·
``opensearch_sync`` 가 각자 ``_basename`` 사본을 두었는데, 코어 로직이 동일해 여기 한 곳으로 모은다
(SSOT). ``src/config/__init__`` 이 비어 있어 heavy 의존(torch 등)을 끌지 않으므로, "표준 라이브러리만
import" 순수 계약인 세 호출처가 모두 안전하게 import 할 수 있다(순환/무거운 import 없음).

**책임 경계**: ``basename_of`` 는 asset_id 프리픽스(``{asset_id}__``)를 **벗기지 않는다**(색인·샘플 경로는
원본 파일명이 그대로 필요). 표시 전용 프리픽스 제거(065 T605)는 ``strip_asset_id_prefix``/``display_file_name``
가 담당한다 — 077 레포 분리에서 백엔드가 쓰는 표시 유틸을 코어(config)로 승격했다(종전 ``ingest.archiver``).
"""

from __future__ import annotations

import os
import re

# registered_dest 가 붙이는 ``{asset_id}__{원본명}`` 프리픽스(UUIDv7 + '__')의 역패턴.
# 표시용 파일명 산출 시 이 프리픽스만 벗겨 원본 파일명을 복원한다(아카이브 이동으로 fs_path 가
# asset_id 프리픽스를 갖게 돼도 프론트·다운로드엔 원본명만 보이게 — 065 T605).
_ASSET_ID_PREFIX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}__"
)


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


def strip_asset_id_prefix(name: str) -> str:
    """basename 앞의 ``{asset_id}__`` 프리픽스 제거(순수·결정적). ``registered_dest`` 역함수.

    프리픽스가 없으면(인입 원본 등) 원본 그대로 반환한다. UUID 프리픽스만 정확 매칭하므로
    원본명에 ``__`` 가 있어도(맨 앞이 UUID 형태가 아니면) 건드리지 않는다.
    """
    return _ASSET_ID_PREFIX.sub("", name or "")


def display_file_name(fs_path: str | None) -> str:
    """fs_path → 표시용 파일명 = basename 에서 archiver 프리픽스(``{asset_id}__``) 제거(순수).

    아카이브 이동 후 fs_path 가 ``.../{asset_id}__{원본명}`` 이어도 원본 파일명만 돌려준다.
    프론트 트리·자산 목록·다운로드 파일명이 asset_id 를 노출하지 않게 하는 단일 출처(065 T605).
    """
    if not fs_path:
        return ""
    return strip_asset_id_prefix(os.path.basename(fs_path))
