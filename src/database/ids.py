"""UUIDv7 생성기 (RFC 9562, 순수 파이썬 — 외부 의존성 없음).

PK 식별자에 시간순 정렬 UUID(v7)를 사용해 전역 유일성 + B-tree 인덱스 지역성을 얻는다.
PostgreSQL 17 은 네이티브 ``uuidv7()`` 가 없어(=PG18+) 애플리케이션에서 생성한다.

레이아웃(128비트): 48b unix_ts_ms | 4b version(0111) | 12b rand_a | 2b variant(10) | 62b rand_b
"""

from __future__ import annotations

import os
import time
import uuid


def uuid7() -> uuid.UUID:
    """현재 시각을 앞자리에 담은 UUIDv7 을 하나 만든다.

    UUIDv4(완전 난수) 대신 v7 을 쓰는 이유: **만든 순서대로 정렬된다**. PK 로 쓰면 인덱스가
    한쪽 끝에만 쌓여 조각화가 적고, id 만 보고도 생성 시점 순서를 알 수 있다.

    Returns:
        UUIDv7 객체(밀리초 타임스탬프 48비트 + 난수).
    """
    ts_ms = int(time.time() * 1000)
    rand = os.urandom(10)  # rand_a(2B) + variant/rand_b(8B) 재료

    b = bytearray(16)
    b[0:6] = ts_ms.to_bytes(6, "big")  # 48b 타임스탬프

    rand_a = int.from_bytes(rand[0:2], "big") & 0x0FFF  # 12b
    b[6] = 0x70 | (rand_a >> 8)  # version 7
    b[7] = rand_a & 0xFF

    b[8] = 0x80 | (rand[2] & 0x3F)  # variant 10 + 6b rand_b
    b[9:16] = rand[3:10]  # 나머지 56b rand_b

    return uuid.UUID(bytes=bytes(b))


def uuid7_str() -> str:
    """UUIDv7 을 문자열로 만든다.

    DB 조회 결과를 다룰 때 id 를 문자열로 통일하는 관례가 있어(비교·JSON 직렬화 일관성),
    새 id 를 만들 때도 대부분 이쪽을 쓴다.

    Returns:
        하이픈 포함 UUID 문자열.
    """
    return str(uuid7())
