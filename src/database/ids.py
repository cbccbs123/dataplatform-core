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
    """현재 시각 기반 UUIDv7 한 개 생성."""
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
    """UUIDv7 문자열."""
    return str(uuid7())
