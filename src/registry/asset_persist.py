"""F-1.3 자산 등록·적재 (모델 A).

``run_extract_meta.py`` 의 ``media_items``/``media_chunks`` 직접 INSERT 를 신규
``asset``/``asset_metadata``/``asset_embedding`` 로 재배선한 통일 영속화 계층.

모델 A 분리
    - ``create_asset``: 파일 픽업 직후 ``asset`` 행을 ``received`` 로 조기 INSERT(asset_id 확보).
    - ``finalize_asset``: 추출 결과(``AssetRecord``)의 메타·임베딩을 적재하고 상태를 ``registered`` 로.
      (호출 전 상태가 ``extracting`` 이어야 함 — 상태 머신 검증)

두 함수 모두 psycopg ``Connection`` 을 받아 오케스트레이터가 트랜잭션 경계를 제어한다
(단계별 짧은 트랜잭션 + 실패 시 fresh 트랜잭션으로 mark_failed — T1-6 참고).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.database.ids import uuid7
from src.dispatch.types import AssetRecord
from src.ingest.status import AssetStatus, set_status


def find_registered_asset_by_hash(conn: Connection[Any], file_hash: str) -> uuid.UUID | None:
    """동일 내용(file_hash)으로 이미 ``registered`` 된 자산의 asset_id. 없으면 None(중복 적재 방지용).

    run_ingest 가 파일 픽업 직후 이 함수로 중복을 검사해, 기존 자산이 있으면 파이프라인을 건너뛴다.
    ``failed``/``deferred`` 상태의 같은 해시는 중복으로 보지 않아 재처리를 허용한다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT asset_id FROM asset WHERE file_hash = %s AND status = 'registered' LIMIT 1",
            (file_hash,),
        )
        row = cur.fetchone()
    return row["asset_id"] if row else None


def create_asset(
    conn: Connection[Any],
    *,
    fs_path: str,
    modality: str,
    domain: str = "general",
    file_hash: str | None = None,
    file_size: int | None = None,
) -> uuid.UUID:
    """``asset`` 행을 ``received`` 상태로 INSERT 하고 asset_id(UUIDv7) 반환(모델 A 조기 INSERT).

    식별자는 앱에서 UUIDv7 로 생성해 명시적으로 INSERT 한다(PG17 네이티브 uuidv7() 부재).
    """
    asset_id = uuid7()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO asset (asset_id, modality, fs_path, file_hash, file_size, domain_label, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'received')
            """,
            (asset_id, modality, fs_path, file_hash, file_size, domain),
        )
    return asset_id


def finalize_asset(conn: Connection[Any], asset_id: uuid.UUID, record: AssetRecord) -> None:
    """``AssetRecord`` 의 메타·임베딩을 적재하고 상태를 ``registered`` 로 전이한다.

    ``asset_metadata`` 1행(core/ext jsonb, tags, search_vector) + ``asset_embedding`` N행을 삽입.
    마지막에 ``set_status(..., REGISTERED)`` — 현재 상태가 ``extracting`` 이어야 한다(상태 머신 검증).

    임베딩이 없을 때(record.embeddings 빈 리스트) executemany 를 건너뛴다.
    벡터는 항상 ``FIX_EMBEDDING_DIMENSION``(1536D) ::vector 캐스트로 저장 — DB CHECK 제약과 일치해야 한다.
    search_vector 는 'simple' 사전으로 FTS 인덱스 생성(언어 무관 토크나이징).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO asset_metadata (asset_id, core_meta, ext_meta, tags, search_vector)
            VALUES (%s, %s::jsonb, %s::jsonb, %s, to_tsvector('simple', coalesce(%s, '')))
            """,
            (
                asset_id,
                json.dumps(record.core_meta, ensure_ascii=False),
                json.dumps(record.ext_meta, ensure_ascii=False),
                list(record.tags),
                record.fts_plain,
            ),
        )
        if record.embeddings:
            # 채널(st/clip)·청크별 1행씩 bulk INSERT — 같은 트랜잭션 안에서 메타와 함께 커밋
            cur.executemany(
                f"""
                INSERT INTO asset_embedding (asset_id, channel, chunk_index, embedding, model_name, model_version)
                VALUES (%s, %s, %s, %s::vector({FIX_EMBEDDING_DIMENSION}), %s, %s)
                """,
                [
                    (asset_id, e.channel, e.chunk_index, e.vector, e.model_name, e.model_version)
                    for e in record.embeddings
                ],
            )
    set_status(conn, asset_id, AssetStatus.REGISTERED)
