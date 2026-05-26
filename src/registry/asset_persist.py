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
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.dispatch.types import AssetRecord
from src.ingest.status import AssetStatus, set_status


def create_asset(
    conn: Connection[Any],
    *,
    fs_path: str,
    modality: str,
    domain: str = "general",
    group_id: int | None = None,
    file_hash: str | None = None,
    file_size: int | None = None,
) -> int:
    """``asset`` 행을 ``received`` 상태로 INSERT 하고 ``asset_id`` 반환(모델 A 조기 INSERT)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO asset (group_id, modality, fs_path, file_hash, file_size, domain_label, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'received')
            RETURNING asset_id
            """,
            (group_id, modality, fs_path, file_hash, file_size, domain),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("asset INSERT 가 asset_id 를 반환하지 않았습니다.")
    return int(row["asset_id"])


def finalize_asset(conn: Connection[Any], asset_id: int, record: AssetRecord) -> None:
    """``AssetRecord`` 의 메타·임베딩을 적재하고 상태를 ``registered`` 로 전이한다.

    ``asset_metadata`` 1행(core/ext jsonb, tags, search_vector) + ``asset_embedding`` N행.
    마지막에 ``set_status(..., REGISTERED)`` (현재 상태가 ``extracting`` 이어야 함).
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
