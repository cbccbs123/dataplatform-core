#!/usr/bin/env python3
"""OLD ``media_*`` → 신규 ``asset_*`` 데이터 마이그레이션.

철칙
    원본 ``media_items`` / ``media_chunks`` / ``media_relation`` 은 **SELECT 만** 한다.
    변경·추가·삭제를 절대 하지 않는다. 신규 ``asset_*`` 에만 INSERT.

특징
    - 전부 set-based ``INSERT … SELECT`` (Python 행 루프 없음).
    - 멱등: 모든 INSERT 는 ``ON CONFLICT DO NOTHING`` → 재실행 시 신규 0건.
    - 단일 트랜잭션(``PostgresUtil.transaction``). 실패 시 신규 적재만 롤백, 원본은 무영향.
    - 원본 id 보존: ``asset.asset_id = media_items.id``, ``asset_embedding.asset_id = media_item_id``.

매핑
    media_items   → asset (+ asset_metadata)
    media_chunks  → asset_embedding
    media_relation→ asset_relation

사용
    POSTGRES_* / DATABASE_URL 환경변수로 접속(PostgresUtil 규칙). 예:
        POSTGRES_HOST=localhost POSTGRES_PORT=5432 POSTGRES_DB=postgres \
        POSTGRES_USER=postgres POSTGRES_PASSWORD=*** python scripts/migrate_media_to_asset.py
    --dry-run 으로 적재 없이 사전/사후 건수만 비교(트랜잭션 롤백).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.postgres_util import PostgresUtil  # noqa: E402

_LOG = logging.getLogger("meta_extract.migrate_media_to_asset")

# metadata jsonb 에서 ext_meta(도메인 신호)로 보낼 키. 나머지는 core_meta(파일/시스템 속성).
_EXT_META_KEYS = ("summary", "keywords", "labels", "objects", "keyframes", "stt", "caption")

# 원본 무변경 증거용으로 캡처할 테이블.
_SOURCE_TABLES = ("media_items", "media_chunks", "media_relation")


def _configure_logging() -> None:
    if _LOG.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


def _source_fingerprint(conn: Connection[Any]) -> dict[str, dict[str, Any]]:
    """원본 테이블별 행수·max(id)·내용 md5 를 캡처(무변경 증거용, 읽기 전용)."""
    fp: dict[str, dict[str, Any]] = {}
    id_col = {
        "media_items": "id",
        "media_chunks": "id",
        "media_relation": "media_relation_id",
    }
    with conn.cursor(row_factory=dict_row) as cur:
        for t in _SOURCE_TABLES:
            idc = id_col[t]
            # to_jsonb(t.*) 를 id 순으로 직렬화해 md5. 내용이 바뀌면 해시가 달라진다.
            cur.execute(
                f"""
                SELECT count(*) AS n,
                       coalesce(max(j.id_val), 0) AS max_id,
                       md5(coalesce(string_agg(j.txt, '' ORDER BY j.txt), '')) AS digest
                FROM (
                    SELECT s.{idc} AS id_val, to_jsonb(s)::text AS txt FROM {t} AS s
                ) AS j
                """
            )
            row = cur.fetchone()
            assert row is not None
            fp[t] = {"n": int(row["n"]), "max_id": int(row["max_id"]), "digest": row["digest"]}
    return fp


def _target_counts(conn: Connection[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    with conn.cursor() as cur:
        for t in ("asset", "asset_metadata", "asset_embedding", "asset_relation"):
            cur.execute(f"SELECT count(*) FROM {t}")
            out[t] = int(cur.fetchone()[0])  # type: ignore[index]
    return out


def _migrate(conn: Connection[Any]) -> dict[str, int]:
    """신규 asset_* 에 set-based INSERT. 반환: 단계별 신규 적재 건수."""
    inserted: dict[str, int] = {}
    ext_keys = list(_EXT_META_KEYS)

    with conn.cursor() as cur:
        # 1) media_items → asset (id 보존)
        cur.execute(
            """
            INSERT INTO asset (asset_id, modality, fs_path, domain_label, status, created_at)
            SELECT mi.id, mi.media_type, mi.file_uri, 'general', 'registered',
                   coalesce(mi.created_at, now())
            FROM media_items mi
            ON CONFLICT (asset_id) DO NOTHING
            """
        )
        inserted["asset"] = cur.rowcount

        # 2) media_items.metadata → asset_metadata (core/ext 무손실 분리 + search_vector 복사)
        cur.execute(
            """
            INSERT INTO asset_metadata (asset_id, core_meta, ext_meta, tags, search_vector)
            SELECT mi.id,
                   coalesce((SELECT jsonb_object_agg(e.key, e.value)
                             FROM jsonb_each(mi.metadata) e
                             WHERE NOT (e.key = ANY(%(ext_keys)s))), '{}'::jsonb),
                   coalesce((SELECT jsonb_object_agg(e.key, e.value)
                             FROM jsonb_each(mi.metadata) e
                             WHERE e.key = ANY(%(ext_keys)s)), '{}'::jsonb),
                   '{}'::text[],
                   mi.search_vector
            FROM media_items mi
            ON CONFLICT (asset_id) DO NOTHING
            """,
            {"ext_keys": ext_keys},
        )
        inserted["asset_metadata"] = cur.rowcount

        # 3) media_chunks → asset_embedding (chunk_index 보존, channel=embedding_kind)
        cur.execute(
            """
            INSERT INTO asset_embedding (asset_id, channel, chunk_index, embedding, model_name)
            SELECT mc.media_item_id, mc.embedding_kind, mc.chunk_index, mc.embedding, 'legacy'
            FROM media_chunks mc
            WHERE mc.embedding IS NOT NULL
            ON CONFLICT (asset_id, channel, chunk_index) DO NOTHING
            """
        )
        inserted["asset_embedding"] = cur.rowcount

        # 4) media_relation → asset_relation (relation_type 카탈로그 공유)
        cur.execute(
            """
            INSERT INTO asset_relation (source_asset_id, target_asset_id, relation_type_id,
                                        confidence, reason, status, created_at, updated_at)
            SELECT mr.source_media_item_id, mr.target_media_item_id, mr.relation_type_id,
                   mr.confidence, mr.reason, mr.status, mr.created_at, mr.updated_at
            FROM media_relation mr
            ON CONFLICT (source_asset_id, target_asset_id, relation_type_id) DO NOTHING
            """
        )
        inserted["asset_relation"] = cur.rowcount

        # 5) id 보존했으므로 asset 시퀀스를 max(asset_id)로 재설정(이후 신규 INSERT 충돌 방지)
        cur.execute(
            """
            SELECT setval(pg_get_serial_sequence('asset', 'asset_id'),
                          GREATEST((SELECT coalesce(max(asset_id), 0) FROM asset), 1))
            """
        )

    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="media_* → asset_* 데이터 마이그레이션(원본 읽기 전용)")
    parser.add_argument("--dry-run", action="store_true", help="적재 후 롤백(건수만 확인)")
    args = parser.parse_args()

    _configure_logging()
    db = PostgresUtil()

    with db:
        with db.transaction() as conn:
            before_src = _source_fingerprint(conn)
            before_tgt = _target_counts(conn)
            _LOG.info("source(before)=%s", json.dumps(before_src, ensure_ascii=False))
            _LOG.info("target(before)=%s", json.dumps(before_tgt, ensure_ascii=False))

            inserted = _migrate(conn)
            _LOG.info("inserted=%s", json.dumps(inserted, ensure_ascii=False))

            after_src = _source_fingerprint(conn)
            after_tgt = _target_counts(conn)
            _LOG.info("target(after)=%s", json.dumps(after_tgt, ensure_ascii=False))

            # 원본 무변경 검증: fingerprint 가 동일해야 한다.
            if before_src != after_src:
                _LOG.error("원본 변경 감지! before=%s after=%s", before_src, after_src)
                raise RuntimeError("원본 media_* 가 변경되었습니다(있을 수 없는 상황). 롤백합니다.")
            _LOG.info("원본 무변경 확인 OK (fingerprint 동일)")

            if args.dry_run:
                # psycopg.Rollback 은 conn.transaction() 블록이 조용히 롤백·흡수하므로
                # 트레이스백 없이 정상 흐름으로 빠져나온다(적재만 취소).
                _LOG.info("--dry-run: 트랜잭션 롤백(적재 취소)")
                raise psycopg.Rollback()

    print(json.dumps({
        "dry_run": args.dry_run,
        "source_before": before_src,
        "target_before": before_tgt,
        "inserted": inserted,
        "target_after": after_tgt,
        "source_unchanged": before_src == after_src,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
