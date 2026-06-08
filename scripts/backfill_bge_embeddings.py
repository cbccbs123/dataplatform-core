#!/usr/bin/env python3
"""BGE-M3 임베딩 백필 — 기존 ``channel='st'`` 자산에 ``channel='st_bge'`` 행을 추가한다 (017 G2).

017 A/B(KoSimCSE vs BGE-M3)는 **같은 자산을 두 채널로 임베딩**해 한국어 retrieval 품질을 비교한다.
이 스크립트는 이미 ``channel='st'``(KoSimCSE) 행이 있는 registered 자산의 본문을 ``fs_path``/
``modality`` 로 재로딩해 BGE-M3 임베딩을 **별도 채널 행**(``channel='st_bge'``)으로 추가한다.

설계 불변식
  - **스키마 무변경(헌법 6조)**: 새 컬럼·마이그레이션·DDL 0. 채널 행만 INSERT 한다.
    기존 ``'st'`` 행은 무변경. A/B 종료 후 정리는 ``DELETE WHERE channel='st_bge'``(DDL 0).
  - **결정성(헌법 3조)**: 본문 청킹·정규화·normalize 를 기존 ingest 와 동일하게 재사용하고
    ``ON CONFLICT (asset_id, channel, chunk_index) DO NOTHING`` 으로 멱등화 → 2회 실행 동일.
  - **학습 0(헌법 1조)**: BGE-M3 는 SentenceTransformer inference only(LLM seam 무관).
  - **배치 격리(FR-002)**: 본문 누락·로딩 실패·자산별 예외는 skip + 로그로 흡수해 배치를
    멈추지 않는다. 처리/skip/청크 카운트를 보고한다.

본문 재로딩 분기(기존 ingest 와 동일 경로 — text_skill/audio_skill)
  - 문서(txt/pdf/json/word/excel/powerpoint) → ``embedding_text_chunks(fs_path, file_kind=modality)``.
  - 오디오(STT) → 전사 사이드카 ``<fs_path>.stt.txt`` 를 plain 으로 읽어
    ``embedding_plain_text_chunks`` (whisper 재실행 금지 — 결정성·비용).
  - 시각(image/video)의 ``'st'``(VLM 텍스트)는 추출 시점 파생물이라 ``fs_path`` 에서
    재구성 불가 → skip(plan §6 후속). unknown 등도 skip.

두 ``embedding_*`` 헬퍼는 내부에서 ``normalize_text_for_embedding`` → ``embed_texts(model_name)``
→ ``pad_embedding_to_storage_dim``(1024→1536) 을 수행하므로, 여기서는 ``model_name`` 만 BGE 로
바꿔 호출하면 같은 청킹으로 BGE 벡터를 얻는다(``'st'`` 와 chunk_index 정합).

실행
    conda activate AuroraFS
    python scripts/backfill_bge_embeddings.py --env dev            # 전체
    python scripts/backfill_bge_embeddings.py --env dev --limit 50 # 단계적(자원·시간 분할)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.embedders.text_embedder import (
    embedding_plain_text_chunks,
    embedding_text_chunks,
)
from src.file.data_loader import normalize_file_kind

_BGE_CHANNEL = "st_bge"

_LOG = logging.getLogger("meta_extract.backfill_bge")

# 멱등 INSERT — (asset_id, channel, chunk_index) PK 충돌은 무시(2회 실행 중복 0, SC-001).
_INSERT_SQL = (
    f"INSERT INTO asset_embedding "
    f"(asset_id, channel, chunk_index, embedding, model_name, model_version) "
    f"VALUES (%s, %s, %s, %s::vector({FIX_EMBEDDING_DIMENSION}), %s, %s) "
    f"ON CONFLICT (asset_id, channel, chunk_index) DO NOTHING"
)


def _configure_logging() -> None:
    if _LOG.handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


def _fetch_st_assets(conn: Any, *, limit: int | None = None) -> list[dict[str, Any]]:
    """``channel='st'`` 행이 있는 registered 자산의 asset_id·fs_path·modality 목록.

    DISTINCT 로 자산당 1행. ``ORDER BY asset_id`` 로 결정적 순서(--limit 단계적 백필이
    매 실행 같은 앞부분을 보도록). ``limit`` 지정 시 ``LIMIT`` 절을 덧붙인다.
    """
    sql = (
        "SELECT DISTINCT a.asset_id, a.fs_path, a.modality "
        "FROM asset a "
        "JOIN asset_embedding e ON e.asset_id = a.asset_id AND e.channel = 'st' "
        "WHERE a.status = 'registered' "
        "ORDER BY a.asset_id"
    )
    params: list[Any] = []
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def _reload_chunks(
    fs_path: str,
    modality: str,
    *,
    model_name: str,
    chunk_size: int,
    encoding: str,
    normalize: bool,
) -> list[dict[str, Any]] | None:
    """``modality`` 로 본문을 재로딩해 BGE 임베딩 청크([{chunk_index, embedding_vector}])를 반환.

    재로딩 불가(사이드카 없음·미지원 modality)면 ``None`` 을 반환해 호출부가 skip 한다.
    파일 부재 등 로딩 예외는 그대로 올려보내 ``backfill_asset`` 이 흡수(skip)한다.
    """
    # 문서 file_kind(txt/pdf/json/word/excel/powerpoint) → 파일에서 직접 청킹+임베딩.
    file_kind = _document_file_kind(modality)
    if file_kind is not None:
        return embedding_text_chunks(
            fs_path,
            file_kind=file_kind,
            encoding=encoding,
            chunk_size=chunk_size,
            embedding_model_name=model_name,
            normalize_embeddings=normalize,
        )
    # 오디오: STT 전사 사이드카(plain)를 재사용(whisper 재실행 금지).
    if modality == "audio":
        text = _read_stt_sidecar(fs_path)
        if text is None:
            return None
        return embedding_plain_text_chunks(
            text,
            chunk_size=chunk_size,
            embedding_model_name=model_name,
            normalize_embeddings=normalize,
        )
    # 시각(image/video)의 'st'(VLM 텍스트)·unknown 등은 fs_path 에서 본문 재구성 불가.
    return None


def _document_file_kind(modality: str) -> str | None:
    """modality 가 문서 file_kind 면 정규화 값, 아니면 None(오디오/시각 등)."""
    try:
        return normalize_file_kind(modality)
    except ValueError:
        return None


def _read_stt_sidecar(fs_path: str) -> str | None:
    """오디오 자산의 STT 전사 사이드카(``<...>.stt.txt``)를 읽는다(없으면 None).

    수집기가 남긴 전사 텍스트 규약(`.stt.txt`)을 두 가지 흔한 위치로 탐색한다:
      1) ``<fs_path>.stt.txt`` (예: ``a.mp3.stt.txt``)
      2) ``<stem>.stt.txt``    (예: ``a.stt.txt``)
    둘 다 없으면 None(전사 부재 → skip). whisper 재실행은 비용·결정성 때문에 하지 않는다.
    """
    p = Path(fs_path)
    candidates = [
        p.with_name(p.name + ".stt.txt"),
        p.with_suffix(".stt.txt"),
    ]
    for c in candidates:
        if c.is_file():
            return c.read_text(encoding="utf-8", errors="replace")
    return None


def backfill_asset(
    conn: Any,
    asset_row: dict[str, Any],
    *,
    model_name: str,
    chunk_size: int,
    encoding: str,
    normalize: bool,
) -> int:
    """한 자산의 BGE 임베딩을 ``channel='st_bge'`` 행으로 적재. 반환=계산한 청크 수(0=skip).

    본문 누락·로딩 실패(파일 부재 등)는 예외를 전파하지 않고 skip + 로그(0 반환)한다 —
    한 자산의 본문 문제가 배치를 멈추면 안 되기 때문(FR-002). INSERT 는 ``ON CONFLICT
    DO NOTHING`` 으로 멱등하므로 같은 자산을 2회 호출해도 중복 행이 생기지 않는다.
    """
    asset_id = asset_row["asset_id"]
    fs_path = asset_row["fs_path"]
    modality = asset_row["modality"]
    try:
        chunks = _reload_chunks(
            fs_path,
            modality,
            model_name=model_name,
            chunk_size=chunk_size,
            encoding=encoding,
            normalize=normalize,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        _LOG.warning("skip(본문 로딩 실패) asset_id=%s %s: %s", asset_id, fs_path, exc)
        return 0
    if not chunks:
        _LOG.info("skip(본문 재로딩 불가) asset_id=%s modality=%s %s", asset_id, modality, fs_path)
        return 0

    rows = [
        (asset_id, _BGE_CHANNEL, int(c["chunk_index"]), c["embedding_vector"], model_name, None)
        for c in chunks
    ]
    with conn.cursor() as cur:
        cur.executemany(_INSERT_SQL, rows)
    return len(rows)


def run_backfill_db(
    db: Any,
    *,
    model_name: str,
    chunk_size: int,
    encoding: str,
    normalize: bool,
    limit: int | None = None,
) -> dict[str, int]:
    """대상 자산을 조회해 자산별 트랜잭션으로 백필한다(자산 단위 격리).

    자산마다 독립 트랜잭션을 열어, 한 자산의 INSERT/임베딩 실패가 다른 자산에 영향을 주지
    않게 한다(run_ingest 의 파일 단위 격리와 동일 사상). 반환: 처리/skip/청크 카운트.
    """
    with db.transaction() as conn:
        rows = _fetch_st_assets(conn, limit=limit)
    _LOG.info("backfill 대상: %s건 (channel='st' & registered)", len(rows))

    processed = skipped = chunks = 0
    for row in rows:
        try:
            with db.transaction() as conn:
                n = backfill_asset(
                    conn,
                    row,
                    model_name=model_name,
                    chunk_size=chunk_size,
                    encoding=encoding,
                    normalize=normalize,
                )
        except Exception as exc:  # noqa: BLE001 — 자산 단위 격리(임베딩/INSERT 모든 실패 흡수)
            _LOG.warning("skip(예외) asset_id=%s: %s: %s", row.get("asset_id"), type(exc).__name__, exc)
            skipped += 1
            continue
        if n <= 0:
            skipped += 1
        else:
            processed += 1
            chunks += n
    _LOG.info("backfill done: processed=%s skipped=%s chunks=%s", processed, skipped, chunks)
    return {"processed": processed, "skipped": skipped, "chunks": chunks}


# ── 부트스트랩(run_ingest/run_search 와 동일 순서) ──────────────────────────────
# 1) load_dotenv(.env.{env}, override=False) → 2) init_settings(env)(필수 env 검증) →
# 3) PostgresUtil() + `with db:` → 4) run_backfill_db. 임베딩 모델은 첫 사용 시 지연 로딩.
def main() -> int:
    import json

    from dotenv import load_dotenv

    from src.config.settings import get_current_settings, init_settings
    from src.database.postgres_util import PostgresUtil

    parser = argparse.ArgumentParser(description="BGE-M3 임베딩 백필 (channel='st_bge' 추가)")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="처리 자산 수 상한(단계적 백필)")
    args = parser.parse_args()

    _configure_logging()
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)
    cfg = get_current_settings()

    db = PostgresUtil()
    with db:
        result = run_backfill_db(
            db,
            model_name=cfg.text_embedding_model_bge,
            chunk_size=cfg.text_embedding_chunk_size,
            encoding=cfg.encoding,
            normalize=cfg.text_embedding_normalize,
            limit=args.limit,
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
