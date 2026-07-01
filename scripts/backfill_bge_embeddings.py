#!/usr/bin/env python3
"""BGE-M3 임베딩 백필 — 기존 ``channel='st'`` 자산에 ``channel='st_bge'`` 행을 추가한다 (017 G2).

017 A/B(KoSimCSE vs BGE-M3)는 **같은 자산을 두 채널로 임베딩**해 한국어 retrieval 품질을 비교한다.
이 스크립트는 이미 ``channel='st'``(KoSimCSE) 행이 있는 registered 자산의 **'st' 본문을
modality 별로 재현**해 BGE-M3 임베딩을 **별도 채널 행**(``channel='st_bge'``)으로 추가한다.

핵심: image/video 의 ``'st'`` 임베딩은 ingest 가 ``ext_meta`` 의 VLM 텍스트로 만든 것이고,
audio 의 ``'st'`` 는 STT 텍스트로 만든 것이다. 그 산출물은 모두 DB ``ext_meta``(또는 사이드카)
에 남아 있으므로, **VLM/CLIP/whisper 를 재실행하지 않고** 본문을 그대로 재현해 BGE 로 임베딩한다.

설계 불변식
  - **스키마 무변경(헌법 6조)**: 새 컬럼·마이그레이션·DDL 0. 채널 행만 INSERT 한다.
    기존 ``'st'`` 행은 무변경. A/B 종료 후 정리는 ``DELETE WHERE channel='st_bge'``(DDL 0).
  - **결정성(헌법 3조)**: 본문 청킹·정규화·normalize 를 ingest ``_embed_*`` 와 동일하게 재사용하고
    ``ON CONFLICT (asset_id, channel, chunk_index) DO NOTHING`` 으로 멱등화 → 2회 실행 동일.
  - **학습 0(헌법 1조)**: BGE-M3 는 SentenceTransformer inference only(LLM seam 무관).
  - **배치 격리(FR-002)**: 본문 누락·로딩 실패·자산별 예외는 skip + 로그로 흡수해 배치를
    멈추지 않는다. 처리/skip/청크 카운트를 보고한다.

본문 재현 분기(ingest ``_embed_*`` 와 동일 입력 → chunk_index 정합 → 공정 비교)
  - **문서**(txt/json/pdf/word/excel/powerpoint) → ``embedding_text_chunks(fs_path, file_kind=…)``.
    파일 본문을 직접 청킹+임베딩(text_skill 과 동일).
  - **image** → ``build_image_vlm_text_for_embedding(core_meta|ext_meta)`` 1청크(image_skill
    ``_embed_image`` 와 동일). summary+keywords+labels 가 없어 본문이 공백뿐이면 skip.
  - **video** → ``ext_meta.keyframes`` 프레임별 ``build_image_vlm_text_for_embedding`` 다청크
    (video_skill ``_embed_video`` 와 동일; chunk_index=프레임 순번). keyframes 가 없으면
    자산 summary 1청크로 폴백, 그것도 없으면 skip.
  - **audio** → ``ext_meta.stt``(없으면 ``<fs_path>.stt.txt`` 사이드카) → ``embedding_plain_text_chunks``
    (audio_skill ``_embed_audio`` 와 동일). 둘 다 없으면 skip(whisper 재실행 금지).

실행
    conda activate AuroraFS
    python scripts/backfill_bge_embeddings.py --env dev            # 전체
    python scripts/backfill_bge_embeddings.py --env dev --limit 50 # 단계적(자원·시간 분할)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# scripts/ 직접 실행(python scripts/backfill_bge_embeddings.py) 시 'src' 패키지를 찾도록
# 저장소 루트를 sys.path 에 추가한다(PYTHONPATH=. 없이도 동작). src 임포트보다 먼저 와야 한다.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from psycopg.rows import dict_row  # noqa: E402 — sys.path 부트스트랩 뒤에 와야 함

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION  # noqa: E402
from src.embedders.text_embedder import (  # noqa: E402
    embed_texts,
    embedding_plain_text_chunks,
    embedding_text_chunks,
    pad_embedding_to_storage_dim,
)
from src.file.data_loader import normalize_file_kind  # noqa: E402
from src.file.file_type_detector import detect_file_kind  # noqa: E402
from src.preprocess.vlm_text_for_embedding import build_image_vlm_text_for_embedding  # noqa: E402

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
    """``channel='st'`` 행이 있는 registered 자산의 asset_id·fs_path·modality·core/ext_meta 목록.

    image/video/audio 의 'st' 본문은 ``fs_path`` 가 아니라 ``ext_meta``(VLM/STT 산출물)에서
    재현하므로 ``asset_metadata`` 를 함께 조회한다. LEFT JOIN 으로 메타 행이 없는 자산도
    빠뜨리지 않는다(문서는 메타 없이 fs_path 만으로 재현 가능). DISTINCT 로 자산당 1행,
    ``ORDER BY asset_id`` 로 결정적 순서(--limit 단계적 백필이 매 실행 같은 앞부분을 보도록).
    """
    sql = (
        "SELECT DISTINCT a.asset_id, a.fs_path, a.modality, m.core_meta, m.ext_meta "
        "FROM asset a "
        "JOIN asset_embedding e ON e.asset_id = a.asset_id AND e.channel = 'st' "
        "LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id "
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


def _document_file_kind(modality: str) -> str | None:
    """modality 가 문서 file_kind(txt/json/pdf/word/excel/powerpoint)면 정규화 값, 아니면 None."""
    try:
        return normalize_file_kind(modality)
    except ValueError:
        return None


def _read_stt_sidecar(fs_path: str) -> str | None:
    """오디오 자산의 STT 전사 사이드카(``<...>.stt.txt``)를 읽는다(없으면 None).

    ``ext_meta.stt`` 가 없을 때의 폴백. 수집기가 남긴 전사 규약(``.stt.txt``)을 두 위치로 탐색한다:
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


def _embed_one_text(text: str, *, model_name: str, normalize: bool) -> list[dict[str, Any]]:
    """단일 텍스트를 1청크(chunk_index=0)로 BGE 임베딩한다(image·video 폴백 공용).

    본문이 공백뿐이면 ``None`` 대신 빈 리스트를 반환해 호출부가 skip 하도록 한다
    (garbage(영벡터) 임베딩을 채널에 남기지 않기 위함).
    """
    if not text.strip():
        return []
    raw = embed_texts([text], model_name=model_name, normalize_embeddings=normalize)[0]
    return [{"chunk_index": 0, "embedding_vector": pad_embedding_to_storage_dim(raw)}]


def _image_chunks(
    core_meta: dict[str, Any] | None,
    ext_meta: dict[str, Any] | None,
    *,
    model_name: str,
    normalize: bool,
) -> list[dict[str, Any]]:
    """image 'st' 본문 재현: ``build_image_vlm_text_for_embedding(core|ext)`` 1청크.

    image_skill ``_embed_image`` 가 ``dict(core_meta)|dict(ext_meta)`` 로 같은 빌더를 호출하므로
    동일 텍스트가 나온다(chunk_index=0 정합). summary/keywords/labels 가 없으면 빈 리스트(skip).
    """
    meta = {**(core_meta or {}), **(ext_meta or {})}
    text = build_image_vlm_text_for_embedding(meta)
    return _embed_one_text(text, model_name=model_name, normalize=normalize)


def _video_chunks(
    core_meta: dict[str, Any] | None,
    ext_meta: dict[str, Any] | None,
    *,
    model_name: str,
    normalize: bool,
) -> list[dict[str, Any]]:
    """video 'st' 본문 재현: ``ext_meta.keyframes`` 프레임별 빌더 → chunk_index=프레임 순번.

    video_skill ``_embed_video`` 와 동일한 평탄화: keyframe.summary(dict)에서 summary 문자열·
    keywords 리스트를, keyframe.labels(list[{label,score}])를 프레임 메타로 만들어 빌더에 넣는다.
    빈 프레임도 ``" "`` 로 청크를 유지해 'st' 와 chunk_index 가 어긋나지 않게 한다(_embed_video 동일).
    keyframes 가 없으면 자산 summary 1청크로 폴백한다.
    """
    ext_meta = ext_meta or {}
    keyframes = ext_meta.get("keyframes")
    if isinstance(keyframes, list) and keyframes:
        out: list[dict[str, Any]] = []
        for i, kf in enumerate(keyframes):
            summ = kf.get("summary") if isinstance(kf.get("summary"), dict) else {}
            frame_meta = {
                "summary": str(summ.get("summary", "") or ""),
                "keywords": summ["keywords"] if isinstance(summ.get("keywords"), list) else [],
                "labels": kf.get("labels") or [],
            }
            text = build_image_vlm_text_for_embedding(frame_meta)
            if not text.strip():
                text = " "  # _embed_video 와 동일 — 빈 프레임도 청크 유지(chunk_index 정합)
            raw = embed_texts([text], model_name=model_name, normalize_embeddings=normalize)[0]
            out.append({"chunk_index": i, "embedding_vector": pad_embedding_to_storage_dim(raw)})
        return out
    # 폴백: keyframes 가 없으면 자산 전체 summary/keywords 1청크.
    return _image_chunks(core_meta, ext_meta, model_name=model_name, normalize=normalize)


def _audio_chunks(
    fs_path: str,
    ext_meta: dict[str, Any] | None,
    *,
    model_name: str,
    chunk_size: int,
    normalize: bool,
) -> list[dict[str, Any]]:
    """audio 'st' 본문 재현: ``ext_meta.stt``(없으면 사이드카) → ``embedding_plain_text_chunks``.

    audio_skill ``_embed_audio`` 와 동일한 STT 청킹. ext_meta.stt 가 비어있으면 ``.stt.txt``
    사이드카로 폴백하고, 둘 다 없으면 빈 리스트(skip).
    """
    ext_meta = ext_meta or {}
    stt = ext_meta.get("stt")
    if not (isinstance(stt, str) and stt.strip()):
        stt = _read_stt_sidecar(fs_path)
    if not (isinstance(stt, str) and stt.strip()):
        return []
    return embedding_plain_text_chunks(
        stt,
        chunk_size=chunk_size,
        embedding_model_name=model_name,
        normalize_embeddings=normalize,
    )


def _reload_chunks(
    fs_path: str,
    modality: str,
    core_meta: dict[str, Any] | None,
    ext_meta: dict[str, Any] | None,
    *,
    model_name: str,
    chunk_size: int,
    encoding: str,
    normalize: bool,
) -> list[dict[str, Any]]:
    """``modality`` 로 'st' 본문을 재현해 BGE 임베딩 청크([{chunk_index, embedding_vector}])를 반환.

    재현 불가(미지원 modality·본문 없음)면 빈 리스트를 반환해 호출부가 skip 한다. 파일 부재 등
    로딩 예외는 그대로 올려보내 ``backfill_asset`` 이 흡수(skip)한다.
    """
    # 문서 file_kind → 파일에서 직접 청킹+임베딩(text_skill 동일).
    file_kind = _document_file_kind(modality)
    if file_kind is None and modality == "text":
        # 053: canonical 'text' 저장값은 _document_file_kind 로 해소 안 됨(None) → skip 회귀.
        # fs_path 에서 세분류(txt/json/pdf/office)를 재도출해 문서 분기를 탄다(구값은 위에서 이미 해소).
        file_kind = _document_file_kind(detect_file_kind(fs_path))
    if file_kind is not None:
        return embedding_text_chunks(
            fs_path,
            file_kind=file_kind,
            encoding=encoding,
            chunk_size=chunk_size,
            embedding_model_name=model_name,
            normalize_embeddings=normalize,
        )
    if modality == "image":
        return _image_chunks(core_meta, ext_meta, model_name=model_name, normalize=normalize)
    if modality == "video":
        return _video_chunks(core_meta, ext_meta, model_name=model_name, normalize=normalize)
    if modality == "audio":
        return _audio_chunks(
            fs_path, ext_meta, model_name=model_name, chunk_size=chunk_size, normalize=normalize
        )
    # unknown 등 — 재현 가능한 'st' 본문이 없음 → skip.
    return []


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
    core_meta = asset_row.get("core_meta")
    ext_meta = asset_row.get("ext_meta")
    try:
        chunks = _reload_chunks(
            fs_path,
            modality,
            core_meta,
            ext_meta,
            model_name=model_name,
            chunk_size=chunk_size,
            encoding=encoding,
            normalize=normalize,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        _LOG.warning("skip(본문 로딩 실패) asset_id=%s %s: %s", asset_id, fs_path, exc)
        return 0
    if not chunks:
        _LOG.info("skip(본문 재현 불가) asset_id=%s modality=%s %s", asset_id, modality, fs_path)
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
    dotenv_path = _REPO_ROOT / f".env.{args.env}"
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
    sys.exit(main())
