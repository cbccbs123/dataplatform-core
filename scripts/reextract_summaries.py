"""요약·키워드 재추출 배치 1단계(026 G4/T009) — 프롬프트 토픽화(G1) 반영분을 기존 코퍼스에 적용.

ingest 재실행이 **아니다**: route/classify/persist 를 다시 타지 않고, registered 자산의
ext_meta 요약·키워드만 모달리티별 추출 경로로 재생성한다(spec FR-006).
  - 문서(txt/json/pdf/word/excel/powerpoint): `summarize_and_extract_keywords`(본문 재요약)
  - audio: ext_meta.stt(없으면 `<fs_path>.stt.txt` 사이드카) → `summarize_and_extract_keywords_from_audio`
  - video: ext_meta.keyframes(기존 장면 캡션) → `summarize_video_from_scene_results` — **VLM 재실행 0**
  - image: `summarize_image_caption_keywords_objects`(VLM 재호출) + **st/st_bge 재임베딩**
    (image 임베딩 입력 = summary+keywords+labels 라 요약이 바뀌면 임베딩도 바뀜 — DELETE+INSERT 교체.
     문서/audio 는 본문/STT 임베딩이라 불변, video 키프레임 캡션은 2단계(G6 판단)라 불변)

안전장치: ① 실행 전 ext_meta **전체 백업**(타임스탬프 JSON — 복구는 백업으로 UPDATE 역적용)
② 자산별 실패 격리(skip+카운트, 배치 무중단) ③ temperature=0 → 재실행 결정적 ④ --dry-run.

실행: conda run -n AuroraFS python scripts/reextract_summaries.py --env dev [--dry-run] [--limit N] [--modality image ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_LOG = logging.getLogger("meta_extract.reextract")

_DOC_KINDS = {"txt", "json", "pdf", "word", "excel", "powerpoint"}

_SELECT = """SELECT a.asset_id, a.modality, a.fs_path, am.ext_meta
FROM asset a JOIN asset_metadata am ON am.asset_id = a.asset_id
WHERE a.status = 'registered'
ORDER BY a.asset_id"""

_UPDATE = "UPDATE asset_metadata SET ext_meta = %s::jsonb WHERE asset_id = %s"

# image 재임베딩 교체용(요약 변경 → 임베딩 입력 변경). PK (asset_id, channel, chunk_index).
_DEL_EMB = "DELETE FROM asset_embedding WHERE asset_id = %s AND channel = %s"
_INS_EMB = (
    "INSERT INTO asset_embedding (asset_id, channel, chunk_index, embedding, model_name, model_version) "
    "VALUES (%s, %s, %s, %s::vector, %s, NULL)"
)

# image 재임베딩 대상 채널(KURE 는 026에서 삭제라 제외). 모델은 단일 출처 settings.model_for_channel
# 로 실행 시점에 해소한다 — 하드코딩 중복이 env 오버라이드와 어긋나는 드리프트 방지(리뷰 후속).
_IMAGE_REEMBED_CHANNELS = ("st", "st_bge")


def _audio_text(ext_meta: dict[str, Any], fs_path: str) -> str:
    """audio 재요약 입력 — ext_meta.stt 우선, 없으면 사이드카(.stt.txt). whisper 재실행 금지."""
    stt = str(ext_meta.get("stt") or "").strip()
    if stt:
        return stt
    sidecar = Path(f"{fs_path}.stt.txt")
    if sidecar.is_file():
        return sidecar.read_text(encoding="utf-8").strip()
    return ""


def _reextract_one(modality: str, fs_path: str, ext_meta: dict[str, Any]) -> dict[str, Any] | None:
    """한 자산의 새 요약·키워드(이미지는 objects 포함)를 만든다. 입력 부재면 None(skip)."""
    if modality in _DOC_KINDS:
        from src.llm.text_summarizer import summarize_and_extract_keywords

        if not Path(fs_path).is_file():
            return None
        out = summarize_and_extract_keywords(file_path=Path(fs_path), file_kind=modality)
        return {k: out[k] for k in ("summary", "keywords") if k in out}
    if modality == "audio":
        from src.llm.text_summarizer import summarize_and_extract_keywords_from_audio

        text = _audio_text(ext_meta, fs_path)
        if not text:
            return None
        out = summarize_and_extract_keywords_from_audio(text=text)
        return {k: out[k] for k in ("summary", "keywords") if k in out}
    if modality == "video":
        from src.llm.video_summarizer import summarize_video_from_scene_results

        scenes = ext_meta.get("keyframes")
        if not isinstance(scenes, list) or not scenes:
            return None
        out = summarize_video_from_scene_results(scenes)
        result = dict(out) if isinstance(out, dict) else {}
        return {k: result[k] for k in ("summary", "keywords") if k in result} or None
    if modality == "image":
        from src.llm.image_summarizer import summarize_image_caption_keywords_objects

        if not Path(fs_path).is_file():
            return None
        out = summarize_image_caption_keywords_objects(file_path=Path(fs_path))
        result = dict(out) if isinstance(out, dict) else {}
        return {k: result[k] for k in ("summary", "keywords", "objects") if k in result} or None
    return None  # unknown 등 — 재추출 대상 아님


def _reembed_image(conn: Any, asset_id: str, ext_meta: dict[str, Any], *, normalize: bool) -> int:
    """image 자산의 st/st_bge 임베딩을 새 요약 기반으로 교체한다(DELETE+INSERT·1청크)."""
    from src.config.settings import model_for_channel
    from src.embedders.text_embedder import embed_texts, pad_embedding_to_storage_dim
    from src.preprocess.vlm_text_for_embedding import build_image_vlm_text_for_embedding

    text = build_image_vlm_text_for_embedding(ext_meta)
    if not text.strip():
        return 0
    n = 0
    for channel in _IMAGE_REEMBED_CHANNELS:
        model = model_for_channel(channel)  # 단일 출처(018) — env 오버라이드와 항상 일치
        raw = embed_texts([text], model_name=model, normalize_embeddings=normalize)[0]
        vec = pad_embedding_to_storage_dim(list(raw))
        literal = "[" + ",".join(repr(float(x)) for x in vec) + "]"
        with conn.cursor() as cur:
            cur.execute(_DEL_EMB, (asset_id, channel))
            cur.execute(_INS_EMB, (asset_id, channel, 0, literal, model))
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="요약·키워드 재추출 배치 1단계(026)")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--dry-run", action="store_true", help="대상 집계만(쓰기 0)")
    parser.add_argument("--limit", type=int, default=None, help="처리 자산 수 상한")
    parser.add_argument("--modality", action="append", default=None, help="대상 모달리티 필터(반복 지정)")
    parser.add_argument("--backup-dir", default="backups", help="ext_meta 백업 디렉터리")
    # 050: stage-2 video 키프레임 VLM 재캡션·재임베딩 모드. v2 토글(049) 전제(FR-101).
    parser.add_argument(
        "--stage2",
        action="store_true",
        help="video 키프레임 VLM 재캡션·재임베딩(049 v2)·DELETE+INSERT 교체",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="--stage2 에서 VLM_SUMMARY_PROMPT_V2=off 게이트를 무시하고 강행(비권장)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / f".env.{args.env}", override=False)
    from src.config.settings import get_current_settings, init_settings

    init_settings(args.env)
    cfg = get_current_settings()

    # 050 FR-101: stage-2(video v2 재캡션)는 049 v2 토글 on 전제. off 면 v1 재캡션 = 무의미라
    # DB 연결 전에 중단한다(--force 로만 강행 가능). 게이트가 가장 먼저라 mock 단위로 검증 가능.
    if args.stage2 and not cfg.vlm_summary_prompt_v2 and not args.force:
        _LOG.error("VLM_SUMMARY_PROMPT_V2=off — v1 재캡션은 무의미. --force 로만 강행.")
        return 2

    from src.database.postgres_util import PostgresUtil

    db = PostgresUtil()
    counts = {"processed": 0, "skipped": 0, "failed": 0, "reembedded_images": 0}
    with db, db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SELECT)
            rows = cur.fetchall()
        # 050: stage-2 는 video 키프레임 재캡션 전용이라 대상을 video 로 강제 한정한다
        # (--modality 를 명시하지 않아도 다른 모달리티가 섞이지 않도록).
        mod_filter = set(args.modality) if args.modality else None
        if args.stage2:
            mod_filter = {"video"}
        targets = [
            (str(aid), mod, fp, em or {})
            for aid, mod, fp, em in rows
            if (mod_filter is None or mod in mod_filter)
        ]
        if args.limit:
            targets = targets[: args.limit]
        _LOG.info("대상 자산: %d건 (dry_run=%s)", len(targets), args.dry_run)
        if args.dry_run:
            if args.stage2:
                # stage-2 dry-run: video 대상 수·fs_path 부재 건만 집계(쓰기 0).
                missing = sum(1 for _a, _m, fp, _e in targets if not Path(fp or "").is_file())
                _LOG.info(
                    "stage-2 dry-run: video 대상 %d건 · fs_path 부재 %d건(쓰기 0)",
                    len(targets),
                    missing,
                )
                return 0
            by_mod: dict[str, int] = {}
            for _a, mod, _f, _e in targets:
                by_mod[mod] = by_mod.get(mod, 0) + 1
            _LOG.info("모달리티 분포: %s", by_mod)
            return 0

        # ── 백업(복구 가능성, FR-006) — 실행 전 전체 ext_meta 스냅샷 ──
        backup_dir = _REPO_ROOT / args.backup_dir
        backup_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"ext_meta_backup_{stamp}.json"
        backup_path.write_text(
            json.dumps(
                [{"asset_id": a, "modality": m, "ext_meta": e} for a, m, _f, e in targets],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _LOG.info("백업: %s (%d건)", backup_path, len(targets))

        for asset_id, modality, fs_path, ext_meta in targets:
            try:
                new_meta = _reextract_one(modality, fs_path, ext_meta)
                if not new_meta:
                    counts["skipped"] += 1
                    continue
                merged = {**ext_meta, **new_meta}  # summary/keywords(/objects)만 교체, 나머지 보존
                with conn.cursor() as cur:
                    cur.execute(_UPDATE, (json.dumps(merged, ensure_ascii=False), asset_id))
                if modality == "image":
                    counts["reembedded_images"] += _reembed_image(
                        conn, asset_id, merged, normalize=cfg.text_embedding_normalize
                    )
                conn.commit()
                counts["processed"] += 1
                if counts["processed"] % 20 == 0:
                    _LOG.info("진행: %s", counts)
            except Exception as e:  # 자산별 격리(FR-006) — 배치 무중단
                conn.rollback()
                counts["failed"] += 1
                _LOG.warning("재추출 실패 격리 asset=%s (%s: %s)", asset_id[:8], type(e).__name__, e)

    _LOG.info("완료: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
