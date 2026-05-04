from __future__ import annotations

import json
import time
import traceback
from typing import Any, Iterable

from psycopg.rows import dict_row

from src.database.postgres_util import PostgresUtil
from src.embedders.image_embedder import (
    clip_zero_shot_ko_meta_items,
    embed_clip_text_query_for_image_search,
    zero_shot_tag_image_korean_clip,
)
from src.embedders.video_embedder import embed_video_keyframes_clip
from src.embedders.text_embedder import (
    embed_texts,
    embedding_text_chunks,
    pad_embedding_to_storage_dim,
)
from src.embedders.text_embedding_normalize import normalize_text_for_embedding
from src.extractors.audio_meta_extractor import extract_audio_meta
from src.extractors.image_meta_extractor import extract_image_meta
from src.extractors.text_meta_extractor import extract_text_meta
from src.llm.image_summarizer import (
    summarize_image_caption_keywords_objects,
    summarize_image_caption_keywords_objects_from_jpeg_bytes,
)
from src.llm.text_summarizer import summarize_and_extract_keywords, summarize_and_extract_keywords_from_audio
from src.llm.video_summarizer import summarize_video_from_scene_results
from src.config.settings import get_current_settings
from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind
from src.file.file_type_detector import detect_file_kind
from src.preprocess.stt import transcribe_audio_local
from src.preprocess.video_keyframes import extract_video_basic_meta, extract_video_representative_frame_bytes

_FIX_EMBEDDING_DIMENSION = 1536

# media_chunks.embedding_kind: 문서·이미지 ST 청크는 st, 이미지 CLIP(픽셀) 청크는 clip
_EMBEDDING_KIND_ST = "st"
_EMBEDDING_KIND_CLIP = "clip"


def build_image_vlm_text_for_embedding(meta: dict[str, Any]) -> str:
    """VLM 요약·키워드·제로샷 라벨을 한 덩어리로 묶어 ST 임베딩·청크 content에 쓴다."""
    summary_txt = str(meta.get("summary", "") or "").strip()
    kws = meta.get("keywords") or []
    kw_line = (
        " ".join(str(k).strip() for k in kws if str(k).strip())
        if isinstance(kws, list)
        else ""
    )
    lab_parts: list[str] = []
    for item in meta.get("labels") or []:
        if isinstance(item, dict):
            lab = item.get("label")
            if lab:
                lab_parts.append(str(lab).strip())
        elif isinstance(item, str) and item.strip():
            lab_parts.append(item.strip())
    label_line = " ".join(lab_parts)
    parts = [p for p in (summary_txt, kw_line, label_line) if p]
    raw = "\n".join(parts).strip() if parts else " "
    return normalize_text_for_embedding(raw) if raw.strip() else " "


def embed_query_for_media_search(query: str) -> list[float]:
    cfg = get_current_settings()
    raw = query.strip() if query.strip() else " "
    q = normalize_text_for_embedding(raw)
    if not q.strip():
        q = " "
    row = embed_texts(
        [q],
        model_name=cfg.text_embedding_model,
        normalize_embeddings=cfg.text_embedding_normalize,
    )[0]
    return pad_embedding_to_storage_dim(row)

def search_media_items(query: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """
    pgvector 코사인 연산(`<=>`, 인덱스 `vector_cosine_ops`)으로 근접 순위를 매긴다.

    ``media_items`` 행은 **중복 없이** 한 줄씩이며, ``similarity``는 그 문서에 속한 청크들 중
    쿼리와 가장 가까운 청크 기준(``MAX(1 - 거리)``)이다.

    쿼리 벡터는 SentenceTransformer 기반이므로 CLIP 이미지 청크와 섞이지 않게
    ``ALLOWED_TEXT_META_FILE_KINDS``에 해당하는 ``media_type``만 조회한다.
    """
    db = PostgresUtil()
    query_vector = embed_query_for_media_search(query)
    vdim = _FIX_EMBEDDING_DIMENSION
    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT
                        mi.id,
                        mi.file_uri,
                        mi.media_type,
                        MAX(1 - (mc.embedding <=> %s::vector({vdim}))) AS similarity
                    FROM media_items mi
                    JOIN media_chunks mc ON mi.id = mc.media_item_id
                    WHERE mc.embedding IS NOT NULL
                      AND mi.media_type = ANY(%s)
                      AND mc.embedding_kind = %s
                    GROUP BY mi.id, mi.file_uri, mi.media_type
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (
                        query_vector,
                        list(ALLOWED_TEXT_META_FILE_KINDS),
                        _EMBEDDING_KIND_ST,
                        limit,
                    ),
                )
                return list(cur.fetchall())


def search_media_images_by_text(
    query: str,
    *,
    limit: int = 20,
    clip_model_name: str = "openai/clip-vit-base-patch32",
) -> list[dict[str, Any]]:
    """
    CLIP 텍스트 인코더 쿼리와 ``media_chunks``의 CLIP 이미지 임베딩을 코사인으로 비교한다.

    ``search_media_items``(SentenceTransformer)와 달리 ``media_type = image`` 만 대상이다.
    """
    db = PostgresUtil()
    query_vector = embed_clip_text_query_for_image_search(query, model_name=clip_model_name)
    vdim = _FIX_EMBEDDING_DIMENSION
    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT
                        mi.id,
                        mi.file_uri,
                        mi.media_type,
                        MAX(1 - (mc.embedding <=> %s::vector({vdim}))) AS similarity
                    FROM media_items mi
                    JOIN media_chunks mc ON mi.id = mc.media_item_id
                    WHERE mc.embedding IS NOT NULL
                      AND mi.media_type = %s
                      AND mc.embedding_kind = %s
                    GROUP BY mi.id, mi.file_uri, mi.media_type
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (query_vector, MediaKind.IMAGE.value, _EMBEDDING_KIND_CLIP, limit),
                )
                return list(cur.fetchall())


def search_media_images_two_stage(
    query: str,
    *,
    stage1_limit: int = 80,
    final_limit: int = 20,
    alpha: float = 0.65,
    clip_model_name: str = "openai/clip-vit-base-patch32",
) -> list[dict[str, Any]]:
    """
    1차: 이미지의 VLM 텍스트(ST) 청크로 넓게 후보를 고른 뒤,
    2차: 동일 후보에 CLIP 텍스트↔CLIP 이미지 벡터 점수를 섞어 재정렬한다.

    ``alpha``·``(1-alpha)`` 가중합으로 ``similarity``를 만든다. CLIP 전용 레거시 행만
    있으면 ``search_media_images_by_text``와 동일하게 동작한다.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha는 0~1 사이여야 합니다.")

    db = PostgresUtil()
    text_q = embed_query_for_media_search(query)
    clip_q = embed_clip_text_query_for_image_search(query, model_name=clip_model_name)
    vdim = _FIX_EMBEDDING_DIMENSION

    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT
                        mi.id,
                        mi.file_uri,
                        mi.media_type,
                        MAX(1 - (mc.embedding <=> %s::vector({vdim}))) AS s_text
                    FROM media_items mi
                    JOIN media_chunks mc ON mi.id = mc.media_item_id
                    WHERE mc.embedding IS NOT NULL
                      AND mi.media_type = %s
                      AND mc.embedding_kind = %s
                    GROUP BY mi.id, mi.file_uri, mi.media_type
                    ORDER BY s_text DESC
                    LIMIT %s
                    """,
                    (text_q, MediaKind.IMAGE.value, _EMBEDDING_KIND_ST, stage1_limit),
                )
                stage1 = list(cur.fetchall())

    if not stage1:
        clip_only = search_media_images_by_text(
            query, limit=final_limit, clip_model_name=clip_model_name
        )
        return [
            {
                **row,
                "s_text": 0.0,
                "s_clip": float(row["similarity"]),
                "similarity": float(row["similarity"]),
            }
            for row in clip_only
        ]

    ids = [int(r["id"]) for r in stage1]
    clip_by_id: dict[int, float] = {}
    with db:
        with db.transaction() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT
                        mi.id,
                        MAX(1 - (mc.embedding <=> %s::vector({vdim}))) AS s_clip
                    FROM media_items mi
                    JOIN media_chunks mc ON mi.id = mc.media_item_id
                    WHERE mc.embedding IS NOT NULL
                      AND mi.media_type = %s
                      AND mc.embedding_kind = %s
                      AND mi.id = ANY(%s)
                    GROUP BY mi.id
                    """,
                    (clip_q, MediaKind.IMAGE.value, _EMBEDDING_KIND_CLIP, ids),
                )
                for row in cur.fetchall():
                    clip_by_id[int(row["id"])] = float(row["s_clip"])

    merged: dict[int, dict[str, Any]] = {}
    for row in stage1:
        iid = int(row["id"])
        s_text = float(row["s_text"])
        s_clip = float(clip_by_id.get(iid, 0.0))
        sim = alpha * s_text + (1.0 - alpha) * s_clip
        merged[iid] = {
            "id": iid,
            "file_uri": row["file_uri"],
            "media_type": row["media_type"],
            "s_text": s_text,
            "s_clip": s_clip,
            "similarity": sim,
        }

    clip_extra = search_media_images_by_text(
        query, limit=max(stage1_limit, final_limit * 4), clip_model_name=clip_model_name
    )
    for row in clip_extra:
        iid = int(row["id"])
        if iid in merged:
            continue
        s_clip = float(row["similarity"])
        merged[iid] = {
            "id": iid,
            "file_uri": row["file_uri"],
            "media_type": row["media_type"],
            "s_text": 0.0,
            "s_clip": s_clip,
            "similarity": (1.0 - alpha) * s_clip,
        }

    ranked = sorted(merged.values(), key=lambda r: r["similarity"], reverse=True)
    return ranked[:final_limit]


def run_extract_meta(file_list: Iterable[str]) -> None:
    db = PostgresUtil()
    with db:
        print("Connected:", db.server_version())
        print("Health:", db.health_check())

    for file in file_list:
        start_time = time.time()
        file_kind = detect_file_kind(file)
        print(f"파일 경로: {file} / 파일명: {file.split('/')[-1]} / 파일 종류: {file_kind}")
        try:
            if file_kind in ALLOWED_TEXT_META_FILE_KINDS:
                cfg = get_current_settings()
                meta = extract_text_meta(
                    file_path=file,
                    file_kind=file_kind,
                    encoding=cfg.encoding,
                    chunk_size=cfg.text_embedding_chunk_size,
                    embedding_model_name=cfg.text_embedding_model,
                )
                summary = summarize_and_extract_keywords(
                    file_path=file,
                    file_kind=file_kind
                )
                meta = meta | summary
                chunks = embedding_text_chunks(
                    file,
                    file_kind=file_kind,
                    encoding=cfg.encoding,
                    chunk_size=cfg.text_embedding_chunk_size,
                    embedding_model_name=cfg.text_embedding_model,
                    normalize_embeddings=cfg.text_embedding_normalize,
                )
                print("meta:", json.dumps(meta, ensure_ascii=False, indent=4))
                print("text_chunks:", len(chunks))
                with db.transaction() as conn:
                    with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute(
                            """
                            INSERT INTO media_items (file_uri, media_type, metadata)
                            VALUES (%s, %s, %s)
                            RETURNING id
                            """,
                            (file, file_kind, json.dumps(meta)),
                        )
                        row = cur.fetchone()
                        if row is None:
                            raise RuntimeError("media_items INSERT did not return id")
                        media_item_id = row["id"]
                        cur.executemany(
                            f"""
                            INSERT INTO media_chunks (
                                media_item_id, chunk_index, content, embedding, embedding_kind
                            ) VALUES (%s, %s, %s, %s::vector({_FIX_EMBEDDING_DIMENSION}), %s)
                            """,
                            [
                                (
                                    media_item_id,
                                    c["chunk_index"],
                                    c["content"],
                                    c["embedding_vector"],
                                    _EMBEDDING_KIND_ST,
                                )
                                for c in chunks
                            ],
                        )
            elif file_kind == MediaKind.IMAGE.value:
                # 1) 파일·이미지 속성 메타 (Pillow 등)
                meta = extract_image_meta(file_path=file)
                # 2) VLM: 캡션·키워드·객체(문자열 목록은 CLIP 후보만; 최종 meta에는 objects 미포함)
                summary = summarize_image_caption_keywords_objects(file_path=file)
                objects = summary.get("objects") or []
                meta = meta | summary
                meta.pop("objects", None)
                # 3) CLIP: 이미지 임베딩 1회(1536) + 한글 라벨·제로샷 점수 → clip_zero_shot_ko
                obj_list = [str(o) for o in objects] if isinstance(objects, list) else []
                zs = zero_shot_tag_image_korean_clip(
                    file_path=file,
                    korean_labels=obj_list,
                )
                meta = meta #| {"clip_image_embedding": zs["clip_image_embedding"]}
                if zs["label_scores"]:
                    meta = meta | {
                        "labels": clip_zero_shot_ko_meta_items(zs["label_scores"]),
                    }
                print("meta:", json.dumps(meta, ensure_ascii=False, indent=4))
                clip_vec = zs["clip_image_embedding"]
                cfg_img = get_current_settings()
                chunk_content = build_image_vlm_text_for_embedding(meta)
                if not chunk_content.strip():
                    chunk_content = " "
                st_raw = embed_texts(
                    [chunk_content],
                    model_name=cfg_img.text_embedding_model,
                    normalize_embeddings=cfg_img.text_embedding_normalize,
                )[0]
                st_vec = pad_embedding_to_storage_dim(st_raw)
                with db.transaction() as conn:
                    with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute(
                            """
                            INSERT INTO media_items (file_uri, media_type, metadata)
                            VALUES (%s, %s, %s)
                            RETURNING id
                            """,
                            (file, file_kind, json.dumps(meta)),
                        )
                        row = cur.fetchone()
                        if row is None:
                            raise RuntimeError("media_items INSERT did not return id")
                        media_item_id = row["id"]
                        cur.executemany(
                            f"""
                            INSERT INTO media_chunks (
                                media_item_id, chunk_index, content, embedding, embedding_kind
                            ) VALUES (%s, %s, %s, %s::vector({_FIX_EMBEDDING_DIMENSION}), %s)
                            """,
                            [
                                (
                                    media_item_id,
                                    0,
                                    chunk_content,
                                    st_vec,
                                    _EMBEDDING_KIND_ST,
                                ),
                                (
                                    media_item_id,
                                    1,
                                    chunk_content,
                                    clip_vec,
                                    _EMBEDDING_KIND_CLIP,
                                ),
                            ],
                        )
            elif file_kind == MediaKind.VIDEO.value:
                cfg = get_current_settings()
                frame_items = extract_video_representative_frame_bytes(
                    video_path=file,
                    max_frames=cfg.video_max_keyframes,
                )
                korean_labels_per_frame: list[list[str]] = []
                result = []
                for item in frame_items:
                    summary = summarize_image_caption_keywords_objects_from_jpeg_bytes(
                        item["jpeg_bytes"]
                    )
                    objects = summary.get("objects") or []
                    obj_list = [str(o) for o in objects] if isinstance(objects, list) else []
                    korean_labels_per_frame.append(obj_list)
                    result.append(
                        {
                            "scene_index": item["scene_index"],
                            "start_sec": item["start_sec"],
                            "end_sec": item["end_sec"],
                            "frame_sec": item["frame_sec"],
                            "jpeg_bytes": item["jpeg_bytes"],
                            "summary": summary,
                        }
                    )
                meta = extract_video_basic_meta(file_path=file)
                meta = meta | summarize_video_from_scene_results(result)
                clip_ve = embed_video_keyframes_clip(
                    result,
                    korean_labels_per_frame=korean_labels_per_frame,
                )
                for _it in result:
                    _it.pop("jpeg_bytes", None)
                meta["keyframes"] = clip_ve["keyframes"]
                print("meta:", json.dumps(meta, ensure_ascii=False, indent=4))
            elif file_kind == MediaKind.AUDIO.value:
                stt_result = transcribe_audio_local(file_path=file)
                meta = extract_audio_meta(file_path=file)
                summary = summarize_and_extract_keywords_from_audio(text=stt_result["text"])
                meta = meta | summary
                print("meta:", json.dumps(meta, ensure_ascii=False, indent=4))
            else:
                raise ValueError(f"파일 종류: {file_kind}는 지원하지 않습니다.")
        except ValueError as e:
            print(f"파일 요약 추출 오류: {e}")
            continue
        except Exception as e:
            print(f"예외 발생: {e}")
            print(traceback.format_exc())
            continue
        finally:
            elapsed_time = time.time() - start_time
            print(f"소요 시간: {elapsed_time:.2f}초")
            print("-" * 150)
