from __future__ import annotations

import json
import time
import traceback
from typing import Any, Iterable

from psycopg.rows import dict_row

from src.database.postgres_util import PostgresUtil
from src.embedders.image_embedder import (
    clip_zero_shot_ko_meta_items,
    zero_shot_tag_image_korean_clip,
)
from src.embedders.video_embedder import embed_video_keyframes_clip
from src.embedders.text_embedder import (
    embed_texts,
    embedding_plain_text_chunks,
    embedding_text_chunks,
    pad_embedding_to_storage_dim,
)
from src.preprocess.vlm_text_for_embedding import build_image_vlm_text_for_embedding
from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.search.media_search import (
    EMBEDDING_KIND_CLIP,
    EMBEDDING_KIND_ST,
)
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
                            ) VALUES (%s, %s, %s, %s::vector({FIX_EMBEDDING_DIMENSION}), %s)
                            """,
                            [
                                (
                                    media_item_id,
                                    c["chunk_index"],
                                    c["content"],
                                    c["embedding_vector"],
                                    EMBEDDING_KIND_ST,
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
                            ) VALUES (%s, %s, %s, %s::vector({FIX_EMBEDDING_DIMENSION}), %s)
                            """,
                            [
                                (
                                    media_item_id,
                                    0,
                                    chunk_content,
                                    st_vec,
                                    EMBEDDING_KIND_ST,
                                ),
                                (
                                    media_item_id,
                                    1,
                                    chunk_content,
                                    clip_vec,
                                    EMBEDDING_KIND_CLIP,
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
                meta["keyframes"] = [
                    {k: v for k, v in kf.items() if k != "clip_image_embedding"}
                    for kf in clip_ve["keyframes"]
                ]
                print("meta:", json.dumps(meta, ensure_ascii=False, indent=4))
                pair_rows: list[tuple[int, str, list[float], str]] = []
                for i, kf in enumerate(clip_ve["keyframes"]):
                    summ = kf.get("summary") or {}
                    if not isinstance(summ, dict):
                        summ = {}
                    frame_meta: dict[str, Any] = {
                        "summary": str(summ.get("summary", "") or ""),
                        "keywords": summ["keywords"]
                        if isinstance(summ.get("keywords"), list)
                        else [],
                        "labels": kf.get("labels") or [],
                    }
                    chunk_content = build_image_vlm_text_for_embedding(frame_meta)
                    if not chunk_content.strip():
                        chunk_content = " "
                    st_raw = embed_texts(
                        [chunk_content],
                        model_name=cfg.text_embedding_model,
                        normalize_embeddings=cfg.text_embedding_normalize,
                    )[0]
                    st_vec = pad_embedding_to_storage_dim(st_raw)
                    clip_vec = kf["clip_image_embedding"]
                    pair_rows.append((2 * i, chunk_content, st_vec, EMBEDDING_KIND_ST))
                    pair_rows.append((2 * i + 1, chunk_content, clip_vec, EMBEDDING_KIND_CLIP))
                print("video_keyframe_chunk_pairs:", len(pair_rows) // 2)
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
                        ins = cur.fetchone()
                        if ins is None:
                            raise RuntimeError("media_items INSERT did not return id")
                        media_item_id = ins["id"]
                        if pair_rows:
                            cur.executemany(
                                f"""
                                INSERT INTO media_chunks (
                                    media_item_id, chunk_index, content, embedding, embedding_kind
                                ) VALUES (%s, %s, %s, %s::vector({FIX_EMBEDDING_DIMENSION}), %s)
                                """,
                                [
                                    (media_item_id, idx, content, emb, kind)
                                    for (idx, content, emb, kind) in pair_rows
                                ],
                            )
            elif file_kind == MediaKind.AUDIO.value:
                stt_result = transcribe_audio_local(file_path=file)
                meta = extract_audio_meta(file_path=file)
                summary = summarize_and_extract_keywords_from_audio(text=stt_result["text"])
                meta = meta | summary
                cfg = get_current_settings()
                chunks = embedding_plain_text_chunks(
                    stt_result["text"],
                    chunk_size=cfg.text_embedding_chunk_size,
                    embedding_model_name=cfg.text_embedding_model,
                    normalize_embeddings=cfg.text_embedding_normalize,
                )
                print("meta:", json.dumps(meta, ensure_ascii=False, indent=4))
                print("audio_stt_chunks:", len(chunks))
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
                            ) VALUES (%s, %s, %s, %s::vector({FIX_EMBEDDING_DIMENSION}), %s)
                            """,
                            [
                                (
                                    media_item_id,
                                    c["chunk_index"],
                                    c["content"],
                                    c["embedding_vector"],
                                    EMBEDDING_KIND_ST,
                                )
                                for c in chunks
                            ],
                        )
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
