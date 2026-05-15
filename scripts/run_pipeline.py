#!/usr/bin/env python3
"""메타 추출 · 통합/이미지 검색 · DB 전체 관계 제안."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_RELATIONS_BATCH_LOG = logging.getLogger("meta_extract.relations_batch")


def _configure_relations_batch_logging() -> None:
    log = _RELATIONS_BATCH_LOG
    if log.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False


def _fetch_media_item_ids(
    db: "PostgresUtil",
    *,
    media_types: list[str] | None,
    limit: int | None,
) -> list[int]:
    """관계 제안 배치용 ``media_items.id`` 목록."""
    if media_types:
        if limit is not None:
            rows = db.fetch_all(
                "SELECT id FROM media_items WHERE media_type = ANY(%s) ORDER BY id ASC LIMIT %s",
                (media_types, limit),
            )
        else:
            rows = db.fetch_all(
                "SELECT id FROM media_items WHERE media_type = ANY(%s) ORDER BY id ASC",
                (media_types,),
            )
    else:
        if limit is not None:
            rows = db.fetch_all(
                "SELECT id FROM media_items ORDER BY id ASC LIMIT %s",
                (limit,),
            )
        else:
            rows = db.fetch_all("SELECT id FROM media_items ORDER BY id ASC", ())
    return [int(r["id"]) for r in rows]


def _make_mock_llm_fn(
    db: "PostgresUtil",
    *,
    source_media_item_id: int,
    top_k_eff: int,
    kind: str,
) -> Callable[[str], dict[str, Any]]:
    first_id: int | None = None
    with db:
        with db.transaction() as conn:
            cands = find_embedding_candidates(
                conn,
                source_media_item_id=source_media_item_id,
                top_k=top_k_eff,
                embedding_kind=kind,  # type: ignore[arg-type]
            )
            if cands:
                first_id = cands[0]["id"]

    def _mock_llm(_: str) -> dict:
        if first_id is None:
            return {"edges": []}
        return {
            "edges": [
                {
                    "target_media_item_id": first_id,
                    "relation_type_code": "same_domain",
                    "confidence": 0.5,
                    "reason": "relation-mock-llm",
                    "topic_ko": "일반",
                    "subtopic_ko": "mock",
                    "topic_en": "general",
                    "subtopic_en": "",
                }
            ]
        }

    return _mock_llm


def _run_propose_relations_once(
    db: "PostgresUtil",
    *,
    source_media_item_id: int,
    relation_top_k: int | None,
    relation_embedding_kind: str,
    relation_mock_llm: bool,
) -> dict:
    """단일 소스 관계 제안 실행. 결과 dict 반환."""
    kind = relation_embedding_kind
    top_k_eff = (
        relation_top_k
        if relation_top_k is not None
        else get_current_settings().relation_top_k
    )
    llm_fn = None
    if relation_mock_llm:
        llm_fn = _make_mock_llm_fn(
            db,
            source_media_item_id=source_media_item_id,
            top_k_eff=top_k_eff,
            kind=kind,
        )
    with db:
        catalog_synced, catalog_skipped, edge_synced, edge_skipped = propose_relations_for_item(
            db,
            source_media_item_id,
            top_k=relation_top_k,
            embedding_kind=kind,  # type: ignore[arg-type]
            llm_fn=llm_fn,
        )
    return {
        "source_media_item_id": source_media_item_id,
        "relation_catalog_synced": catalog_synced,
        "relation_catalog_skipped": catalog_skipped,
        "relation_edges_upserted": edge_synced,
        "relation_edges_skipped": edge_skipped,
        "relation_top_k": relation_top_k,
        "relation_embedding_kind": kind,
        "relation_mock_llm": relation_mock_llm,
    }


try:
    from src.app.run_extract_meta import run_extract_meta
    from src.config.settings import get_current_settings, init_settings
    from src.database.postgres_util import PostgresUtil
    from src.file.directory_paths import list_file_paths_under_directory
    from src.relations.candidates import find_embedding_candidates
    from src.relations.entry import propose_relations_for_item
    from src.search.media_search import search_media_all_grouped, search_media_images_two_stage
    from src.search.query_preprocess import structure_user_query
except ImportError:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.app.run_extract_meta import run_extract_meta
    from src.config.settings import get_current_settings, init_settings
    from src.database.postgres_util import PostgresUtil
    from src.file.directory_paths import list_file_paths_under_directory
    from src.relations.candidates import find_embedding_candidates
    from src.relations.entry import propose_relations_for_item
    from src.search.media_search import search_media_all_grouped, search_media_images_two_stage
    from src.search.query_preprocess import structure_user_query


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="메타 추출 · 통합/이미지 검색 · DB 전체 관계 제안",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--env",
        choices=["dev", "prod"],
        default="dev",
        help="설정 프로파일",
    )
    parser.add_argument(
        "--relations-all",
        "--propose-relations-all",
        dest="relations_all",
        action="store_true",
        help="DB의 media_items 전체에 대해 관계 제안(구 이름 propose 호환)",
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "relations-run"],
        default=None,
        help="relations-run: 관계 배치(--relations-item-ids 없으면 DB 전체). ingest: 파일만(명시용)",
    )
    parser.add_argument(
        "--relations-item-ids",
        dest="relations_item_ids",
        nargs="*",
        type=int,
        default=None,
        metavar="ID",
        help="--mode relations-run 일 때만: 지정 id만 처리. 생략 시 DB 전체 스캔",
    )
    parser.add_argument(
        "--relations-min-similarity",
        dest="relations_min_similarity",
        type=float,
        default=None,
        help="예약(후보 SQL 미적용)",
    )
    parser.add_argument(
        "--relation-top-k",
        "--relations-top-k",
        dest="relation_top_k",
        type=int,
        default=None,
        metavar="K",
        help="임베딩 후보 상위 k",
    )
    parser.add_argument(
        "--relation-embedding-kind",
        "--relations-embedding-kind",
        dest="relation_embedding_kind",
        choices=["st", "clip", "both"],
        default="both",
        help="후보 embedding_kind",
    )
    parser.add_argument(
        "--relation-mock-llm",
        "--relations-mock-llm",
        dest="relation_mock_llm",
        action="store_true",
        help="LLM 대신 고정 응답",
    )
    parser.add_argument(
        "--relation-media-type",
        dest="relation_media_types",
        action="append",
        default=None,
        metavar="TYPE",
        help="media_type 필터(DB 전체 스캔 시). 반복 가능",
    )
    parser.add_argument(
        "--relation-batch-limit",
        dest="relation_batch_limit",
        type=int,
        default=None,
        metavar="N",
        help="처리 최대 건수(DB 전체 스캔 시)",
    )
    parser.add_argument(
        "--search",
        metavar="QUERY",
        default=None,
        help="통합 검색(JSON 출력)",
    )
    parser.add_argument(
        "--search-debug",
        action="store_true",
        help="통합 검색 ST 하이브리드에 candidate_count",
    )
    parser.add_argument(
        "--search-images",
        dest="search_images",
        metavar="QUERY",
        default=None,
        help="이미지 검색(JSON 출력)",
    )
    parser.add_argument(
        "--image-search-alpha",
        type=float,
        default=0.65,
        metavar="A",
        help="하이브리드 가중 0~1",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="FILE",
        help="메타 추출할 파일(관계/검색 모드와 배타)",
    )
    parser.add_argument(
        "--file-list",
        dest="file_list",
        type=Path,
        metavar="PATH",
        help="파일 경로 목록(UTF-8 한 줄 하나)",
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        type=Path,
        metavar="DIR",
        help="디렉터리에서 파일 수집",
    )
    parser.add_argument(
        "--dir-no-recurse",
        action="store_true",
        help="--input-dir 시 비재귀",
    )
    parser.add_argument(
        "--dir-include-hidden",
        action="store_true",
        help="--input-dir 시 숨김 포함",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    dotenv_path = PROJECT_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    init_settings(args.env)

    db_scan = bool(
        args.relations_all
        or (args.mode == "relations-run" and args.relations_item_ids is None)
    )
    id_pick = bool(args.mode == "relations-run" and args.relations_item_ids is not None)

    if args.relations_all and args.mode == "relations-run" and args.relations_item_ids is not None:
        parser.error("--relations-all(--propose-relations-all) 과 --relations-item-ids 는 함께 쓸 수 없습니다.")

    if args.relations_item_ids is not None and args.mode != "relations-run":
        parser.error("--relations-item-ids 는 --mode relations-run 과 함께만 사용하세요.")

    if args.mode == "ingest":
        if args.relations_all:
            parser.error("--mode ingest 는 --relations-all 과 함께 쓸 수 없습니다.")
        if args.search is not None or args.search_images is not None:
            parser.error("--mode ingest 는 --search / --search-images 와 함께 쓸 수 없습니다.")

    want_rel_batch = db_scan or id_pick
    want_search = args.search is not None
    want_img = args.search_images is not None
    n_cmd = int(want_rel_batch) + int(want_search) + int(want_img)
    if n_cmd > 1:
        parser.error("관계 배치, --search, --search-images 중 하나만 지정하세요.")

    if want_rel_batch:
        if args.paths or args.file_list is not None or args.input_dir is not None:
            parser.error("관계 배치는 파일 경로 옵션과 함께 쓸 수 없습니다.")
        if (args.relation_media_types or args.relation_batch_limit is not None) and not db_scan:
            parser.error("--relation-media-type / --relation-batch-limit 는 DB 전체 스캔에만 사용하세요.")
        if id_pick and (args.relation_media_types or args.relation_batch_limit is not None):
            parser.error("명시 id 배치에는 --relation-media-type / --relation-batch-limit 를 쓸 수 없습니다.")
        if args.relation_batch_limit is not None and args.relation_batch_limit <= 0:
            parser.error("--relation-batch-limit N 은 양의 정수여야 합니다.")

        db = PostgresUtil()
        if db_scan:
            ids = _fetch_media_item_ids(
                db,
                media_types=args.relation_media_types,
                limit=args.relation_batch_limit,
            )
        else:
            ids = list(args.relations_item_ids or [])

        _configure_relations_batch_logging()
        n_ids = len(ids)
        _RELATIONS_BATCH_LOG.info(
            "relations batch start mode=%s items=%s mock_llm=%s",
            "relations_all" if db_scan else "relations_run_ids",
            n_ids,
            bool(args.relation_mock_llm),
        )

        results: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        total_catalog_synced = 0
        total_catalog_skipped = 0
        total_edges_upserted = 0
        total_edges_skipped = 0
        for i, sid in enumerate(ids, start=1):
            t0 = time.perf_counter()
            _RELATIONS_BATCH_LOG.info(
                "relations item %s/%s source_media_item_id=%s start",
                i,
                n_ids,
                sid,
            )
            try:
                row = _run_propose_relations_once(
                    db,
                    source_media_item_id=sid,
                    relation_top_k=args.relation_top_k,
                    relation_embedding_kind=args.relation_embedding_kind,
                    relation_mock_llm=args.relation_mock_llm,
                )
                total_catalog_synced += int(row["relation_catalog_synced"])
                total_catalog_skipped += int(row["relation_catalog_skipped"])
                total_edges_upserted += int(row["relation_edges_upserted"])
                total_edges_skipped += int(row["relation_edges_skipped"])
                results.append(row)
                elapsed = time.perf_counter() - t0
                _RELATIONS_BATCH_LOG.info(
                    "relations item %s/%s source_media_item_id=%s ok catalog_synced=%s catalog_skipped=%s edges_upserted=%s edges_skipped=%s elapsed_s=%.2f",
                    i,
                    n_ids,
                    sid,
                    row["relation_catalog_synced"],
                    row["relation_catalog_skipped"],
                    row["relation_edges_upserted"],
                    row["relation_edges_skipped"],
                    elapsed,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                _RELATIONS_BATCH_LOG.exception(
                    "relations item %s/%s source_media_item_id=%s failed after_s=%.2f",
                    i,
                    n_ids,
                    sid,
                    elapsed,
                )
                failed.append({"source_media_item_id": sid, "error": str(exc)})
        out: dict[str, Any] = {
            "mode": "relations_all" if db_scan else "relations_run_ids",
            "media_item_ids": ids,
            "count": len(ids),
            "relation_media_types": args.relation_media_types,
            "relation_batch_limit": args.relation_batch_limit,
            "totals": {
                "relation_catalog_synced": total_catalog_synced,
                "relation_catalog_skipped": total_catalog_skipped,
                "relation_edges_upserted": total_edges_upserted,
                "relation_edges_skipped": total_edges_skipped,
            },
            "items": results,
            "failed": failed,
        }
        _RELATIONS_BATCH_LOG.info(
            "relations batch done items=%s ok=%s failed=%s totals_catalog_synced=%s totals_catalog_skipped=%s totals_edges_upserted=%s totals_edges_skipped=%s",
            n_ids,
            len(results),
            len(failed),
            total_catalog_synced,
            total_catalog_skipped,
            total_edges_upserted,
            total_edges_skipped,
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))
        sys.exit(1 if failed else 0)

    if want_search:
        if args.paths or args.file_list is not None or args.input_dir is not None:
            parser.error("--search 는 파일 경로 옵션과 함께 쓸 수 없습니다.")
        if not str(args.search).strip():
            parser.error("--search QUERY 에 빈 문자열을 쓸 수 없습니다.")
        a = args.image_search_alpha
        if not 0.0 <= a <= 1.0:
            parser.error("--image-search-alpha 는 0 이상 1 이하여야 합니다.")
        grouped = search_media_all_grouped(
            str(args.search),
            limit_per_bucket=20,
            image_search_alpha=args.image_search_alpha,
            debug=args.search_debug,
        )
        print(json.dumps(grouped, indent=2, ensure_ascii=False))
        sys.exit(0)

    if want_img:
        if args.paths or args.file_list is not None or args.input_dir is not None:
            parser.error("--search-images 는 파일 경로 옵션과 함께 쓸 수 없습니다.")
        if not str(args.search_images).strip():
            parser.error("--search-images QUERY 에 빈 문자열을 쓸 수 없습니다.")
        a = args.image_search_alpha
        if not 0.0 <= a <= 1.0:
            parser.error("--image-search-alpha 는 0 이상 1 이하여야 합니다.")
        sr = structure_user_query(str(args.search_images))
        rows = search_media_images_two_stage(
            sr["semantic_query"],
            sr.get("semantic_query_en") or "",
            alpha=args.image_search_alpha,
        )
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        sys.exit(0)

    if args.mode == "ingest" and not (args.paths or args.file_list is not None or args.input_dir is not None):
        parser.error("--mode ingest 는 FILE / --file-list / --input-dir 가 필요합니다.")

    files: list[str] = []
    if args.file_list is not None:
        if not args.file_list.is_file():
            parser.error(f"--file-list 가 파일이 아닙니다: {args.file_list}")
        text = args.file_list.read_text(encoding="utf-8")
        files.extend(line.strip() for line in text.splitlines() if line.strip())
    if args.input_dir is not None:
        root = args.input_dir.expanduser()
        if not root.is_dir():
            parser.error(f"--input-dir 가 디렉터리가 아닙니다: {args.input_dir}")
        files.extend(
            list_file_paths_under_directory(
                root,
                recursive=not args.dir_no_recurse,
                include_hidden=args.dir_include_hidden,
                dedup_by_prefix=True,
                sample_seed=42,
            )
        )
    files.extend(str(p) for p in args.paths)
    if not files:
        parser.error(
            "메타 추출할 파일이 없습니다. FILE, --file-list, --input-dir 을 주거나 "
            "--relations-all / --mode relations-run / --search / --search-images 를 쓰세요."
        )
    n_failed = run_extract_meta(files)
    sys.exit(1 if n_failed else 0)
