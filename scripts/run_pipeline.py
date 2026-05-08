from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from src.app.run_extract_meta import run_extract_meta
    from src.config.settings import init_settings
    from src.file.directory_paths import list_file_paths_under_directory
    from src.search.media_search import search_media_all_grouped, search_media_images_two_stage
    from src.search.query_preprocess import structure_user_query
except ImportError:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.app.run_extract_meta import run_extract_meta
    from src.config.settings import init_settings
    from src.file.directory_paths import list_file_paths_under_directory
    from src.search.media_search import search_media_all_grouped, search_media_images_two_stage
    from src.search.query_preprocess import structure_user_query

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run meta-extract pipeline", allow_abbrev=False)
    parser.add_argument(
        "--env",
        choices=["dev", "prod"],
        default="dev",
        help="Configuration profile to use",
    )

    parser.add_argument(
        "--search",
        metavar="QUERY",
        default=None,
        help="통합 검색: 문서·음성·이미지·영상을 한 번에 조회하고 매체별로 나눈 JSON을 출력",
    )
    parser.add_argument(
        "--search-debug",
        action="store_true",
        help="ST 하이브리드 행에 candidate_count(임베딩 후보 행 수) 포함",
    )
    parser.add_argument(
        "--search-images",
        dest="search_images",
        metavar="QUERY",
        default=None,
        help="이미지 검색: ST(VLM 텍스트) 후보 + CLIP + BM25 하이브리드 재정렬",
    )
    parser.add_argument(
        "--image-search-alpha",
        type=float,
        default=0.65,
        metavar="A",
        help="하이브리드 가중: similarity = A*s_text + (1-A)*s_clip (기본 0.65)",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="FILE",
        help="처리할 미디어 파일 경로 (--search / --search-images 없을 때)",
    )
    parser.add_argument(
        "--file-list",
        dest="file_list",
        type=Path,
        metavar="PATH",
        help="한 줄에 하나씩 파일 경로가 적힌 텍스트 파일 (UTF-8)",
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        type=Path,
        metavar="DIR",
        help="디렉터리 아래 파일 경로를 모아 수집 (--file-list·FILE 과 병행 가능)",
    )
    parser.add_argument(
        "--dir-no-recurse",
        action="store_true",
        help="--input-dir 사용 시 바로 아래 파일만 (하위 폴더 미탐색)",
    )
    parser.add_argument(
        "--dir-include-hidden",
        action="store_true",
        help="--input-dir 사용 시 이름이 . 으로 시작하는 경로도 포함",
    )
    args = parser.parse_args()

    dotenv_path = PROJECT_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    init_settings(args.env)

    if args.search is not None and args.search_images:
        parser.error("--search 와 --search-images 는 동시에 쓸 수 없습니다.")

    if args.search is not None or args.search_images:
        a = args.image_search_alpha
        if not 0.0 <= a <= 1.0:
            parser.error("--image-search-alpha는 0 이상 1 이하여야 합니다.")

    if args.search is not None:
        if not str(args.search).strip():
            parser.error("--search QUERY에 빈 문자열을 쓸 수 없습니다.")

        def run_grouped_search_once(query: str) -> None:
            grouped = search_media_all_grouped(
                query,
                limit_per_bucket=20,
                image_search_alpha=args.image_search_alpha,
                debug=args.search_debug,
            )
            print(json.dumps(grouped, indent=2, ensure_ascii=False))

        run_grouped_search_once(str(args.search))
        print("\n반복 검색 모드입니다. 종료하려면 빈 입력/exit/quit를 입력하세요.")
        while True:
            try:
                next_query = input("\nsearch> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n종료합니다.")
                break
            if not next_query or next_query.lower() in {"exit", "quit"}:
                print("종료합니다.")
                break
            run_grouped_search_once(next_query)
    elif args.search_images is not None:
        if not str(args.search_images).strip():
            parser.error("--search-images QUERY에 빈 문자열을 쓸 수 없습니다.")
        sr = structure_user_query(str(args.search_images))
        rows = search_media_images_two_stage(
            sr["semantic_query"],
            sr.get("semantic_query_en") or "",
            alpha=args.image_search_alpha,
        )
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        files: list[str] = []
        if args.file_list is not None:
            if not args.file_list.is_file():
                parser.error(f"--file-list is not a file: {args.file_list}")
            text = args.file_list.read_text(encoding="utf-8")
            files.extend(line.strip() for line in text.splitlines() if line.strip())
        if args.input_dir is not None:
            root = args.input_dir.expanduser()
            if not root.is_dir():
                parser.error(f"--input-dir is not a directory: {args.input_dir}")
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
                "파일 경로가 없습니다. FILE 인자, --file-list, --input-dir 중 하나 이상을 주거나 "
                "--search / --search-images 로 검색하세요."
            )
        n_failed = run_extract_meta(files)
        sys.exit(1 if n_failed else 0)
