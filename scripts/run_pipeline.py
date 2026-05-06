from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.run_embedding import run_embedding
from src.config.settings import init_settings
from src.app.run_extract_meta import run_extract_meta
from src.search.media_search import (
    search_media_images_by_text,
    search_media_images_two_stage,
    search_media_text_items,
)
from src.search.query_preprocess import structure_user_query

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run meta-extract pipeline")
    parser.add_argument(
        "--env",
        choices=["dev", "prod"],
        default="dev",
        help="Configuration profile to use",
    )

    parser.add_argument(
        "--search",
        metavar="QUERY",
        help="SentenceTransformer 텍스트 임베딩으로 문서(텍스트 계열) 검색",
    )
    parser.add_argument(
        "--search-images",
        dest="search_images",
        metavar="QUERY",
        help="이미지 검색: 기본은 ST(VLM 텍스트) 후보 + CLIP 재정렬, --search-images-clip-only면 CLIP만",
    )
    parser.add_argument(
        "--search-images-clip-only",
        action="store_true",
        help="이미지 검색을 CLIP 코사인만 사용 (레거시와 동일한 단일 축)",
    )
    parser.add_argument(
        "--image-search-alpha",
        type=float,
        default=0.65,
        metavar="A",
        help="하이브리드 가중: similarity = A*s_text + (1-A)*s_clip (기본 0.65)",
    )
    args = parser.parse_args()

    dotenv_path = PROJECT_ROOT / f".env.{args.env}"
    print(dotenv_path)
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    init_settings(args.env)

    if args.search_images and not args.search_images_clip_only:
        a = args.image_search_alpha
        if not 0.0 <= a <= 1.0:
            parser.error("--image-search-alpha는 0 이상 1 이하여야 합니다.")


    if args.search_images:
        search_result = structure_user_query(args.search_images)
        print("search_result: ", json.dumps(search_result, indent=4, ensure_ascii=False))
        if args.search_images_clip_only:
            rows = search_media_images_by_text(search_result["semantic_query_en"])
        else:
            rows = search_media_images_two_stage(
                search_result["semantic_query"],
                search_result["semantic_query_en"],
                alpha=args.image_search_alpha,
            )
        print(json.dumps(rows, indent=4, ensure_ascii=False))
    elif args.search:
        search_result = structure_user_query(args.search)
        print("search_result: ", json.dumps(search_result, indent=4, ensure_ascii=False))
        print("1) search_media_text_items: ", json.dumps(search_media_text_items(args.search, limit=3), indent=4, ensure_ascii=False))   

        print("2) search_media_text_items: ", json.dumps(search_media_text_items(search_result["semantic_query"], limit=3), indent=4, ensure_ascii=False))   
    else:
        files = [
            '/Users/cbccbs/Downloads/test data/경찰청 대구광역시경찰청_대구경찰청 홍보영상(3초의 배려)_20260417.mp4',
            '/Users/cbccbs/Downloads/IMG_1941.MOV',   
            '/Users/cbccbs/Downloads/IMG_1942.MOV',
            '/Users/cbccbs/Downloads/[본문1] 2025-IITP-034 고속병렬파일시스템_연구개발계획서_v0.33.pdf',
            '/Users/cbccbs/Downloads/test data/한국가스공사_인천해저배관 사설항로표지 위치도 및 현황조서_20110101.docx',
            '/Users/cbccbs/Downloads/posting_10_5.webp',
            '/Users/cbccbs/Downloads/test data/부산관광공사_부산원아시아페스티벌  홈페이지 콘텐츠 관리_20200825/파크콘서트-김범수.jpg',
            '/Users/cbccbs/Downloads/test data/부산관광공사_부산원아시아페스티벌  홈페이지 콘텐츠 관리_20200825/0_thum.jpg',
            '/Users/cbccbs/Downloads/188683460-anatomy-of-human-body-with-organs-3d-illustration.avif',
            '/Users/cbccbs/Downloads/test data/한국인터넷진흥원_전화상담 가명정보 현황_20251208/전화상담데이터 (1).txt',
            '/Users/cbccbs/Downloads/test data/대전교통공사_열차안내방송_20251127/갑천.mp3',
            '/Users/cbccbs/Downloads/20260401_고속병렬파일시스템_1단계_2차년도(2026)회의자료.pptx',
            '/Users/cbccbs/Downloads/test data/문화재 현황 (2014.6월 기준)..xlsx',
            '/Users/cbccbs/Downloads/(공개용) 고속 병렬 파일시스템_제안평가_발표자료_한국전자기술연구원.pdf',
            '/Users/cbccbs/Downloads/test data/전북특별자치도 남원시_1세단위 인구현황(2026년 3월 기준).json',
        ]
        run_extract_meta(files)
    #run_embedding(files)