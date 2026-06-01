"""F-4.3 하이브리드 검색 CLI 진입점 — 한국어 질의로 asset_* 인덱스를 검색해 결과를 출력한다.

예) python -m src.app.run_search --env dev --query "작년 워크숍 발표자료" --modalities text,image

검색은 파이프라인(run_ingest/run_relations) 밖의 라이브러리 계층이라 레지스트리·도메인 팩
import 부수효과가 필요 없다. 질의 구조화 LLM 은 공통 seam(``src.llm.client``)을, 임베딩/FTS 는
현행 ``asset_metadata``/``asset_embedding`` 을 쓴다.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.search.search_service import search_hybrid


def _parse_modalities(raw: str | None) -> list[str] | None:
    """``--modalities`` 콤마 구분 문자열 → 라벨 리스트. 미지정/공백이면 None(전체 버킷)."""
    if not raw:
        return None
    items = [m.strip() for m in raw.split(",") if m.strip()]
    return items or None


def _run(
    args: argparse.Namespace,
    *,
    search_fn: Callable[..., dict[str, Any]] = search_hybrid,
) -> dict[str, Any]:
    """파싱된 인자를 검색 서비스 호출로 매핑한다. ``search_fn`` 은 테스트 주입 seam."""
    return search_fn(
        args.query,
        modalities=_parse_modalities(args.modalities),
        limit_per_bucket=args.limit,
        text_hybrid_alpha=args.alpha,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="하이브리드 검색 (asset_* 인덱스)")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--query", required=True, help="검색 질의(한국어)")
    parser.add_argument(
        "--modalities",
        default=None,
        help="콤마 구분 모달리티(text,audio,image,video). 미지정=전체",
    )
    parser.add_argument("--limit", type=int, default=20, help="버킷당 최대 결과 수")
    parser.add_argument("--alpha", type=float, default=0.75, help="텍스트 하이브리드 임베딩 가중(0~1)")
    return parser


# 런타임 순서(run_ingest 와 동일): 1) load_dotenv(.env.{env}, override=False) →
# 2) init_settings(env)(필수 환경변수 검증) → 3) 검색 실행. LLM/임베딩 클라이언트는 첫 사용 시 지연 초기화.
def main() -> int:
    from dotenv import load_dotenv

    from src.config.settings import init_settings

    args = _build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    result = _run(args)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
