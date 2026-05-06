"""
사용자 질의를 검색용 구조화 JSON으로 바꾸는 샘플.

  python test.py "어제 공장 CCTV에서 안전모 안 쓴 사람"
  python test.py --env prod --query "..."

필요: 프로젝트 루트의 ``.env.dev`` / ``.env.prod`` (OPENAI_BASE_URL, OPENAI_API_KEY, META_MODEL 등)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from openai import OpenAI


from src.config.settings import get_current_settings, init_settings

DEFAULT_TZ = "Asia/Seoul"


def reference_dates_block(*, tz_name: str = DEFAULT_TZ) -> str:
    """LLM이 '어제' 등을 절대 날짜로 풀 수 있도록 기준 시각·오늘 날짜를 넣는다."""
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    today = now.date()
    return f"""[검색 기준 시각]
- timezone: {tz_name}
- now: {now.isoformat(timespec="seconds")}
- 오늘 날짜(today): {today.isoformat()}

상대 표현(어제, 그제, 지난주, 지난달, 이번 주, 최근 N일 등)은 **반드시 위 today·now를 기준**으로 풀어
date_start / date_end에 ``YYYY-MM-DD hh:mm:ss`` 형식만 사용해라. 범위가 없으면 둘 다 null.
"""


STRUCTURE_PROMPT_HEAD = """당신은 미디어 검색 시스템용 쿼리 분석기다. 사용자 한국어 질의를 아래 JSON 스키마에만 맞춰 출력해라.
추측이 필요하면 보수적으로 "unknown"을 쓰고, 설명 문장은 출력하지 마라. 반드시 JSON 객체만 출력한다.

"""


STRUCTURE_PROMPT_SCHEMA = """스키마:
- keywords: 문자열 배열. BM25/전문검색용 짧은 토큰(한글 가능). 없으면 []
- keywords_en: 문자열 배열. keywords를 영어 검색·CLIP 보강용으로 짧게 번역한 토큰. 없으면 []
- semantic_query: 문자열. 보존하되, 벡터 검색에 적합하도록 1문장 요약 형태로 재작성.  
    - "검색/찾아줘/보여줘/추천" 같은 검색 지시어를 절대 넣지 마라.
    - "이미지/영상/사진/문서/데이터/장면" 같은 모달리티 단어를 절대 넣지 마라.
    - 하나의 단어이 경우, 원본 그대로 사용.
    - 길이: 30~120자.
- semantic_query_en: 문자열. semantic_query를 의미 동일하게 영어로 번역하되,
  semantic_query의 금지어(검색 지시어/모달리티 단어/또는 관련 확장)를 그대로 지켜라.
- date_start: 문자열. SQL ``>=``에 쓸 **포함** 시작일 ``YYYY-MM-DD hh:mm:ss``. 날짜 조건이 없으면 "unknown"(불명확)
- date_end: 문자열. SQL ``<=``에 쓸 **포함** 종료일 ``YYYY-MM-DD hh:mm:ss``. 날짜 조건이 없으면 "unknown"(불명확)
  (예: 어제만 → date_start와 date_end를 **같은 날**(어제)로. 지난달 전체 → 그 달 1일~말일)

사용자 질의:
"""

def build_user_message(user_text: str, *, tz_name: str = DEFAULT_TZ) -> str:
    return (
        reference_dates_block(tz_name=tz_name)
        + "\n"
        + STRUCTURE_PROMPT_HEAD
        + STRUCTURE_PROMPT_SCHEMA
        + user_text.strip()
    )


def structure_user_query(
    user_text: str,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    tz_name: str = DEFAULT_TZ,
) -> dict[str, Any]:
    """``client``·``model`` 생략 시 ``init_settings`` 이후 현재 설정으로 OpenAI 클라이언트를 만든다."""
    if client is None or model is None:
        cfg = get_current_settings()
        if client is None:
            client = OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)
        if model is None:
            model = cfg.meta_model

    msg = build_user_message(user_text, tz_name=tz_name)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": msg}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    empty = {
        "keywords": [],
        "keywords_en": [],
        "semantic_query": user_text.strip(),
        "semantic_query_en": "",
        "date_start": None,
        "date_end": None,
    }
    if not raw:
        return empty
    try:
        return dict(json.loads(raw))
    except json.JSONDecodeError:
        return {**empty, "semantic_query": raw}


def main() -> None:
    parser = argparse.ArgumentParser(description="질의 구조화 샘플 (Gemma/OpenAI 호환 게이트웨이)")
    parser.add_argument("query", nargs="?", help="사용자 질의 (한국어)")
    parser.add_argument("--query", "-q", dest="query_opt", help="질의 (--query 로만 줄 때)")
    parser.add_argument(
        "--env",
        choices=["dev", "prod"],
        default="dev",
        help=".env.{dev|prod} 로드",
    )
    parser.add_argument(
        "--tz",
        default=DEFAULT_TZ,
        metavar="ZONE",
        help="상대 날짜 해석용 타임존 (기본 Asia/Seoul)",
    )
    args = parser.parse_args()

    text = (args.query or args.query_opt or "").strip()
    if not text:
        parser.error("질의를 인자로 주거나 --query 로 지정하세요.")

    init_settings(args.env)
    cfg = get_current_settings()

    client = OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)
    structured = structure_user_query(text, client=client, model=cfg.meta_model, tz_name=args.tz)

    print(json.dumps(structured, ensure_ascii=False, indent=2))