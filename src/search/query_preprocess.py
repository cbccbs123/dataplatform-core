"""
사용자 질의를 검색용 구조화 JSON(keywords, semantic_query 등)으로 바꾼다.

``src/search/media_search.py`` 의 하이브리드 검색에서 사용한다(검색 API는 Phase 2에서 asset_* 로 재배선 예정).

필요: 프로젝트 루트의 ``.env.dev`` / ``.env.prod`` (OPENAI_BASE_URL, OPENAI_API_KEY, META_MODEL 등)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.config.settings import init_settings

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
- keywords: 문자열 배열. 질의에서 뽑은 짧은 토큰(한글 가능); 검색 코드는 주로 넘긴 **전체 질의 문자열**로 FTS를 구성하므로 간접 참고용. 없으면 []
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
    client: Any | None = None,
    model: str | None = None,
    tz_name: str = DEFAULT_TZ,
) -> dict[str, Any]:
    """사용자 질의를 검색용 구조화 JSON으로 변환한다.

    ``client``·``model`` 생략 시 공통 seam(``src.llm.client.complete_text``)이
    현재 설정의 온프레미스 LLM 클라이언트를 사용한다.
    ``client``/``model``을 주입하면 그대로 전달되어 테스트에서 네트워크 없이 동작한다.
    LLM 응답이 비어있으면 기본 empty dict를 반환하고,
    JSON 파싱 실패 시 ``semantic_query``에 원문을 넣어 반환한다(폴백).
    """
    from src.llm.client import complete_text

    msg = build_user_message(user_text, tz_name=tz_name)
    raw = complete_text(msg, model=model, client=client)
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

