"""
사용자 질의를 검색용 구조화 JSON(keywords, semantic_query 등)으로 바꾼다.

``src/search/media_search.py`` 의 하이브리드 검색에서 사용한다(현행 `asset_metadata`/`asset_embedding` 스키마 기반).

필요: 프로젝트 루트의 ``.env.dev`` / ``.env.prod`` (OPENAI_BASE_URL, OPENAI_API_KEY, META_MODEL 등)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

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
    client: Any | None = None,  # 테스트용 주입 seam(미주입=공통 seam 의 운영 클라이언트)
    model: str | None = None,
    tz_name: str = DEFAULT_TZ,
) -> dict[str, Any]:
    """사용자 질의를 검색용 구조화 JSON으로 변환한다.

    ``client``·``model`` 생략 시 공통 seam(``src.llm.client.complete_text``)이
    현재 설정의 온프레미스 LLM 클라이언트를 사용한다.
    ``client``/``model``을 주입하면 그대로 전달되어 테스트에서 네트워크 없이 동작한다.
    LLM 응답이 비어있으면 기본 empty dict를 반환하고,
    JSON 파싱 실패 시 ``semantic_query``에 원문을 넣어 반환한다(폴백).
    파싱은 됐으나 dict가 아니면(배열·스칼라) 기본 empty dict로 폴백한다.
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
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {**empty, "semantic_query": raw}
    # LLM이 스키마를 어겨 비-dict(JSON 배열·스칼라)를 내면 dict(...) 가 TypeError 를
    # 던지므로, 객체가 아니면 안전한 기본 dict 로 폴백한다(원본 질의를 semantic_query 로).
    if not isinstance(parsed, dict):
        return empty
    return parsed


# ── 029 FR-004: 검색 질의 명사구 정규화(021 FR-004 토글 개정 — 헌법 §3 결정성 제약 준수) ──────────
# 021 이 OS 검색 read 경로에서 LLM 질의 구조화를 제거한 실제 사유는 ``reference_dates_block`` 의
# ``datetime.now``(env 의존 입력)였다. 029 명사구 정규화는 그 비결정성을 **재도입하지 않는다** — 아래
# 프롬프트에 datetime/now/오늘/경로/랜덤 등 **env 의존 입력이 0**(순수 질의→명사구 매핑)이고, 호출은
# ``complete_json`` 단일 seam(temperature 기본 0)을 경유한다. 028 측정 근거대로 **명사구 정규화로만 한정**
# 한다 — 풀어쓰기·문장 확장·가설 답변(HyDE)은 역효과(측정)라 금지. 토글(SEARCH_OS_QUERY_NORM_ENABLED)
# on 일 때만 호출되며, off(기본)면 normalize_query 가 원문 passthrough 한다(027 바이트 동일·SC-001).
QUERY_NORM_PROMPT = """당신은 검색 질의 정규화기다. 사용자 한국어 질의를 검색에 가장 적합한 **핵심 명사구**로 바꿔라.
규칙:
- 질의 의도를 가장 잘 나타내는 핵심 명사구(주제어)만 남긴다.
- "찾아줘·보여줘·추천해줘·방법·하는 법" 같은 검색 지시어·구어체 군더더기를 제거한다.
- 풀어쓰기·문장 확장·가설 답변 금지 — 명사구 한 덩어리로만.
- 모달리티 단어(사진·영상·문서·이미지)는 넣지 마라.
반드시 아래 JSON 객체만 출력한다(설명 문장 금지).

스키마:
- query_norm: 문자열. 질의의 핵심 명사구. (예: "별 보는 방법" → "천체 관측", "물고기 잡는 법" → "낚시")

사용자 질의:
"""


def noun_phrase_query(query: str, *, client: Any | None = None) -> str:
    """검색 질의를 LLM 핵심 명사구로 정규화한다(029 FR-004·헌법 §3 결정성 제약).

    ``src.llm.client.complete_json`` **단일 seam**을 경유해 ``{"query_norm": "<명사구>"}`` JSON 스키마를
    요구하고 ``query_norm`` 을 파싱한다 — ``complete_text`` 의 평문은 seam 의 ``response_format=
    json_object`` 강제로 받을 수 없으므로 complete_json 의 스키마 호출을 쓴다. ``temperature`` 인자를
    전달하지 않아 seam 기본값 0 을 유지한다(결정 재현성 100%·numeric 비0 리터럴 미도입 — policy_gate).
    프롬프트(``QUERY_NORM_PROMPT``)는 **순수 질의→명사구 매핑**으로 datetime/경로/랜덤 등 env 의존 입력이
    0 이다(021 이 제거한 비결정성 재도입 차단).

    fail-safe(원문 폴백 → 027 경로): LLM 응답이 비었거나(``{}``), ``query_norm`` 키가 없거나, 빈 문자열·
    비-문자열이면 **원문 질의를 그대로 반환**한다 — 정규화 실패가 검색을 깨지 않게(SC-001). 빈/None 질의도
    정규화할 내용이 없어 LLM 호출 없이 원문 그대로 반환한다.

    ``client`` 미주입 시 공통 seam(``complete_json``)이 현 설정의 온프레미스 LLM 클라이언트를 쓴다 —
    테스트는 ``client`` 주입으로 네트워크 없이 결정성을 검증한다.
    """
    if not query or not query.strip():
        return query
    from src.llm.client import complete_json

    data = complete_json(QUERY_NORM_PROMPT + query.strip(), client=client)
    norm = data.get("query_norm")
    if isinstance(norm, str) and norm.strip():
        return norm.strip()
    return query  # fail-safe: 빈/스키마 위반 응답 → 원문 폴백(027 경로 — 결정성·회귀 0 보존)

