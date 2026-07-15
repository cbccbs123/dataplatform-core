"""
검색 질의 명사구 정규화(``noun_phrase_query``·``morph_noun_phrase_query``)의 단일 정의처.

069 US-C(037 잔재 철거): 037 이 OS 검색 read 경로에서 은퇴시킨 LLM 질의 구조화(``structure_user_query``
+ ``reference_dates_block`` 의 ``datetime.now`` 비결정 패턴)를 삭제했다 — 운영 호출 0. 검색 시점 질의
정규화는 query-norm 토글(072 형태소 / 075 gemma)이 담당한다.

필요(``llm`` 방식 선택 시에만): 온프레미스 LLM seam 설정(``.env.dev``/``.env.prod``). 형태소(``morph``·
기본) 경로는 nori ``_analyze`` 만 쓰므로 LLM/OPENAI 설정은 **불요**다.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

# ── 029 FR-004: 검색 질의 명사구 정규화(021 FR-004 토글 개정 — 헌법 §3 결정성 제약 준수) ──────────
# 021 이 OS 검색 read 경로에서 LLM 질의 구조화를 은퇴시킨 실제 사유는 옛 질의 구조화 프롬프트의
# ``datetime.now``(env 의존 입력) 비결정 패턴이었다(069 US-C 로 그 死코드 삭제). 029 명사구 정규화는
# 그 비결정성을 **재도입하지 않는다** — 아래
# 프롬프트에 datetime/now/오늘/경로/랜덤 등 **env 의존 입력이 0**(순수 질의→명사구 매핑)이고, 호출은
# ``complete_json`` 단일 seam(temperature 기본 0)을 경유한다. 028 측정 근거대로 **명사구 정규화로만 한정**
# 한다 — 풀어쓰기·문장 확장·가설 답변(HyDE)은 역효과(측정)라 금지. 토글(SEARCH_OS_QUERY_NORM_ENABLED)
# on 일 때만 호출되며, off(기본)면 normalize_query 가 원문 passthrough 한다(027 바이트 동일·SC-001).
# 076: 범용 strict 정규화 프롬프트. 측정(2026-07-14·자연어 334질의×gemma 실연관)에서 형태소 대비
# nDCG 0.438→0.531·recall 0.414→0.529 로 앞섰다. 핵심은 **원문 등장어만·의역/부연 금지·복합어 원형
# 보존** — 구 프롬프트가 예시로 의역("별 보는 방법"→"천체 관측")을 유도해 임베딩에 없던 노이즈("레시피·
# 강습·명소")를 보태던 것을 차단했다. 예시는 코퍼스 무관 도메인 중립(과적합 배제·범용성 실측 확인).
QUERY_NORM_PROMPT = """당신은 검색 질의 정규화기다. 사용자 질의를 검색 임베딩에 넣을 **핵심어**로 정규화한다.
원칙:
1. **원문에 실제로 등장한 단어만** 사용한다 — 없는 단어를 새로 지어내거나 유의어로 치환·부연하지 않는다(의역 금지).
2. 핵심 개체·주제 명사만 남긴다. 감상·상태·평가 수식어(예: 맛있는·멋진·예쁜·신비로운·좋은)를 제거한다.
   검색 지시·구어체(예: 찾아줘·추천·하는 법·어디서·알려줘)와 매체·형식어(예: 사진·영상·이미지·문서·자료·소스)도 제거한다.
3. 복합어·고유명사·외래어·영숫자 표기는 **분해하지 말고 원형 그대로** 둔다.
4. 핵심어 1~3개를 공백으로 구분해 한 덩어리로 출력한다. 한정어가 여럿이면 가장 핵심적인 것만 남긴다. 핵심 명사가 없으면 원문 질의를 그대로 둔다.
반드시 아래 JSON 객체만 출력한다(설명 문장 금지).
스키마:
- query_norm: 문자열(핵심어).
예시(원칙 시연):
- "아주 멋진 리소토 만드는 법 영상 보여줘" → "리소토"
- "노을 지는 해변 사진 예쁜 거 없나?" → "해변 노을"
- "왕초보가 볼만한 첼로 연습곡 추천해줘" → "첼로 연습곡"
- "GT-R 엔진 소리는 어디서 들어?" → "GT-R 엔진 소리"
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


# ── 072: 검색 질의 형태소 명사 정규화(029 LLM 정규화 대체 — nori _analyze·LLM 0·결정적) ──────────
def morph_noun_phrase_query(
    query: str,
    *,
    analyze_fn: Callable[[str], list[tuple[str, str]]],
    stopwords: Collection[str],
    noun_pos: Collection[str],
    min_word_tokens: int,
) -> str:
    """검색 질의를 nori 형태소 **명사구**로 정규화한다(072 FR-001~004·헌법 §3 결정성).

    029 ``noun_phrase_query``(gemma·LLM)를 대체하는 **LLM-free 정규화**다 — 측정(2026-07-13)에서 형태소
    명사추출+스톱워드가 자연어 nDCG@10 0.490→0.591 로 LLM 정규화(0.575)를 웃돌고, 검색시점 LLM 0·
    결정적(``_analyze`` 사전 기반)이라 지연도 없앤다. 효과의 본질은 kNN 입력 문장에서 껍데기어(영상·추천·
    방법) 제거이며, 복합어 토큰정확도·사전등록·재색인은 측정상 무효(범위 밖·072 spec).

    - ``analyze_fn``: ``text → [(token, pos)]`` **주입 seam**. 실제는 nori ``_analyze``(OS IO·
      ``opensearch_search.nori_analyze_tokens``) 래퍼이고, 단위 테스트는 가짜 함수를 주입해 OS 없이
      순수 검증한다(헌법 3조 — 결정적 순수 함수).
    - **판별(FR-001)**: 어절 수(공백 분리) < ``min_word_tokens`` 면 **단어 질의**로 보고 원문 그대로
      반환한다 — analyze_fn 미호출(단어 검색은 IO·지연 0). 빈/공백 질의도 원문 그대로.
    - **명사 추출(FR-002)**: ``noun_pos`` 품사만 남기고 **순서 보존·중복 제거**.
    - **스톱워드(FR-003)**: ``stopwords`` 의 모달리티어·지시성 명사 제거.
    - **폴백(FR-004)**: 남은 명사가 없으면 **원문 질의 그대로** 반환(정규화가 검색을 깨지 않게 — SC-001).
    """
    if not query or not query.strip():
        return query
    if len(query.split()) < min_word_tokens:
        return query  # 단어 질의 — 정규화 불요(analyze 미호출·지연 0)
    seen: set[str] = set()
    nouns: list[str] = []
    for token, pos in analyze_fn(query):
        if pos in noun_pos and token not in stopwords and token not in seen:
            seen.add(token)
            nouns.append(token)
    return " ".join(nouns) if nouns else query  # 빈결과 → 원문 폴백

