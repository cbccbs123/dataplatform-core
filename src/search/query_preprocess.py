"""검색 질의를 핵심 명사구로 줄이는 두 가지 방법의 단일 정의처.

``morph_noun_phrase_query``(기본)는 형태소 분석기로 명사만 뽑아 LLM 없이 처리하고,
``noun_phrase_query``는 온프레미스 LLM 에 맡긴다. 어느 쪽을 쓸지는 설정 토글이 정하며, 형태소
경로는 LLM 설정이 전혀 필요 없다.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

# ── LLM 정규화 프롬프트 ──────────────────────────────────────────────────────
# 이 프롬프트가 지켜야 할 두 가지.
#   ① **결정성**: 날짜·경로·난수 같은 환경 의존 입력을 넣지 않는다. 넣는 순간 같은 질의가 실행할
#      때마다 다른 결과를 내고, 검색 결과를 재현할 수 없게 된다.
#   ② **의역 금지**: 원문에 있는 단어만 남긴다. "별 보는 방법"을 "천체 관측"으로 바꾸면 문서
#      임베딩에 없던 말이 질의에 섞여 엉뚱한 결과를 끌고 온다(실제로 그랬다).
# 예시를 도메인 중립으로 둔 것도 같은 이유다 — 코퍼스에 맞춘 예시는 그 코퍼스에만 잘 듣는다.
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

    JSON 스키마(``{"query_norm": "..."}``)를 요구하는 호출을 쓴다 — 평문 응답은 seam 이 받지 않는다.
    temperature 를 넘기지 않아 기본값 0 이 유지되므로 같은 질의는 늘 같은 명사구가 된다.

    Args:
        query: 사용자 질의 원문. 비었거나 공백뿐이면 **LLM 을 부르지 않고** 그대로 돌려준다.
        client: **테스트용 LLM 클라이언트 주입 seam** — 미주입이면 설정의 온프레미스 클라이언트를
            쓴다. 주입하면 네트워크 없이 결정성을 검증할 수 있다.

    Returns:
        정규화된 핵심 명사구. 응답이 비었거나 스키마를 어기면 **원문 질의 그대로**(정규화 실패가
        검색을 깨지 않게 하는 안전장치).
    """
    if not query or not query.strip():
        return query
    from src.llm.client import complete_json

    data = complete_json(QUERY_NORM_PROMPT + query.strip(), client=client)
    norm = data.get("query_norm")
    if isinstance(norm, str) and norm.strip():
        return norm.strip()
    return query  # 응답이 비었거나 스키마를 어기면 원문으로 — 정규화 실패가 검색을 깨면 안 된다


# ── 형태소 정규화(기본 경로) ─────────────────────────────────────────────────
def morph_noun_phrase_query(
    query: str,
    *,
    analyze_fn: Callable[[str], list[tuple[str, str]]],
    stopwords: Collection[str],
    noun_pos: Collection[str],
    min_word_tokens: int,
) -> str:
    """검색 질의에서 **명사만 남겨** 핵심어 나열로 바꾼다(LLM 없이·결정적).

    "무선 충전기 추천 영상"에서 '추천'·'영상' 같은 껍데기어를 떼면, 벡터 검색이 실제 내용어에만
    반응한다. 이것이 효과의 전부다 — 사전 기반이라 같은 질의는 늘 같은 결과이고 지연도 없다.

    처리 순서: 짧은 질의는 그대로 통과 → 명사 품사만 추출(순서 보존·중복 제거) → 불용어 제거 →
    남은 게 없으면 원문 반환.

    Args:
        query: 사용자 질의 원문.
        analyze_fn: ``text → [(토큰, 품사)]`` **주입 seam**. 운영은 nori ``_analyze`` 래퍼,
            단위 테스트는 가짜 함수를 넣어 OpenSearch 없이 검증한다.
        stopwords: 제거할 명사(모달리티어·지시성 명사). 비어 있어도 된다.
        noun_pos: 남길 품사 태그 집합. 여기 없는 품사는 버린다.
        min_word_tokens: **이 어절 수 미만이면 정규화하지 않는다** — 단어 질의는 이미 핵심어라
            건드릴 필요가 없고, ``analyze_fn`` 을 부르지 않아 지연도 0 이다.

    Returns:
        공백으로 이어붙인 명사구. 짧은 질의·빈 결과면 원문 그대로.
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

