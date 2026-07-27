"""질의와 아무 증거도 겹치지 않는 행을 걸러낸다(순수 함수·LLM 호출 없음).

**흐름에서의 위치**: 융합·게이트·컷을 모두 통과한 버킷 행에 마지막으로 적용된다. 점수는 건드리지
않고 **드롭만** 한다.

    행 유지 = 질의 명사 ↔ 적재 때 확정한 개체  OR  판별력 있는 질의 명사 ↔ 키워드·파일명

**왜 OR 인가**: 개체(about)만 보면 너무 희소하다 — 자산당 1~3개뿐이라 표현이 조금만 달라도
관련 자산이 떨어진다(김치 영상의 개체가 '김장' 하나면 "김치" 질의에서 탈락). 키워드가 어휘 폭을
보완해 그 손실을 메운다.

**판별력 있는 명사만 쓰는 이유**: "씨름 기술"의 '기술'처럼 흔한 말은 후보 대부분의 키워드에 있어
매칭을 무력화한다. 정적 불용어 사전은 유지보수가 끝없으므로, 대신 **그 질의의 후보 행 중 몇 %에
등장하는가**로 판별한다 — 자산이 늘어도 자가 적응하고 결과는 결정적이다.

안전장치 둘 — 어느 쪽이든 **원본을 그대로 돌려준다**:
  - 살아남은 행이 0이면(어휘가 전혀 안 겹치는 정상 질의를 필터가 죽이지 않게)
  - 행에 증거 키(``_about``·``_kwtext``)가 아예 없으면(옛 색인·목 데이터)
"""

from __future__ import annotations

from typing import Any

from src.config.search_constants import ABOUT_FILTER_NOUN_MAX_MATCH_RATIO


def _amatch(nouns: list[str], about: list[str]) -> bool:
    """질의 명사 ↔ about 개체 양방향 부분일치(073 측정 규칙 그대로).

    완전일치는 길이 무관, 부분일치는 포함되는 쪽이 2자 이상일 때만("배"⊂"배드민턴" 오매칭 차단).

    Args:
        nouns: 질의에서 뽑은 명사 목록.
        about: 그 행이 적재 때 확정한 개체 목록.

    Returns:
        하나라도 겹치면 True. 둘 중 하나가 비면 False.
    """
    return any(
        n == a or (len(n) >= 2 and n in a) or (len(a) >= 2 and a in n)
        for n in nouns
        for a in about
    )


def about_or_filter(
    rows: list[dict[str, Any]],
    query: str,
    *,
    max_match_ratio: float = ABOUT_FILTER_NOUN_MAX_MATCH_RATIO,
) -> list[dict[str, Any]]:
    """OR-증거 필터(순수·결정적·행 순서 보존 — 드롭만, 재정렬·점수 변경 없음).

    Args:
        rows: 융합·컷을 통과한 버킷 행들. 각 행의 ``_about``(적재 때 확정한 개체)·``_kwtext``
            (keywords+파일명 합본)를 증거로 본다 — ``os_hit_to_row`` 가 싣고 응답 직전에 제거되는
            내부 키다.
        query: 검색이 **실제로 사용한** 질의(질의 정규화가 켜져 있으면 명사구). 공백으로 쪼갠
            것이 명사 목록이 된다.
        max_match_ratio: 흔한 명사를 걸러내는 기준. 후보 행의 이 비율을 **넘겨** 등장하는 명사는
            변별력이 없다고 보고 keywords 매칭에서 뺀다(정적 불용어 사전 없이 질의마다 자가 적응).

    Returns:
        살아남은 행(입력 순서·점수 그대로). **두 경우엔 원본을 그대로 돌려준다** — 증거 키가 아예
        없는 구 색인이거나, 필터 결과가 0건일 때(어휘가 안 겹치는 정상 질의를 죽이지 않기 위한
        안전장치).
    """
    nouns = [w for w in (query or "").split() if w]
    if not rows or not nouns:
        return rows
    # 하위호환 가드: 어떤 행에도 증거 키가 없으면(구 색인·mock) 필터할 근거가 없다 — passthrough.
    if not any(("_about" in r) or ("_kwtext" in r) for r in rows):
        return rows

    # 판별력 명사 선별(질의별 상대 DF): 후보 행의 max_match_ratio 초과에 등장하는 명사는 흔한
    # 명사('기술'·'풍경')로 보고 kmatch 에서 제외한다. amatch(about)는 개체가 이미 정제돼 있어 전 명사 사용.
    # ⚠️ 1자 명사는 kmatch(부분일치)에서 제외한다 — "배"⊂"택배" 우발 매칭 차단(_amatch 의 len≥2 가드와
    # 동일 원칙·리뷰 지적). 1자 명사는 amatch 의 완전일치(n == a)로만 증거에 기여한다.
    n_rows = len(rows)
    usable: list[str] = []
    for n in nouns:
        if len(n) < 2:
            continue
        hit = sum(1 for r in rows if n in str(r.get("_kwtext") or ""))
        if hit / n_rows <= max_match_ratio:
            usable.append(n)

    kept: list[dict[str, Any]] = []
    for r in rows:
        about = [str(a) for a in (r.get("_about") or [])]
        kwtext = str(r.get("_kwtext") or "")
        if _amatch(nouns, about) or any(n in kwtext for n in usable):
            kept.append(r)
    # fail-safe: 전멸이면 원 행 유지 — 어휘가 전혀 안 겹치는 정상 질의(패러프레이즈)를 필터가 죽이지 않게.
    return kept if kept else rows


__all__ = ["about_or_filter"]
