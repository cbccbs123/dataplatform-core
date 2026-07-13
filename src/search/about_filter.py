"""검색 OR-증거 필터 (spec 073 — 자연어 무관 L1 검색시점 층·순수·LLM 0).

융합·게이트·컷을 통과한 버킷 행에서, **질의 개체와의 증거가 전혀 없는 행**을 걸러낸다:

    행 유지 = amatch(질의 명사 ↔ 적재시 확정한 about 개체)
              OR kmatch(판별력 있는 질의 명사 ↔ keywords+파일명 텍스트)

왜 OR 인가(073 측정): about 단독은 1~3개로 희소해 표현이 조금 다르면 관련 자산까지 떨어졌다
(김치 영상의 about 이 '김장' → "김치" 질의에서 드롭·연관 −0.15). keywords 가 어휘 폭을 보완해
연관 손실이 −0.01~0.05 로 회복되면서 전체노출(@10) 무관 −4~7%p 는 유지됐다.

**판별력(usable) 명사 — 정적 사전 없이 질의별 상대 DF**: "씨름 기술"의 '기술'처럼 흔한 명사는
후보 행 대부분의 keywords 에 등장해 kmatch 를 무력화한다. 정적 불용어 사전은 끝없는 유지보수라
(사용자 정책), 대신 **그 질의의 후보 행 중 몇 %에 등장하는가**로 판별한다 — 후보의
``max_match_ratio`` 이하에만 등장하는 명사만 kmatch 에 쓴다(자산이 늘어도 자가 적응·결정적).

fail-safe 2종(FR-004 — 필터가 검색을 깨지 않게):
  - 매칭 행이 0 이면 **원 행 그대로**(패러프레이즈 질의처럼 어휘가 전혀 안 겹치는 정상 질의 보호).
  - 행들에 ``_about``/``_kwtext`` 내부키가 아예 없으면(구 문서·mock) 그대로 — 배선 전 하위호환.
"""

from __future__ import annotations

from typing import Any

from src.config.search_constants import ABOUT_FILTER_NOUN_MAX_MATCH_RATIO


def _amatch(nouns: list[str], about: list[str]) -> bool:
    """질의 명사 ↔ about 개체 양방향 부분일치(073 측정 규칙 그대로).

    완전일치는 길이 무관, 부분일치는 포함되는 쪽이 2자 이상일 때만("배"⊂"배드민턴" 오매칭 차단).
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

    ``query`` 는 검색이 실제 사용한 질의(072 query-norm on 이면 명사구)다 — 공백 split 이 명사
    리스트가 된다. 행의 ``_about``(적재시 확정 개체)·``_kwtext``(keywords+파일명 합본)는
    ``os_hit_to_row`` 가 실어주는 내부키다(응답 전 bucket_policy clean 이 제거).
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
