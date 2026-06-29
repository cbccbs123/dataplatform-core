"""VLM 요약 키워드 후처리(정규화·objects 승격) — 순수·결정적·LLM 0.

049: 캡션(``image_summarizer``)·reduce(``video_summarizer``) 가 v2 토글(``vlm_summary_prompt_v2``)
일 때만 거치는 검색 신호 보강 레이어다. VLM 출력 ``keywords`` 를 결정적으로 정규화(strip·casefold
dedup·generic 제거)하고, ``objects`` 를 키워드 후보로 승격해 검색에 쓸 구체 명사를 더 살린다.

헌법 3조(결정성): set 은 **dedup 용도로만** 쓰고 출력 순서는 first-seen(입력 등장 순서)을 보존한다 —
정렬하지 않아 VLM relevance 순서가 살고(BM25 무영향), 동일 입력 → 동일 출력이 보장된다. LLM·settings·
DB 를 일절 건드리지 않는 순수 함수라 단위 테스트가 격리된다(FR-402).
"""

from __future__ import annotations

# 검색 신호 0인 일반어(키워드/objects 에서 제거). settings 아닌 코드 상수(결정적·도메인 불가지).
# 토픽화 프롬프트(026)가 막아도 VLM 이 가끔 매체 일반어를 키워드로 흘리므로 후처리에서 한 번 더 거른다.
_GENERIC_KEYWORDS: frozenset[str] = frozenset(
    {
        "영상", "동영상", "비디오", "장면", "이미지", "사진", "그림", "화면",
        # 서술형 변형(VLM 이 "~입니다" 로 흘릴 때 2차 차단 — 토픽화 프롬프트가 1차).
        "영상입니다", "장면입니다", "사진입니다", "이미지입니다", "동영상입니다", "화면입니다",
    }
)


def normalize_keywords(keywords: list[object]) -> list[str]:
    """strip·casefold dedup·generic 제거·first-seen 순서 보존(결정적)."""
    seen: set[str] = set()
    out: list[str] = []
    for kw in keywords or []:
        k = str(kw).strip()
        kl = k.casefold()
        if not k or kl in seen or kl in _GENERIC_KEYWORDS:
            continue
        seen.add(kl)
        out.append(k)
    return out


def promote_objects_to_keywords(keywords: list[object], objects: list[object], *, limit: int) -> list[str]:
    """normalize_keywords(keywords) 뒤에 objects 를 키워드 후보로 합쳐(중복·generic 제외) limit 까지."""
    out = normalize_keywords(keywords)
    seen = {k.casefold() for k in out}
    for obj in objects or []:
        o = str(obj).strip()
        ol = o.casefold()
        if not o or ol in seen or ol in _GENERIC_KEYWORDS:
            continue
        seen.add(ol)
        out.append(o)
        if len(out) >= limit:
            break
    return out[:limit]
