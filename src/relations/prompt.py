"""LLM **관계 제안**용 프롬프트 문자열 생성.

구성
    1. **소스 요약·타입** — LLM이 소스 맥락을 알도록.
    2. **임베딩 후보 JSON** — ``target_media_item_id``, 요약, ``embedding_similarity`` (1에 가까울수록 유사).
    3. **활성 relation_kind 카탈로그** — ``relation_kinds_catalog`` (DB ``fetch_active_relation_kinds`` 결과).
       ``type_code`` 는 ``relation_kind.kind_code``. 주제(topic)는 카탈로그가 아니라 엣지의 ``topic_ko``/``topic_en`` 으로 LLM이 자유 기입.
    4. **선택 가이드** — ``RELATION_KIND_HINTS_KO`` + 카탈로그에 없는 코드는 DB ``description`` 일부 표시.

출력 규격
    LLM 은 **JSON 객체 하나**만 반환하도록 지시한다(코드 펜스 금지).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

# relation_type_code(= kind_code) 선택 힌트: DB에 없을 때 프롬프트에 보조 설명으로 쓴다.
RELATION_KIND_HINTS_KO: dict[str, str] = {
    "same_domain": "같은 주제·분야·도메인으로 묶일 때(예: 둘 다 게임, 둘 다 교통)",
    "same_series": "같은 시리즈·연작·브랜드 라인업 등 **연속·묶음**일 때",
    "duplicate_near": "임베딩 유사도로 가져온 후보처럼 **내용·장면·주제 근접**할 때",
    "references": "명시적 인용·링크·제목 참조 등 **참조** 관계일 때",
    "derived_from": "한쪽이 다른쪽에서 **파생·생성**된 관계일 때",
}


def _build_relation_kind_guide(catalog: Sequence[Mapping[str, Any]]) -> str:
    """
    카탈로그에 등장하는 ``type_code`` 마다 한 줄 힌트(또는 DB 설명)를 붙인 Markdown 블록.

    ``duplicate_near`` 와 ``same_domain`` 이 동시에 있으면 혼동 방지 문구를 추가한다.
    """
    codes = sorted({str(r.get("type_code", "")).strip() for r in catalog if str(r.get("type_code", "")).strip()})
    if not codes:
        return ""
    lines: list[str] = []
    for code in codes:
        hint = RELATION_KIND_HINTS_KO.get(code)
        if hint:
            lines.append(f"- ``{code}``: {hint}")
        else:
            desc = str(next((r.get("description") for r in catalog if str(r.get("type_code")) == code), "") or "")
            lines.append(f"- ``{code}``: (DB 설명) {desc[:200]}")
    body = "\n".join(lines)
    anti_dup = ""
    if "duplicate_near" in codes and "same_domain" in codes:
        anti_dup = (
            "\n\n**구분:** 단순히 주제가 같으면 ``same_domain`` , 유사도·근접 후보라면 ``duplicate_near`` 를 우선 고려한다."
        )
    return f"""### relation_kind (= ``type_code``) 선택 가이드
아래는 **왜 두 미디어가 연결되는지**에 대한 거친 분류다. **업종·소재(의료·게임 등)** 는 여기서 고르지 말고 ``topic_ko`` / ``topic_en`` 에 넣는다.

{body}{anti_dup}
"""


def build_relation_proposal_prompt(
    *,
    source_summary: str,
    source_media_type: str,
    candidates: Sequence[Mapping[str, Any]],
    relation_kinds_catalog: Sequence[Mapping[str, Any]],
) -> str:
    """
    관계 제안 전체 프롬프트(단일 문자열)를 조립한다.

    Args:
        source_summary: 소스 자산 요약(길이 상한은 본문에서 잘라 씀).
        source_media_type: 매체 타입 구분용 문자열.
        candidates: ``find_embedding_candidates`` 결과 행들.
        relation_kinds_catalog: **active** ``relation_kind`` 목록(``type_code``/``type_name``/``description``/``is_symmetric``).
    """
    cand_lines: list[str] = []
    for c in candidates:
        cand_lines.append(
            json.dumps(
                {
                    "target_media_item_id": c["id"],
                    "file_uri": c["file_uri"],
                    "media_type": c["media_type"],
                    "summary": (c["summary"] or "")[:500],
                    "embedding_similarity": round(c["emb_score"], 6),
                },
                ensure_ascii=False,
            )
        )
    candidates_block = "\n".join(cand_lines) if cand_lines else "(후보 없음)"

    if relation_kinds_catalog:
        catalog_block = json.dumps(
            [
                {
                    "type_code": str(r.get("type_code", "")),
                    "type_name": str(r.get("type_name", "")),
                    "description": str(r.get("description", "")),
                }
                for r in relation_kinds_catalog
            ],
            ensure_ascii=False,
            indent=2,
        )
        selection_guide = _build_relation_kind_guide(relation_kinds_catalog)
        catalog_rules = f"""아래는 현재 DB의 **활성 relation_kind** 목록(JSON). 각 엣지의 ``relation_type_code`` 는 **반드시** 아래 ``type_code``(= 관계 종류 코드) 중 하나와 **완전히 동일**해야 한다(소문자).

{catalog_block}

{selection_guide}
"""
    else:
        catalog_rules = "현재 DB에 프롬프트용 활성 relation_kind 이 없다. 이 경우 엣지를 비우거나 ``edges``: [] 로 반환해도 된다."

    example_codes = {str(r.get("type_code", "")) for r in relation_kinds_catalog} if relation_kinds_catalog else set()
    if "same_domain" in example_codes:
        example_relation_type = "same_domain"
    elif "duplicate_near" in example_codes:
        example_relation_type = "duplicate_near"
    else:
        example_relation_type = (
            next(iter(sorted(example_codes)), "same_domain") if example_codes else "same_domain"
        )

    return f"""너는 멀티모달 미디어 간 관계를 표현하는 JSON만 출력하는 도우미다.

규칙:
- 반드시 JSON 객체 하나만 출력한다. 코드 블록·설명 문장 금지.
- 각 엣지의 ``relation_type_code`` 는 **관계 종류**(왜 연결되는지: same_domain, duplicate_near 등). 카탈로그의 ``type_code`` 와 **완전히 동일**해야 한다.
{catalog_rules}
- 기본은 카탈로그 코드 사용. 다만 목록에 정말 맞는 관계 종류가 없을 때만, 소문자 ``[a-z][a-z0-9_]*`` 새 코드를 제안할 수 있다. 서버는 이를 ``relation_kind`` 에 비활성으로 기록하고 검토 큐에 넣는다.
- **토피(주제·도메인):** ``topic_ko`` 는 한국어 **한 단어 카테고리**로 필수(예: ``의료``, ``교통``, ``여행``, ``반도체``, ``부동산``). ``topic_en`` 은 같은 의미의 짧은 영어 카테고리(예: ``medical``, ``traffic``).
- **서브토픽(세부 주제, 선택):** ``subtopic_ko`` / ``subtopic_en`` 은 **토피와 같이 한 단어**로 쓴다(한국어는 **한 어절**·공백 금지, 영어는 **한 토큰**·소문자 권장). 맥락이 있으면 **비우지 말고** 한 단어로라도 채우는 것을 권장한다.
  - **문서·파일·상품·인물 등 고유명**은 subtopic 에 넣지 말고 ``reason`` 등에 적는다.
- ``reason``: 연결 근거 한 줄(한국어 권장). 고유명·세부 맥락은 여기에 둬도 된다.

소스 요약: {source_summary[:1200]}
소스 매체 타입: {source_media_type}

후보 목록(embedding_similarity 는 1에 가까울수록 유사):
{candidates_block}

출력 형식 예:
{{"edges":[
  {{"target_media_item_id":123,"relation_type_code":"{example_relation_type}","confidence":0.75,
    "topic_ko":"게임","subtopic_ko":"플랫폼","topic_en":"gaming","subtopic_en":"platform",
    "reason":"유사 후보·주제 일치"
  }}
]}}
"""
