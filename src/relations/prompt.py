"""LLM **관계 제안**용 프롬프트 문자열 생성.

구성
    1. **소스 요약·타입** — LLM이 소스 맥락을 알도록.
    2. **임베딩 후보 JSON** — ``target_media_item_id``, 요약, ``embedding_similarity`` (1에 가까울수록 유사).
    3. **활성 relation_kind 카탈로그** — ``relation_kinds_catalog`` (DB ``fetch_active_relation_kinds`` 결과).
       ``type_code`` 는 ``relation_kind.kind_code``. **관계 종류**(왜 연결되는지)만 여기서 고른다.
    4. **선택 가이드** — ``RELATION_KIND_HINTS_KO`` + 카탈로그에 없는 코드는 DB ``description`` 일부 표시.
    5. **닫힌 topic 분류체계 목록** — ``topic_ko`` 는 ``taxonomy_seed.json`` 의 **27+미분류**에서 하나 선택
       (spec 058 v2·FR-401v2). subtopic 은 열린 층이라 자유 기입(구체 주제어).

출력 규격
    LLM 은 **JSON 객체 하나**만 반환하도록 지시한다(코드 펜스 금지).

topic 지시 변경(spec 058 v2·FR-401v2·2026-07-07 닫힌 분류체계 전환)
    과거에는 ``topic_ko`` 를 한 단어 카테고리로 **자유 기입**시켰으나(동의어 난립),
    이제 ``taxonomy_seed.json`` 의 닫힌 27+미분류 목록을 프롬프트에 통째로 주입하고 **그 중 하나를
    선택**하게 한다(확신 없으면 ``미분류``). 관계종류(kind)·후보·경로신호·JSON 출력 지시는 불변.
    중복 제거는 schema.parse_llm_edges 에서 처리.
"""

from __future__ import annotations

import functools
import json
import posixpath
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# ── 대주제 닫힌 목록 ─────────────────────────────────────────────────────────
# 대주제는 자유 기입이 아니라 **목록에서 고르는 것**이다. 목록이 작아서 프롬프트에 통째로 넣을 수
# 있고, 그래서 생성 시점부터 정본 어휘로 수렴한다(세부주제는 열린 층이라 자유 기입).
# 이 파일이 시드·정본화 로직과 **같은 목록 파일**을 본다 — 셋이 어긋나면 분류가 깨진다.
# 위치가 ``src/`` 안인 이유: 패키징이 ``src`` 만 담으므로 여기 없으면 런타임에 파일을 못 찾는다.
_TAXONOMY_SEED_PATH = Path(__file__).resolve().parent / "taxonomy_seed.json"


@functools.lru_cache(maxsize=1)
def _load_taxonomy_topics() -> tuple[tuple[str, str], ...]:
    """taxonomy_seed.json → ``((topic_ko, topic_en), ...)`` (파일 순서 보존·결정적).

    파일 I/O 는 ``lru_cache`` 로 1회만 수행한다(프롬프트 조립마다 재읽기 방지). 시드 파일이
    정본이므로 같은 파일 → 같은 목록(재현성). 라벨은 ``str()`` 강제(graph_query 관례).

    시드 파일이 없으면(패키징 누락 등) 무엇이 빠졌는지 명확한 한국어 에러로 실패시킨다 —
    조용한 빈 목록으로 프롬프트가 망가지지 않도록.

    Returns:
        ``((topic_ko, topic_en), ...)`` 튜플. 파일에 적힌 순서를 그대로 유지한다.

    Raises:
        FileNotFoundError: 시드 파일이 없을 때(패키징에서 빠진 경우가 대표적).
    """
    try:
        with open(_TAXONOMY_SEED_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:  # 패키징 누락·경로 오류 시 원인을 즉시 드러낸다.
        raise FileNotFoundError(
            f"topic 닫힌 분류체계 시드를 찾을 수 없습니다: {_TAXONOMY_SEED_PATH}. "
            "이 파일은 src/relations 패키지에 포함돼야 한다(PR #81 이관·단일 출처)."
        ) from e
    return tuple((str(t["topic_ko"]), str(t["topic_en"])) for t in data["topics"])


def _build_topic_taxonomy_block() -> str:
    """프롬프트의 topic 지시부에 넣을 **닫힌 목록** Markdown 블록을 만든다.

    목록을 통째로 보여줘 LLM이 그 안에서만 고르게 한다(자유 기입 방지).

    Returns:
        ``- ``주제``(topic_en)`` 형태의 여러 줄 문자열.
    """
    return "\n".join(f"- ``{ko}`` ({en})" for ko, en in _load_taxonomy_topics())

# 파일명·폴더 신호는 **보조**라는 것을 LLM 에 명시하는 블록. 이름만 비슷하고 내용이 무관한 쌍을
# 연작·파생으로 단정하지 않게 한다. 경로로 들어온 후보는 유사도가 0.0 인데, 그것을 "안 닮았다"로
# 오해하지 않도록 "경로 신호"라는 표식을 함께 보여준다.
_PATH_SIGNAL_GUIDE_KO = """### 파일명·폴더 경로 신호 가이드 (보조)
후보에는 **임베딩 유사도(embedding_similarity)** 외에 **파일명·폴더 신호**가 함께 올 수 있다.
``signal`` 이 ``경로 신호`` 인 후보는 동일 폴더이거나 파일명 stem 이 일치/근접해서 추가된 것으로,
``embedding_similarity`` 가 0.0 이어도 **비유사가 아니라** 임베딩 점수가 없을 뿐이다(파일명·폴더로 매칭).

흔한 경로 패턴(파일명·폴더):
- **연작(same_series):** ``강의_1부`` / ``강의_2부``, ``manual_v1`` / ``manual_v2`` 처럼 같은 stem + 순번/버전.
- **파생(derived_from):** ``report`` → ``report_summary``(요약), ``원문`` → ``번역`` / ``전사`` 처럼 한쪽이 다른쪽에서 생성됨.
- **참조(references):** 제목·파일명이 다른 자산을 명시적으로 가리킬 때.

**가드(중요):** 파일명·폴더 신호는 **보조이며**, 후보 요약·내용이 실제로 합치할 때만
``same_series`` / ``derived_from`` / ``references`` 를 고른다. 파일명만 비슷하고 내용이 무관하면
그 종류를 고르지 말 것(오탐 방지). 내용 근거가 없으면 엣지를 만들지 않아도 된다.
"""

# relation_type_code(= kind_code) 선택 힌트: DB에 없을 때 프롬프트에 보조 설명으로 쓴다.
RELATION_KIND_HINTS_KO: dict[str, str] = {
    "same_domain": "같은 주제·분야·도메인으로 묶일 때(예: 둘 다 게임, 둘 다 교통)",
    "same_series": "같은 시리즈·연작·브랜드 라인업 등 **연속·묶음**일 때",
    "duplicate_near": "임베딩 유사도로 가져온 후보처럼 **내용·장면·주제 근접**할 때",
    "references": "명시적 인용·링크·제목 참조 등 **참조** 관계일 때",
    "derived_from": "한쪽이 다른쪽에서 **파생·생성**된 관계일 때",
}


def _build_relation_kind_guide(catalog: Sequence[Mapping[str, Any]]) -> str:
    """카탈로그의 ``type_code`` 마다 한 줄 힌트를 붙인 선택 가이드 블록을 만든다.

    ``duplicate_near`` 와 ``same_domain`` 이 동시에 있으면 혼동 방지 문구를 덧붙인다.

    Args:
        catalog: 활성 relation_kind 목록(``fetch_active_relation_kinds`` 결과).

    Returns:
        Markdown 블록 문자열. 카탈로그가 비면 **빈 문자열**(프롬프트에서 이 절이 통째로 빠진다).

    힌트 우선순위
        ``RELATION_KIND_HINTS_KO`` 에 정의된 5종 통제어휘는 미리 작성된 한국어 힌트를 쓴다.
        그 외 DB 에 직접 추가된 kind 는 description 첫 200자를 보조 설명으로 표시한다.
        따라서 새 통제어휘를 추가할 때 ``RELATION_KIND_HINTS_KO`` 에도 등록하면 프롬프트 품질이 올라간다.
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


def _fmt_topic(topic: Mapping[str, Any] | None) -> str:
    """(topic_ko / subtopic_ko) 를 사람이 읽는 한 줄 표기로 — 미부여/부분값도 견고하게.

    관계 LLM 에 주제를 **참고 신호**로 보여줄 문자열이다.

    Args:
        topic: ``{"topic_ko", "subtopic_ko"}`` dict. ``None`` 이나 빈 dict 도 받는다.

    Returns:
        ``대주제 / 세부주제``, 한쪽만 있으면 그 값, 둘 다 없으면 ``(주제 없음)``.
    """
    if not topic:
        return "(주제 없음)"
    topic_ko = str(topic.get("topic_ko") or "").strip()
    subtopic_ko = str(topic.get("subtopic_ko") or "").strip()
    if topic_ko and subtopic_ko:
        return f"{topic_ko} / {subtopic_ko}"
    return topic_ko or subtopic_ko or "(주제 없음)"


# 주제를 **참고 신호로만** 쓰게 하는 지시. "주제가 다르면 무조건 무관"으로 굳으면
# 레시피↔주방도구처럼 정상적인 교차 주제 관계가 전부 죽는다.
# 판단 우선순위는 항상 **내용**이며, 주제는 same_domain 오매칭을 줄이는 참고 신호일 뿐이다.
_TOPIC_SOFT_GUIDE_KO = """### 자기주제(topic) 참고 가이드 (soft·보조)
소스와 각 후보에는 **자기주제(topic_ko / subtopic_ko)** 가 함께 제공된다(자산이 자기 내용에서 확정한 주제).
- 소스와 후보의 **주제(topic)가 다르면** 같은 도메인(``same_domain``)으로 보지 마라.
- **단** 주제가 달라도 후보 요약·내용이 **실제로 관련**되면, 판단은 **내용을 우선**한다(주제는 **참고 신호**일 뿐, 하드 배제 금지).
- 정상적인 크로스-주제 관계(예: 레시피 ↔ 주방도구)는 내용이 합치하면 그대로 연결한다.
"""


def build_relation_proposal_prompt(
    *,
    source_summary: str,
    source_media_type: str,
    candidates: Sequence[Mapping[str, Any]],
    relation_kinds_catalog: Sequence[Mapping[str, Any]],
    source_topic: Mapping[str, Any] | None = None,
) -> str:
    """관계 제안 프롬프트 전체를 하나의 문자열로 조립한다.

    순수 함수(DB·LLM 호출 없음). 후보 경로는 **파일명만** 싣는다 — 디렉터리 전체 경로는 LLM 입력에
    노출하지 않는다(결정성·개인정보 누출 방지).

    Args:
        source_summary: 소스 자산 요약(길이 상한은 본문에서 잘라 씀).
        source_media_type: 매체 타입 구분용 문자열.
        candidates: ``find_embedding_candidates`` 결과 행들(066: topic_ko/subtopic_ko 동반 가능).
        relation_kinds_catalog: **active** ``relation_kind`` 목록(``type_code``/``type_name``/``description``/``is_symmetric``).
        source_topic: 소스 자산의 자기주제 ``{"topic_ko","subtopic_ko"}``. 주제가 달라도 내용이
            맞으면 연결하라는 **soft 신호**로만 쓰인다(하드 배제 금지). ``None`` 이면 주제 표기를
            통째로 생략한다 — 주제 미부여 자산·구 호출부 경로.

    Returns:
        LLM에 그대로 넘길 단일 프롬프트 문자열.
    """
    cand_lines: list[str] = []
    for c in candidates:
        # FR-008(SC-006): 디렉터리 풀경로를 LLM 입력으로 노출하지 않는다(결정성·PHI 누출 방지,
        # 헌법 3조·10조). 후보 식별엔 파일명만 충분하므로 basename 만 ``filename`` 으로 내보낸다.
        filename = posixpath.basename(str(c["file_uri"] or ""))
        emb_score = round(c["emb_score"], 6)
        # C-3: emb_score=0.0 인 후보는 경로 신호(파일명·폴더 매칭)로 추가된 것 — LLM 이 0.0 을
        # "비유사"로 오해하지 않게 ``signal`` 표식을 붙인다(가이드 문구와 호응).
        signal = "경로 신호" if emb_score == 0.0 else "임베딩"
        # 066 FR-202: 후보의 자기주제(topic_ko/subtopic_ko)를 후보 표기에 실어 주제 정합을 보게 한다.
        cand_topic_ko = c.get("topic_ko")
        cand_subtopic_ko = c.get("subtopic_ko")
        cand_lines.append(
            json.dumps(
                {
                    "target_media_item_id": c["id"],
                    "filename": filename,
                    "media_type": c["media_type"],
                    "summary": (c["summary"] or "")[:500],
                    "embedding_similarity": emb_score,
                    "signal": signal,
                    "topic_ko": cand_topic_ko,
                    "subtopic_ko": cand_subtopic_ko,
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

    # 출력 예시(JSON 샘플)에 쓸 relation_type_code 를 실제 카탈로그에서 고른다.
    # same_domain → duplicate_near → 알파벳 첫 코드 → 하드코드 폴백 순으로 선택한다.
    # 이렇게 하면 카탈로그가 바뀌어도 예시가 항상 유효한 코드를 가리킨다.
    example_codes = {str(r.get("type_code", "")) for r in relation_kinds_catalog} if relation_kinds_catalog else set()
    if "same_domain" in example_codes:
        example_relation_type = "same_domain"
    elif "duplicate_near" in example_codes:
        example_relation_type = "duplicate_near"
    else:
        example_relation_type = (
            next(iter(sorted(example_codes)), "same_domain") if example_codes else "same_domain"
        )

    # topic 지시부에 주입할 닫힌 27+미분류 목록(taxonomy_seed.json 단일 출처·결정적).
    topic_taxonomy_block = _build_topic_taxonomy_block()

    # 066 FR-202: 소스 자기주제를 한 줄로 표기(soft 신호). None(미부여·구 호출부)이면 (주제 없음).
    source_topic_line = _fmt_topic(source_topic)

    return f"""너는 멀티모달 미디어 간 관계를 표현하는 JSON만 출력하는 도우미다.

규칙:
- 반드시 JSON 객체 하나만 출력한다. 코드 블록·설명 문장 금지.
- 각 엣지의 ``relation_type_code`` 는 **관계 종류**(왜 연결되는지: same_domain, duplicate_near 등). 카탈로그의 ``type_code`` 와 **완전히 동일**해야 한다.
{catalog_rules}
- 기본은 카탈로그 코드 사용. 다만 목록에 정말 맞는 관계 종류가 없을 때만, 소문자 ``[a-z][a-z0-9_]*`` 새 코드를 제안할 수 있다. 서버는 이를 ``relation_kind`` 에 비활성으로 기록하고 검토 큐에 넣는다.
- **토픽(주제 대분류):** ``topic_ko`` 는 아래 **범주 목록에서 정확히 하나**를 골라 **그대로** 적는다(자유 기입·신조어 금지). 확신이 없거나 어느 범주에도 맞지 않으면 ``미분류`` 를 고른다(억지로 배정하지 말 것). ``topic_en`` 은 고른 범주의 짝 영어 코드를 그대로 쓴다.

범주 목록(topic_ko · topic_en — 이 목록 밖 값은 쓰지 말 것):
{topic_taxonomy_block}

- **서브토픽(세부 주제, 자유):** ``subtopic_ko`` 는 고른 범주 **밑의 구체 주제어**를 자유롭게 적는다(한국어는 **한 어절**·공백 금지 권장, 예: 범주 ``음식·요리`` 밑 ``김밥`` / ``라면``). ``subtopic_en`` 은 같은 뜻의 짧은 영어(한 토큰·소문자 권장). 맥락이 있으면 **비우지 말고** 채우는 것을 권장한다.
  - **문서·파일·상품·인물 등 고유명**은 subtopic 에 넣지 말고 ``reason`` 등에 적는다.
- ``reason``: 연결 근거 한 줄(한국어 권장). 고유명·세부 맥락은 여기에 둬도 된다.

소스 요약: {source_summary[:1200]}
소스 매체 타입: {source_media_type}
소스 주제(topic): {source_topic_line}

후보 목록(embedding_similarity 는 1에 가까울수록 유사. ``signal`` 이 ``경로 신호`` 면 파일명·폴더로 추가된 후보. ``topic_ko`` / ``subtopic_ko`` 는 후보의 자기주제):
{candidates_block}

{_TOPIC_SOFT_GUIDE_KO}

{_PATH_SIGNAL_GUIDE_KO}

출력 형식 예:
{{"edges":[
  {{"target_media_item_id":123,"relation_type_code":"{example_relation_type}","confidence":0.75,
    "topic_ko":"음식·요리","subtopic_ko":"김밥","topic_en":"food_cooking","subtopic_en":"gimbap",
    "reason":"유사 후보·주제 일치"
  }}
]}}
"""
