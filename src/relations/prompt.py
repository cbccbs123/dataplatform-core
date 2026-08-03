"""LLM **관계 제안**용 프롬프트 문자열 생성.

구성
    1. **소스 요약·타입** — LLM이 소스 맥락을 알도록.
    2. **임베딩 후보 JSON** — ``target_media_item_id``, 요약, ``embedding_similarity`` (1에 가까울수록 유사).
    3. **활성 relation_kind 카탈로그** — ``relation_kinds_catalog`` (DB ``fetch_active_relation_kinds`` 결과).
       ``type_code`` 는 ``relation_kind.kind_code``. **관계 종류**(왜 연결되는지)만 여기서 고른다.
    4. **선택 가이드** — ``RELATION_KIND_HINTS_KO`` + 카탈로그에 없는 코드는 DB ``description`` 일부 표시.
    5. **닫힌 topic 분류체계 목록** — ``topic_ko`` 는 ``taxonomy_seed.json`` 의 **27+미분류**에서 하나 선택
       세부주제는 열린 층이라 자유 기입한다(구체 주제어).

출력 규격
    LLM 은 **JSON 객체 하나**만 반환하도록 지시한다(코드 펜스 금지).

대주제 지시 — 닫힌 목록에서 고르게 한다
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
# ⚠️ duplicate_near 힌트에 "임베딩 유사도로 가져온 후보처럼" 류의 표현을 쓰지 말 것 —
#    후보는 정의상 전부 임베딩 유사도로 온 것이라 "모든 후보=duplicate_near" 순환 지시가 된다
#    (shadow A/B 로 검증된 오분류 원인 · docs/관계_품질_측정_20260728.md §4).
RELATION_KIND_HINTS_KO: dict[str, str] = {
    "same_domain": "대상은 다르지만 같은 분야로 묶일 때",
    "same_series": "같은 시리즈·연작·브랜드 라인업 등 **연속·묶음**일 때",
    "duplicate_near": "**같은 구체적 대상**을 거의 같은 형식으로 담은 사실상 중복본일 때",
    "references": "명시적 인용·링크·제목 참조 등 **참조** 관계일 때",
    "derived_from": "한쪽이 다른쪽에서 **파생·생성**된 관계일 때",
}

# ``duplicate_near`` 와 ``same_domain`` 이 함께 활성일 때만 덧붙는 혼동 방지 문구.
# 인라인 문자열이 아니라 모듈 상수인 이유: shadow A/B 측정이 **이 문구만** 갈아끼워 비교할 수 있어야
# 한다(조립을 복제하면 "운영 프롬프트 vs 재구현" 비교가 돼 실험이 무효가 된다).
# ⚠️ 문구를 고치면 운영 관계 생성 출력이 바뀐다 — 변형은 측정 스크립트의 override 로만 주고,
#    검증된 뒤에야 여기를 바꾼다.
RELATION_ANTI_DUP_HINT_KO = (
    "\n\n**구분:** 주제·세부주제가 같아도 **다루는 대상이 다르면** "
    "``duplicate_near`` 가 아니다. 대상이 다르고 분야만 같으면 ``same_domain`` 이다."
)

# ``confidence`` 채점 기준 — **"선택한 종류의 정의 안에서, 이 관계가 얼마나 뚜렷한가"**
# (종류 조건부 척도 · 2026-07-31 채택 · shadow A/B 3회 측정 끝의 v3).
#
# 왜 이 축인가 — 세 번의 실측이 좁혀 온 결론이다:
#   v1 "근거 정보가 충분했나"(자기평가) → 93% 가 최고값으로 붕괴. LLM 은 자기 판단의 정보
#      부족을 인정하지 않는다.
#   v2 "두 요약이 같은 대상인가"(쌍 단위) → same_domain 에서는 완벽 단조였으나
#      duplicate_near 에서 97% 붕괴 — 그 질문이 dup 을 **고른 이유 그 자체**라 동어반복.
#   v3 종류마다 척도를 달리해 선택과 분리 → dup 값별 실제 strong 비율 46→72→84%(단조) ·
#      전체 이름표 정확도 79.6→83.5% · 채택. (측정 기록: scripts/measure_relation_quality.py
#      PROMPT_VARIANTS 주석 · specs/081 §2026-07-31)
#
# 설계 장치: ①판단 절차 4단계 — "망설이면 낮은 값" 하향 규칙은 v1 실측 상향 편향의 상쇄
# ②예시/기준 분리 + "예시일 뿐" 전역 가드 — same_series "라인업" 오독 사고의 재발 방지
# ③dup 0.9 두-조건 성립제 ④경계 강등 문구.
# ⚠️ 문구를 고치면 운영 관계 생성 출력·점수 분포가 함께 바뀐다 — 변형은 측정 스크립트의
#    confidence_guide_override 로만 실험하고, 검증된 뒤에야 여기를 바꾼다.
RELATION_CONFIDENCE_GUIDE_KO = (
    "\n- ``confidence``: 선택한 ``relation_type_code`` 의 관계가 이 쌍에서 **얼마나 뚜렷하게 "
    "성립하는지**다. 자기 확신이 아니라 요약에서 관찰한 사실로 판단한다. "
    "네 값(0.9/0.7/0.5/0.3)만 쓴다.\n"
    "  판단 절차: 1) 두 요약에서 연결 근거를 찾아 ``reason`` 에 먼저 쓴다. "
    "2) 선택한 종류의 기준표에서 그 근거가 만족하는 값을 고른다. "
    "3) 두 값 사이에서 망설여지면 **낮은 값**을 쓴다. "
    "4) 요약·키워드·파일명 밖의 배경지식은 근거로 치지 않는다.\n"
    "  기준은 각 줄의 문장이다. \"예:\"는 이해를 돕는 예시일 뿐이다 — 예시와 다른 소재라도 "
    "기준 문장에 맞으면 그 값을 쓴다.\n"
    "  [duplicate_near 를 골랐을 때 — 내용 겹침의 정도]\n"
    "  - ``0.9`` 같은 대상 **그리고** 같은 측면, 두 조건을 모두 만족할 때만 — 핵심 내용까지 "
    "사실상 같다. 하나라도 아니면 0.7 이하로 내린다. 예: 같은 폭포를 소개하는 문서와 영상.\n"
    "  - ``0.7`` 같은 대상이지만 다루는 측면이 다르다. "
    "예: 같은 궁궐을 다룬 역사 문서와 관광 안내 영상.\n"
    "  - ``0.5`` 같은 대상이 등장하지만 한쪽에서는 중심 소재가 아니다. "
    "경계: 그 대상이 양쪽 모두의 중심이면 0.7, 한쪽에서 스쳐 가면 0.5. "
    "예: 김치 문서 ↔ 한식 전반을 다루며 김치를 한 단락만 언급하는 영상.\n"
    "  - ``0.3`` 같은 대상이라는 근거가 요약 안에 없다 — 파일명 등 간접 단서뿐이다.\n"
    "  [same_domain 을 골랐을 때 — 분야 공유의 좁기]\n"
    "  - ``0.9`` 같은 세부 활동·목적까지 공유한다. 예: 둘 다 김장 방법을 가르친다.\n"
    "  - ``0.7`` 같은 세부 분야를 다룬다. "
    "예: 김치 담그기 ↔ 된장 만들기 (발효식품 요리라는 세부 분야).\n"
    "  - ``0.5`` 같은 대분야 안에서 소재만 다르다. "
    "예: 김치 레시피 ↔ 커피 내리는 법 (음식·요리라는 틀만 공유).\n"
    "  - ``0.3`` 분야 명칭 외에 공통점이 없다.\n"
    "  [references·derived_from·same_series 를 골랐을 때 — 근거의 명시성]\n"
    "  - ``0.9`` 관계의 직접 증거가 요약이나 파일명에 있다. "
    "예: 제목을 그대로 인용 / \"~을 요약한 문서\" 문구 / 같은 어간 + 1부·2부 순번.\n"
    "  - ``0.7`` 직접 증거는 없으나 내용상 강하게 시사된다.\n"
    "  - ``0.5`` 정황뿐이다.\n"
    "  - ``0.3`` 추측에 가깝다."
)


def _build_relation_kind_guide(
    catalog: Sequence[Mapping[str, Any]],
    *,
    kind_hints_override: Mapping[str, str] | None = None,
    anti_dup_override: str | None = None,
) -> str:
    """카탈로그의 ``type_code`` 마다 한 줄 힌트를 붙인 선택 가이드 블록을 만든다.

    ``duplicate_near`` 와 ``same_domain`` 이 동시에 있으면 혼동 방지 문구를 덧붙인다.

    Args:
        catalog: 활성 relation_kind 목록(``fetch_active_relation_kinds`` 결과).
        kind_hints_override: kind 별 힌트를 **부분 교체**한다(주지 않은 kind 는 원래 힌트 유지).
            ``None``(기본)이면 운영 힌트를 그대로 쓴다. **측정 전용 seam** — A/B 에서 프롬프트
            조립을 복제하면 "운영 프롬프트 vs 재구현" 비교가 돼 실험이 무효가 되므로, 조립은
            한 곳에 두고 문구만 주입한다.
        anti_dup_override: ``duplicate_near``/``same_domain`` 구분 문구를 통째로 교체한다.
            ``None``(기본)이면 ``RELATION_ANTI_DUP_HINT_KO``.

    Returns:
        Markdown 블록 문자열. 카탈로그가 비면 **빈 문자열**(프롬프트에서 이 절이 통째로 빠진다).

    힌트 우선순위
        ``RELATION_KIND_HINTS_KO`` 에 정의된 5종 통제어휘는 미리 작성된 한국어 힌트를 쓴다.
        그 외 DB 에 직접 추가된 kind 는 description 첫 200자를 보조 설명으로 표시한다.
        따라서 새 통제어휘를 추가할 때 ``RELATION_KIND_HINTS_KO`` 에도 등록하면 프롬프트 품질이 올라간다.
    """
    # 힌트 출처를 지역 변수로 뽑는다 — override 가 오면 그 kind 만 덮어쓴 사본을 쓴다
    # (기본 ``None`` 이면 운영 dict 그대로라 출력이 바이트 단위로 동일하다).
    hints = (RELATION_KIND_HINTS_KO if kind_hints_override is None
             else {**RELATION_KIND_HINTS_KO, **kind_hints_override})
    codes = sorted({str(r.get("type_code", "")).strip() for r in catalog if str(r.get("type_code", "")).strip()})
    if not codes:
        return ""
    lines: list[str] = []
    for code in codes:
        hint = hints.get(code)
        if hint:
            lines.append(f"- ``{code}``: {hint}")
        else:
            desc = str(next((r.get("description") for r in catalog if str(r.get("type_code")) == code), "") or "")
            lines.append(f"- ``{code}``: (DB 설명) {desc[:200]}")
    body = "\n".join(lines)
    # ⚠️ **명시 주입은 조건을 이긴다.** `anti_dup_override` 를 준 것은 "이 문구를 붙여라"는
    # 의사 표현이므로 카탈로그 구성과 무관하게 적용한다. 조건에 종속시키면 **종류를 뺀 프롬프트에
    # 억제 문구를 살리는 실험이 불가능**해진다(081 Y팔 측정에서 실제로 막혔다 — 2026-07-30).
    # 운영은 override 를 주지 않으므로 아래 elif 만 타고, 출력은 바이트 불변이다.
    anti_dup = ""
    if anti_dup_override is not None:
        anti_dup = anti_dup_override
    elif "duplicate_near" in codes and "same_domain" in codes:
        anti_dup = RELATION_ANTI_DUP_HINT_KO
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
    source_keywords: str | None = None,
    source_filename: str | None = None,
    kind_hints_override: Mapping[str, str] | None = None,
    anti_dup_override: str | None = None,
    confidence_guide_override: str | None = None,
) -> str:
    """관계 제안 프롬프트 전체를 하나의 문자열로 조립한다.

    순수 함수(DB·LLM 호출 없음). 후보 경로는 **파일명만** 싣는다 — 디렉터리 전체 경로는 LLM 입력에
    노출하지 않는다(결정성·개인정보 누출 방지).

    Args:
        source_summary: 소스 자산 요약(길이 상한은 본문에서 잘라 씀).
        source_media_type: 매체 타입 구분용 문자열.
        candidates: 후보 행들. 각 후보의 주제(``topic_ko``/``subtopic_ko``)도 함께 실린다.
        relation_kinds_catalog: **active** ``relation_kind`` 목록(``type_code``/``type_name``/``description``/``is_symmetric``).
        source_topic: 소스 자산의 자기주제 ``{"topic_ko","subtopic_ko"}``. 주제가 달라도 내용이
            맞으면 연결하라는 **soft 신호**로만 쓰인다(하드 배제 금지). ``None`` 이면 주제 표기를
            통째로 생략한다 — 주제 미부여 자산·구 호출부 경로.
        kind_hints_override: kind 별 선택 힌트를 **부분 교체**한다(주지 않은 kind 는 원래 힌트
            유지). ``None``(기본)이면 운영 힌트를 그대로 쓴다. **측정 전용 seam** — A/B 에서
            프롬프트 조립을 복제하면 "운영 프롬프트 vs 재구현" 비교가 돼 실험이 무효가 되므로,
            조립은 한 곳에 두고 문구만 주입한다.
        anti_dup_override: ``duplicate_near``/``same_domain`` 구분 문구를 통째로 교체한다.
            ``None``(기본)이면 ``RELATION_ANTI_DUP_HINT_KO``.
        source_keywords: 소스 자산 키워드(원시 JSON 문자열 · 2026-07-31 채택 — 운영 호출부
            ``asset_entry`` 가 공급). ``None`` 이면 소스 키워드 줄을 통째로 생략한다(키워드 이전
            호출부·측정 대조군과의 하위호환). 후보 ``keywords`` 와 짝 — 한쪽만 주면 비대칭이다.
        source_filename: 소스 자산 **파일명만**(디렉터리 경로 제외 — 후보 ``filename`` 과 같은 규칙).
            ``None`` 이면 줄을 생략한다. **왜 필요한가**: 후보는 ``filename`` 을 받는데 소스는
            받지 않아 **양쪽 파일명 비교가 원리상 불가능**했다. ``same_series``("같은 어간 +
            순번/버전")·``references``("제목·파일명이 상대를 가리킴")·``derived_from``
            (``report``→``report_summary``)은 그 비교를 전제하는 정의라, 재료 없이 판정을
            요구하던 셈이다(2026-08-03 발견 · `docs/관계_재생성_테스트결과_20260731.md`).
            ⚠️ 전체 경로를 넣지 말 것 — 개인정보 누출·환경 의존(헌법 3조·10조).
        confidence_guide_override: ``confidence`` 채점 기준 문구를 교체한다.
            ``None``(기본)이면 운영 채택본 ``RELATION_CONFIDENCE_GUIDE_KO``(2026-07-31 v3).
            ``""`` 를 주면 기준 문구를 **제거**한다 — 채택 이전 프롬프트의 재현(측정 대조군)용.
            문자열을 주면 통째로 교체(shadow A/B). 주입 문구는 **줄바꿈으로 시작**해야 한다
            (직전 줄 끝에 이어 붙는다).

    Returns:
        LLM에 그대로 넘길 단일 프롬프트 문자열.
    """
    cand_lines: list[str] = []
    for c in candidates:
        # 디렉터리 전체 경로를 LLM 입력에 넣지 않는다(개인정보 누출 방지·환경 의존 제거,
        # 헌법 3조·10조). 후보 식별엔 파일명만 충분하므로 basename 만 ``filename`` 으로 내보낸다.
        filename = posixpath.basename(str(c["file_uri"] or ""))
        emb_score = round(c["emb_score"], 6)
        # C-3: emb_score=0.0 인 후보는 경로 신호(파일명·폴더 매칭)로 추가된 것 — LLM 이 0.0 을
        # "비유사"로 오해하지 않게 ``signal`` 표식을 붙인다(가이드 문구와 호응).
        signal = "경로 신호" if emb_score == 0.0 else "임베딩"
        # 후보의 주제를 함께 보여줘 주제 정합을 참고하게 한다(하드 조건은 아니다).
        cand_topic_ko = c.get("topic_ko")
        cand_subtopic_ko = c.get("subtopic_ko")
        cand_obj: dict[str, Any] = {
            "target_media_item_id": c["id"],
            "filename": filename,
            "media_type": c["media_type"],
            "summary": (c["summary"] or "")[:500],
        }
        # 후보 dict 에 ``keywords`` 가 실려 온 경우에만 싣는다(2026-07-31 채택 — 운영 후보 경로
        # `asset_candidates` 가 공급). 요약이 짧은 자산(전체의 1/3 이 80자 이하)에서 판단 재료를
        # 보강한다 — A/B 실측: 요약 80자 이하 쌍 정확도 76.3→83.4%. 키가 없으면 필드 생략
        # (keywords 이전 호출부·측정 대조군과의 하위호환).
        # 길이 상한 150자는 판정 프롬프트(`judge_relations.build_judge_prompt`)와 같은 값.
        if c.get("keywords"):
            cand_obj["keywords"] = str(c["keywords"])[:150]
        cand_obj.update(
            {
                "embedding_similarity": emb_score,
                "signal": signal,
                "topic_ko": cand_topic_ko,
                "subtopic_ko": cand_subtopic_ko,
            }
        )
        cand_lines.append(json.dumps(cand_obj, ensure_ascii=False))
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
        selection_guide = _build_relation_kind_guide(
            relation_kinds_catalog,
            kind_hints_override=kind_hints_override,
            anti_dup_override=anti_dup_override)
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

    # 소스 주제를 한 줄로 표기한다. 미부여면 '(주제 없음)'.
    source_topic_line = _fmt_topic(source_topic)

    # confidence 채점 기준 — 기본은 운영 채택본(v3 · 2026-07-31). None/""/문자열 구분에 주의:
    # "" 는 falsy 지만 "기준 제거"라는 명시적 의사이므로 `or` 로 합치면 안 된다(대조군 재현 불가).
    confidence_guide_block = (
        confidence_guide_override
        if confidence_guide_override is not None
        else RELATION_CONFIDENCE_GUIDE_KO
    )

    # 소스 키워드 줄(측정 전용 주입) — None 이면 줄 자체를 생략해 기본 출력 바이트 불변.
    source_keywords_line = (
        f"\n소스 키워드: {str(source_keywords)[:150]}" if source_keywords else ""
    )
    # 소스 파일명 줄 — 후보 ``filename`` 과 짝을 맞춘다(파일명 기반 종류 판정의 전제).
    # basename 은 호출부 책임이다(여기서 다시 자르면 이미 파일명만 온 값에 무해하지만,
    # 전체 경로가 흘러들어오는 것을 조용히 덮어 실수를 숨기게 된다).
    source_filename_line = (
        f"\n소스 파일명: {str(source_filename)[:120]}" if source_filename else ""
    )

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
- ``reason``: 연결 근거 한 줄(한국어 권장). 고유명·세부 맥락은 여기에 둬도 된다.{confidence_guide_block}

소스 요약: {source_summary[:1200]}
소스 매체 타입: {source_media_type}
소스 주제(topic): {source_topic_line}{source_keywords_line}{source_filename_line}

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
