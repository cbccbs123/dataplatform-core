"""LLM이 반환한 **엣지 JSON** 정규화·검증.

필드 모델
    - ``relation_type_code``: 관계 **종류**(``relation_kind.kind_code`` 와 동일한 snake_case). 프롬프트 허용 목록과 대조.
    - ``topic_ko`` / ``topic_en`` / ``subtopic_*``: 엣지 ``graph_edge.topic`` jsonb 로 들어갈 값.
      주제 라벨은 **한글 첫 어절·영어 최대 2토큰(3토큰 이상이면 첫 토큰만)** 으로 짧게 잘라 서로
      다른 표기가 무한히 늘어나는 것을 막는다(하드코드 동의어 맵은 쓰지 않는다).

이 모듈은 전부 **순수 함수**다 — DB·네트워크·시각에 의존하지 않으므로 DB 없이 단위 테스트한다.

신규 코드
    ``sanitize_llm_proposed_type_code``: 카탈로그에 없는 제안을 DB에 **inactive** 로 넣기 전 형식·레거시 검사.

호출 순서 (파이프라인 내 위치)
    1. ``parse_llm_edges`` — LLM 응답 루트 파싱
    2. ``normalize_relation_type_code`` — 코드 소문자·trim
    3. ``sanitize_llm_proposed_type_code`` — 형식·레거시 필터(카탈로그 밖 코드만 통과 필요)
    4. ``coerce_topic_fields_mvp`` — topic 폴백 보정 후 graph_edge.topic jsonb 에 적재
"""

from __future__ import annotations

import re
from typing import Any

# 과거 MVP: 도메인을 kind_code 로 쓰던 값 — 프롬프트·sanitize 에서 제외.
LEGACY_DOMAIN_TYPE_CODES: frozenset[str] = frozenset({"medical", "computer", "marriage", "general"})

# LLM이 제안한 신규 relation kind 코드(비활성 자동 등록용): 소문자 영문·숫자·밑줄, 최대 100자
_LLM_PROPOSED_TYPE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,99}$")

TOPIC_KO_MAX = 200
SUBTOPIC_KO_MAX = 200
TOPIC_EN_MAX = 200
SUBTOPIC_EN_MAX = 200
_TOPIC_DEFAULT_KO = "일반"
_TOPIC_DEFAULT_EN = "general"

_MULTI_SPACE_RE = re.compile(r"\s+")
_EDGE_PUNCT_STRIP = re.compile(r"^[\s.,;:!?·、，。]+|[\s.,;:!?·、，。]+$")


def _collapse_whitespace(s: str) -> str:
    """연속 공백(탭·줄바꿈 포함)을 공백 하나로 합치고 앞뒤를 자른다.

    Args:
        s: 원본 문자열.

    Returns:
        공백이 정리된 문자열.
    """
    return _MULTI_SPACE_RE.sub(" ", s).strip()


def _strip_edge_punctuation(s: str) -> str:
    """문자열 **앞뒤**의 잡음 공백·구두점만 제거한다(가운데 내용은 건드리지 않는다).

    Args:
        s: 원본 문자열.

    Returns:
        앞뒤가 정리된 문자열. 전부 구두점이면 빈 문자열.
    """
    # 정규식 한 번으로 제거되지 않는 중첩 패턴(예: " , " + " . ")에 대응하기 위해
    # 변경이 없을 때까지 반복한다. 일반 입력에서는 1~2회로 수렴한다.
    prev = None
    while prev != s:
        prev = s
        s = _EDGE_PUNCT_STRIP.sub("", s)
    return s


def parse_llm_edges(data: dict[str, Any]) -> list[dict[str, Any]]:
    """LLM 응답 루트에서 엣지 ``dict`` 리스트를 꺼낸다.

    허용 형태
        - ``{"edges": [ {...}, ... ]}``
        - ``{"items": [ ... ]}`` (별칭)
        - 단일 엣지 객체(``target_media_item_id`` 키가 있으면 한 원소 리스트로 감쌈)

    LLM 이 ``edges`` 키를 빠뜨리고 객체를 직접 반환하는 경우를 단일 엣지 폴백으로 처리한다.

    Args:
        data: LLM 응답 루트 dict.

    Returns:
        엣지 dict 리스트. dict 가 아닌 원소(문자열 등)는 **조용히 버린다**. 알아볼 수 있는 형태가
        없으면 빈 리스트 — "응답이 망가졌다"는 판정은 호출부(``propose_edges_json``)가 한다.
    """
    raw = data.get("edges")
    if raw is None and isinstance(data.get("items"), list):
        raw = data["items"]
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    if isinstance(data.get("target_media_item_id"), int | str):
        return [data]
    return []


def normalize_relation_type_code(raw: Any) -> str | None:
    """``relation_type_code`` 를 소문자로 맞추고 앞뒤 공백을 자른다.

    Args:
        raw: LLM이 준 값. 어떤 타입이든 문자열로 변환해 처리한다.

    Returns:
        정규화된 코드. 값이 없거나 공백뿐이면 ``None``(호출자가 그 엣지를 건너뛴다).
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    return s or None


def sanitize_llm_proposed_type_code(code: str) -> str | None:
    """카탈로그에 없는 코드를 **inactive 로 자동 등록하기 전** 통과시킬지 검사한다.

    거부 조건
        - 길이 0 또는 100 초과
        - ``LEGACY_DOMAIN_TYPE_CODES``(도메인을 관계 종류로 쓰던 과거 코드 — 섞이면 LLM이 도메인과
          관계를 혼동해 제안한다)
        - 정규식 ``[a-z][a-z0-9_]{0,99}`` 불일치

    Args:
        code: 검사할 코드. **호출 전에** ``normalize_relation_type_code`` 로 소문자화해 두어야
            한다(이 함수는 대문자를 정규식 불일치로 거부한다).

    Returns:
        통과하면 입력 그대로, 거부되면 ``None``.
    """
    if not code or len(code) > 100:
        return None
    if code in LEGACY_DOMAIN_TYPE_CODES:
        return None
    if not _LLM_PROPOSED_TYPE_CODE_RE.fullmatch(code):
        return None
    return code


def normalize_topic_ko(raw: Any) -> str:
    """한글 주제 라벨을 정규화한다 — 앞뒤 공백 제거 → 여러 어절이면 **첫 어절만** → 길이 상한.

    문장형 라벨("여행 사진 모음")이 그대로 저장되면 주제 값이 무한히 늘어나 패싯이 망가지므로
    첫 어절로 줄인다.

    Args:
        raw: LLM이 준 값(문자열이 아니어도 된다).

    Returns:
        정규화된 라벨. 값이 없으면 빈 문자열(폴백은 ``coerce_topic_fields_mvp`` 소관).
    """
    s = str(raw or "").strip()
    if not s:
        return s
    if " " in s:
        s = s.split()[0].strip()
    if len(s) > TOPIC_KO_MAX:
        return s[:TOPIC_KO_MAX].rstrip()
    return s


def normalize_subtopic_ko(raw: Any) -> str:
    """한글 세부주제를 정규화한다 — 공백 정리 · 앞뒤 구두점 제거 · **첫 어절만** · 길이 상한.

    대주제와 같은 단어 수 규칙을 쓴다(일관된 카디널리티).

    Args:
        raw: LLM이 준 값.

    Returns:
        정규화된 세부주제. 값이 없으면 빈 문자열.
    """
    s = _collapse_whitespace(str(raw or ""))
    s = _strip_edge_punctuation(s)
    if not s:
        return ""
    parts = s.split()
    s = parts[0].strip()
    if len(s) > SUBTOPIC_KO_MAX:
        return s[:SUBTOPIC_KO_MAX].rstrip()
    return s


def normalize_topic_en(raw: Any) -> str:
    """영어 주제 라벨을 정규화한다 — 소문자 · **3토큰 이상이면 첫 토큰만** · 길이 상한.

    2토큰까지는 공백을 살린다("machine learning" 처럼 두 단어가 한 개념인 경우가 있어서다).

    Args:
        raw: LLM이 준 값.

    Returns:
        정규화된 영문 라벨. 값이 없으면 빈 문자열.
    """
    s = str(raw or "").strip()
    if not s:
        return s
    s = s.lower()
    tokens = s.split()
    if len(tokens) > 2:
        s = tokens[0]
    else:
        s = " ".join(tokens)
    if len(s) > TOPIC_EN_MAX:
        return s[:TOPIC_EN_MAX].rstrip()
    return s


def normalize_subtopic_en(raw: Any) -> str:
    """영어 세부주제를 정규화한다 — 소문자 · 공백 정리 · 앞뒤 구두점 제거 · **첫 토큰만** · 길이 상한.

    Args:
        raw: LLM이 준 값.

    Returns:
        정규화된 영문 세부주제. 값이 없으면 빈 문자열.
    """
    s = _collapse_whitespace(str(raw or "")).lower()
    s = _strip_edge_punctuation(s)
    if not s:
        return ""
    tokens = s.split()
    s = tokens[0]
    if len(s) > SUBTOPIC_EN_MAX:
        return s[:SUBTOPIC_EN_MAX].rstrip()
    return s


def extract_topic_fields_from_edge(edge: dict[str, Any]) -> tuple[str, str, str, str]:
    """엣지 dict 에서 주제 4필드를 꺼내 정규화한다.

    LLM이 키 이름을 흔들기 때문에 별칭을 함께 본다 — ``topic`` → ``topic_ko``,
    ``subtopic``/``sub_topic_ko`` → ``subtopic_ko``, ``sub_topic_en`` → ``subtopic_en``.

    Args:
        edge: LLM 엣지 dict 하나.

    Returns:
        ``(topic_ko, subtopic_ko, topic_en, subtopic_en)``. 없는 값은 빈 문자열이다
        (기본값 채우기는 ``coerce_topic_fields_mvp`` 가 한다).
    """
    tk = edge.get("topic_ko")
    if tk is None or not str(tk).strip():
        tk = edge.get("topic")
    sk = edge.get("subtopic_ko")
    if sk is None or not str(sk).strip():
        sk = edge.get("sub_topic_ko") or edge.get("subtopic")
    ten = edge.get("topic_en")
    sen = edge.get("subtopic_en")
    if sen is None or not str(sen).strip():
        sen = edge.get("sub_topic_en")
    return (
        normalize_topic_ko(tk),
        normalize_subtopic_ko(sk),
        normalize_topic_en(ten),
        normalize_subtopic_en(sen),
    )


def coerce_topic_fields_mvp(edge: dict[str, Any]) -> tuple[str, str, str, str, bool]:
    """``topic_ko`` 가 비어 있으면 폴백으로 채운다 — 주제가 없다고 엣지를 버리지 않기 위해서다.

    Args:
        edge: LLM 엣지 dict 하나.

    우선순위
        1. 엣지에 유효한 ``topic_ko`` 가 있으면 그대로(``topic_en`` 비면 ``general``).
        2. 없으면 ``reason`` 첫 줄을 잘라 토피로 사용.
        3. ``reason`` 도 없으면 ``일반`` / ``general``.

    Returns:
        ``(topic_ko, subtopic_ko, topic_en, subtopic_en, mvp_topic_auto)`` — 마지막 bool 은 자동 보정 여부.

    설계 의도
        ``topic_ko`` 가 비어 있어도 엣지를 버리지 않고 여기서 보정해 DB 에 저장한다.
        ``mvp_topic_auto=True`` 는 나중에 재검토·재정제가 필요한 행을 식별하는 마커.

    auto 변수 초기화 패턴
        ``auto = False`` 로 시작하고 1번 분기에서 조기 반환(``False`` 하드코드)하므로
        ``auto`` 가 True 로 바뀌는 경로는 2·3번 폴백 경로뿐이다.
        이 패턴은 분기 추가 시 ``auto`` 를 빠뜨려 잘못된 False 가 반환될 위험을 내포하므로 주의.
    """
    topic_ko, subtopic_ko, topic_en, subtopic_en = extract_topic_fields_from_edge(edge)
    auto = False
    if topic_ko.strip():
        # topic_ko 가 유효한 정상 경로 — auto 는 False 로 조기 반환
        if not topic_en.strip():
            topic_en = _TOPIC_DEFAULT_EN
        return topic_ko, subtopic_ko, topic_en, subtopic_en, False
    # 이 아래는 모두 폴백 경로: auto = True 로 반환
    reason = str(edge.get("reason") or "").strip()
    if reason:
        first = reason.split("\n")[0].strip()
        # 069 B5(P2-5): 원문 첫 줄(문장형·최대 200자)을 그대로 폴백하면 문장형 topic 이 저장돼
        # 패싯 오염·canonicalize LLM 폭주를 부른다. normalize_topic_ko 로 **첫 어절**만 취해
        # 문장형 저장을 차단한다(정합·카디널리티 개선 — 첫 어절이 조사/무의미어일 수 있어 주제
        # 품질 개선은 아니다). 비면 기본값으로 폴백.
        topic_ko = normalize_topic_ko(first) or _TOPIC_DEFAULT_KO
    else:
        topic_ko = _TOPIC_DEFAULT_KO
    if not topic_en.strip():
        topic_en = _TOPIC_DEFAULT_EN
    auto = True
    return topic_ko, subtopic_ko, topic_en, subtopic_en, auto


def type_label_from_kind_code(code: str, *, max_len: int = 120) -> str:
    """``kind_code``(snake_case)를 사람이 읽을 라벨로 바꾼다 — 밑줄을 공백으로.

    DB 시드에 ``kind_name_ko`` 가 있으면 그쪽이 우선이고, 없을 때만 이 폴백을 쓴다.

    Args:
        code: 관계 종류 코드(예: ``same_series``).
        max_len: 라벨 길이 상한. 넘으면 잘라낸다.

    Returns:
        읽기용 라벨(예: ``same series``). 코드가 비면 ``"기타"``.
    """
    s = (code or "").strip().lower()
    if not s:
        return "기타"
    label = " ".join(p for p in s.split("_") if p).strip() or "기타"
    if len(label) <= max_len:
        return label
    return label[:max_len].rstrip()


def description_ko_from_type_name_ko(type_name_ko: str) -> str:
    """관계 종류 이름을 ``… 도메인 연관`` 형태의 기본 설명문으로 만든다.

    신규 kind 를 등록할 때 description 이 비어 있으면 쓰는 폴백 문구다. 비한글 라벨에도 같은
    규칙을 적용한다.

    Args:
        type_name_ko: 관계 종류의 표시 이름.

    Returns:
        설명 문자열. 이름이 비면 ``"기타 도메인 연관"``.
    """
    raw = type_name_ko.replace("·", " ").split()
    parts = [p for p in raw if p]
    if not parts:
        return "기타 도메인 연관"
    if len(parts) == 1:
        return f"{parts[0]} 도메인 연관"
    return "·".join(parts) + " 도메인 연관"
