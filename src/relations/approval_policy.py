"""관계의 **자동승인 자격·영속화 여부·노출 등급·검토 대상**을 정하는 정책 — 순수 함수.

**왜 한 모듈에 모으는가**: 같은 임계가 세 곳에서 쓰인다 — ① 영속화(`graph_persist`: 행을
만들 것인가) ② 노출(`graph_query`: 화면에 보일 것인가) ③ 검토 큐(`review`: 사람이 볼 것인가).
각자 상수를 들고 있으면 어긋나고, 어긋남이 **조용한 유령 구간**을 만든다 — 예컨대 영속화 하한이
0.75인데 노출 하한이 0.8이면 그 사이 구간은 *"DB에 행은 있는데 화면에는 없는"* 관계가 되어
"왜 관계가 안 보이나"를 아무도 추적하지 못한다. 그래서 판정을 여기 모으고, 영속화 판정과 노출
판정이 **항상 일치**함을 테스트로 봉인한다(`tests/test_approval_policy.py`).

**두 계열로 나누는 근거**(관계 종류 5종은 성격이 둘로 갈린다):

- ``SIMILARITY_KINDS``(유사도 추론 계열) — 근거가 임베딩 유사도라 **저신뢰 꼬리의 값이 낮다.**
  특히 ``same_domain`` 은 정의 자체가 *"대상이 다르고 분야만 같다"* 라서 **설계상 weak** 다.
- ``EXPLICIT_KINDS``(명시적 근거 계열) — 인용·파생·연작처럼 근거가 텍스트·파일명에 드러나
  있어 저신뢰여도 정보량이 있다. 게다가 경로 신호(``path_signal``)가 되살아나면 저신뢰
  명시적 제안이 정당하게 늘어난다. 그래서 **폐기 게이트에서 면제**한다.

**임계는 원리로 정하고 라벨로 검정만 했다**(헌법 1조 · 학습 배제). 기본 0.75 의 원리는
*"설계상 weak 인 종류와 그 이웃의 저신뢰 꼬리"* 이고, 라벨로는 그 구간의 유효 관계 밀도가
낮음을 **확인**했을 뿐이다 — 라벨에서 임계를 역산하거나 격자 탐색으로 고르지 않았다.

**모르는 kind 는 보수적으로 유지한다.** 통제어휘는 ``promote_relation_kind`` 로 늘어날 수
있는데, 분류표에 없는 종류를 조용히 버리면 새 종류가 통째로 사라진다.
"""
from __future__ import annotations

# 유사도 추론 계열 — 근거가 임베딩 유사도. 저신뢰 꼬리를 폐기·비노출 대상으로 본다.
SIMILARITY_KINDS: frozenset[str] = frozenset({"duplicate_near", "same_domain"})

# 명시적 근거 계열 — 인용·파생·연작. 저신뢰여도 폐기하지 않는다(위 모듈 docstring 참조).
EXPLICIT_KINDS: frozenset[str] = frozenset({"references", "derived_from", "same_series"})

# 노출·검토에서 "끝난 것"으로 보는 상태 — 사람이 내린 결정을 되살리지 않는다.
_TERMINAL_STATUSES: frozenset[str] = frozenset({"rejected", "expired"})


def parse_kind_set(raw: str | None, *, default: frozenset[str]) -> frozenset[str]:
    """설정의 쉼표 구분 문자열을 kind 집합으로 바꾼다.

    설정 계층은 원시 문자열만 보관하고 도메인 타입 변환은 여기서 한다 — 설정이 관계 어휘를
    몰라도 되게 하려는 것이다.

    ⚠️ **빈 문자열은 기본값으로 되돌리지 않는다.** ``""`` 을 주는 것은 *"이 게이트를 끈다"* 는
    명시적 의사인데, 기본값으로 대체하면 끌 방법이 없어진다(롤백 불가).

    Args:
        raw: 설정에서 읽은 원시 값. ``None`` 은 "설정하지 않음"(기본값 사용),
            ``""``·공백은 "빈 집합"(게이트 끔)으로 구분해 다룬다.
        default: ``raw`` 가 ``None`` 일 때 쓸 기본 집합.

    Returns:
        소문자·공백 제거된 kind 집합. 빈 항목은 버린다.
    """
    if raw is None:
        return default
    return frozenset(tok.strip().lower() for tok in raw.split(",") if tok.strip())


def should_persist(
    kind_code: str,
    conf: float | None,
    *,
    min_conf_similarity: float,
) -> bool:
    """이 제안을 ``graph_edge`` 행으로 **만들 것인가**.

    유사도 추론 계열의 저신뢰 꼬리만 버린다 — 검토 큐를 채우기만 하고 유효 관계 밀도가 낮은
    구간이다. 숨기는 것이 아니라 **행을 만들지 않는다**: 만들어 두고 조회 계층마다 숨기면
    필터 로직이 산재하고 쓰레기가 영구히 남는다.

    Args:
        kind_code: 관계 종류 코드(대소문자 무관).
        conf: LLM 신뢰도 0~1. ``None`` 은 판정 불가 — 유사도 계열에서는 미달로 본다.
        min_conf_similarity: 유사도 계열 신뢰도 하한. **``0`` 이하면 게이트를 끈다**
            (전부 영속화 — 기존 동작).

    Returns:
        ``True`` 면 행을 만든다.
    """
    if min_conf_similarity <= 0.0:
        return True
    if kind_code.strip().lower() not in SIMILARITY_KINDS:
        return True   # 명시적 계열·미지의 신규 kind 는 보수적으로 유지
    return conf is not None and conf >= min_conf_similarity


def is_auto_approvable(kind_code: str, *, exclude_kinds: frozenset[str]) -> bool:
    """이 종류가 **자동승인 자격**이 있는가(신뢰도 관문과는 별개).

    ``same_domain`` 을 제외하는 근거: 정의상 "대상이 다르고 분야만 같다"라서 신뢰도가 높아도
    강한 관계가 아니다. 실측으로 자동승인분의 정밀도가 58.8→74.8% 로 오른다
    (`docs/관계_품질_측정_20260728.md` §6.1).

    Args:
        kind_code: 관계 종류 코드(대소문자 무관).
        exclude_kinds: 자동승인에서 제외할 종류 집합. 비면 전부 자격 있음(기존 동작).

    Returns:
        ``True`` 면 신뢰도 관문으로 넘어간다. ``False`` 면 신뢰도와 무관하게 ``proposed``.
    """
    return kind_code.strip().lower() not in exclude_kinds


def is_review_exempt(kind_code: str, *, exempt_kinds: frozenset[str]) -> bool:
    """이 종류를 **사람 검토 큐에서 뺄 것인가**.

    ``same_domain`` 을 면제하는 근거: 자동승인도 안 되고(`is_auto_approvable`) 종착지가 약칸
    노출이라 **승격이라는 개념 자체가 없다.** 검토해도 할 일이 없는 것을 큐에 쌓으면 밀도가
    떨어져 볼 만한 것까지 묻힌다.

    ⚠️ 삭제가 아니라 **필터**다 — 면제를 해제하면 즉시 되돌아온다. 그리고 면제된 관계도 약칸에
    계속 노출되므로 사용자에게서 사라지지 않는다(2단 노출이 필수 전제인 이유).

    Args:
        kind_code: 관계 종류 코드(대소문자 무관).
        exempt_kinds: 검토 큐에서 면제할 종류 집합. 비면 전부 검토 대상(기존 동작).

    Returns:
        ``True`` 면 검토 큐에서 뺀다.
    """
    return kind_code.strip().lower() in exempt_kinds


def exposure_tier(
    status: str,
    kind_code: str,
    conf: float | None,
    *,
    min_conf_similarity: float,
) -> str | None:
    """화면 노출 등급 — ``"strong"``(연관 자료) · ``"weak"``(비슷한 주제) · ``None``(노출 안 함).

    2단으로 나누는 이유: 자동승인 게이트가 `same_domain` 을 강등하면 관계 보유 자산이 26%
    줄어 화면이 빈다. 강등된 관계를 **약칸으로 살려** 커버리지를 지키면서 강칸의 정밀도만
    올린다. 두 오류의 비용이 비대칭이라는 점이 근거다 — 강→약 오강등은 *여전히 보이지만*,
    약한 관계가 강칸에 올라오면 **가릴 방법이 없다**.

    ``active`` 는 신뢰도와 무관하게 항상 강칸이다 — 사람이 승인했거나 자동승인을 통과한 것을
    노출 하한으로 되돌려 숨기면 그 결정을 무시하는 것이 된다.

    Args:
        status: 엣지 상태(``active``·``proposed``·``rejected``·``expired``).
        kind_code: 관계 종류 코드(대소문자 무관).
        conf: LLM 신뢰도 0~1.
        min_conf_similarity: 약칸 노출 하한. **``should_persist`` 와 같은 값을 쓴다** —
            다르면 "만들지만 안 보이는" 유령 구간이 생긴다(테스트가 이 일치를 봉인한다).

    Returns:
        등급 문자열, 또는 노출하지 않을 때 ``None``.
    """
    st = status.strip().lower()
    if st == "active":
        return "strong"
    if st in _TERMINAL_STATUSES:
        return None
    if st != "proposed":
        return None   # 미지의 상태는 노출하지 않는다(닫힌 어휘가 늘어나면 여기서 판단)
    if should_persist(kind_code, conf, min_conf_similarity=min_conf_similarity):
        return "weak"
    return None
