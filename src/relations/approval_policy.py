"""관계의 **자동승인 자격·영속화 여부·노출 등급·표시 이름·검토 대상**을 정하는 정책 — 순수 함수.

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

from collections.abc import Mapping, Sequence
from typing import Any

# 유사도 추론 계열 — 근거가 임베딩 유사도. 저신뢰 꼬리를 폐기·비노출 대상으로 본다.
SIMILARITY_KINDS: frozenset[str] = frozenset({"duplicate_near", "same_domain"})

# 명시적 근거 계열 — 인용·파생·연작. 저신뢰여도 폐기하지 않는다(위 모듈 docstring 참조).
EXPLICIT_KINDS: frozenset[str] = frozenset({"references", "derived_from", "same_series"})

# 노출·검토에서 "끝난 것"으로 보는 상태 — 사람이 내린 결정을 되살리지 않는다.
_TERMINAL_STATUSES: frozenset[str] = frozenset({"rejected", "expired"})

# 동시보유 접기 — 같은 자산 쌍에 이름표가 둘 이상 붙는 것을 조회에서 하나로 접는다.
#
# 왜 생기나: 저장 키가 `(src_node, dst_node, relation_kind_id)` 라 **종류가 다르면 다른 행**이다.
# 스키마의 정상 동작이지 결함이 아니다. 화면은 이웃 하나를 카드 하나로 그리므로, 접지 않으면
# 한 카드에 `["duplicate_near", "same_domain"]` 같은 모순된 이름표가 함께 붙는다.
#
# DB 는 건드리지 않는다(두 행 유지) — 조회에서만 고른다. 지우면 되돌릴 수 없고 판정 근거가
# 사라진다. 접기는 필터라 언제든 끌 수 있다.
#
# 조합으로 가르고 점수로 가르지 않는 이유: 판정 라벨 대조에서 `duplicate_near` 는 `same_domain`
# 과 **같이 붙었을 때** 점수가 높아도 맞는 비율이 크게 떨어졌다. 이름표가 둘 달렸다는 사실
# 자체가 판정이 흔들렸다는 신호라, 그 상황에서는 점수를 신호로 쓸 수 없다.
#
# 지우면 재발하는 것: 이 규칙을 "점수 우선 → 동점이면 duplicate_near" 로 되돌리면 **정확한
# 이름표를 버리고 절반 틀린 것을 남긴다**(원안이 그랬고 실측으로 기각했다). 테스트가 봉인한다.
#
# 관계 이슈 진단: "관계가 사라졌다"→접힌 것은 사라지지 않는다(`folded_kind_codes` 확인) ·
# "DB 와 화면 건수가 다르다"→차이가 곧 동시보유 쌍 수다 · "규칙을 바꾸겠다"→재측정 먼저.
# 설계 배경: docs/설계_변경이력.md(2026-08-07) · docs/관계_품질_측정_20260728.md

# 조합 → 남길 이름표. 조합 **전체가 키로 일치할 때만** 적용한다 — 부분 일치로 넓히면 세 종류가
# 붙은 미지의 경우까지 근거 없이 판정하게 된다. 표본이 있는 조합만 등재하고, 나머지는 점수·
# 결정적 순서 규칙으로 넘긴다.
FOLD_PREFERRED_KIND: dict[frozenset[str], str] = {
    frozenset({"duplicate_near", "same_domain"}): "same_domain",
}

# 화면 표시 이름 — **LLM 프롬프트용 이름과 일부러 다르다.** 실수가 아니다.
#
# 같은 `kind_code` 에 이름이 두 벌 붙는다. 역할이 다르기 때문이다:
#   프롬프트용 = DB `relation_kind.kind_name_ko` — LLM 이 읽는다. 카탈로그 블록에 그대로 실려
#                판정에 개입하므로, **바꾸면 관계 생성 결과가 바뀐다**(A/B 검증 없이 손대지 말 것).
#   표시용     = 아래 표 — 사람이 읽는다. 판정에 개입하지 않으므로 자유롭게 고칠 수 있다.
# 둘을 잇는 진짜 식별자는 `kind_code` 이고, 두 이름은 그 코드에 붙은 서로 다른 설명이다.
#
# 왜 나눴나: 화면의 `same_domain` 이 "동일 주제"로 나가고 있었다. 뜻이 틀렸고(주제가 아니라
# 분야만 같다) 자산 자기주제 기반 주제 탭·패싯과 이름까지 겹쳐 다른 두 기능이 같게 읽힌다.
# 고치려 보니 그 값이 프롬프트에 실려 있어, 위험 없는 쪽만 먼저 고쳤다.
#
# 지우면 재발하는 것: 두 이름이 다른 것을 보고 "맞춰야지" 하며 DB 값을 고치면 **프롬프트가
# 바뀐다.** 이 분리는 최종 설계가 아니라 안전하게 가는 경로이며, 관계 재생성 기회에 A/B 로
# 검증하고 합칠 수 있다. 그때까지 관리자 화면은 DB 이름을 그대로 보여준다(알고 두는 불일치).
# 설계 배경: docs/설계_변경이력.md(2026-08-07) · docs/과제_책무_KPI.md(재정합 백로그)
RELATION_KIND_DISPLAY_KO: dict[str, str] = {
    "same_domain": "같은 분야",      # ← "동일 주제"(프론트 하드코딩)를 바로잡은 것
    "duplicate_near": "유사 중복",
    "derived_from": "파생 자료",
    "references": "참조",            # ← 프론트에 매핑이 없어 "기타 연관"으로 뭉개지던 것
    "same_series": "같은 연작",      # ← 위와 같음
}


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
    """화면 노출 등급 — ``"strong"``(연관 자료) · ``"weak"``(참고 자료) · ``None``(노출 안 함).

    표시 문구에 **"주제"라는 말을 쓰지 않는다**(종전 "비슷한 주제"를 버린 이유). 자산
    자기주제 기반의 주제 탭·패싯이 이미 그 이름을 써서 근거가 다른 두 기능이 같게 읽힌다.
    두 칸의 실제 차이는 주제가 아니라 사람이 확인했는가다.

    2단으로 나누는 이유: 자동승인 게이트가 `same_domain` 을 강등하면 관계 보유 자산이 크게
    줄어 화면이 빈다. 강등분을 약칸으로 살려 커버리지를 지키면서 강칸의 정밀도만 올린다.
    두 오류의 비용이 비대칭이라는 것이 근거다 — 강→약 오강등은 여전히 보이지만, 약한 관계가
    강칸에 올라오면 가릴 방법이 없다.

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


def display_name_ko(kind_code: str, *, fallback: str | None = None) -> str:
    """관계 종류를 화면에 보여줄 이름으로 바꾼다(**LLM 프롬프트용 이름이 아니다**).

    모르는 코드를 *"기타 연관"* 같은 말로 덮지 않는다 — 통제어휘가 늘어나면 새 종류가
    화면에서 통째로 사라져 늘어난 사실 자체가 안 보인다. 위 표에 없으면 호출자가 준
    DB 이름을, 그것도 없으면 코드를 그대로 보인다.

    Args:
        kind_code: 관계 종류 코드(대소문자·공백 무관).
        fallback: 표에 없을 때 쓸 이름. 보통 호출자가 DB ``kind_name_ko`` 를 넘긴다.
            비면 ``kind_code`` 를 그대로 돌려준다.

    Returns:
        표시용 이름. 코드가 비면 ``fallback`` → 그것도 없으면 빈 문자열.
    """
    code = (kind_code or "").strip().lower()
    known = RELATION_KIND_DISPLAY_KO.get(code)
    if known:
        return known
    fb = (fallback or "").strip()
    return fb or code


def choose_folded_edge(edges: Sequence[Mapping[str, Any]]) -> tuple[int, list[str]]:
    """같은 이웃에 붙은 엣지 여럿 중 화면에 남길 **하나**를 고른다(순수 판정).

    판정 순서는 ``tier`` → 조합 규칙 → 신뢰도 → ``kind_code`` 사전순 → 입력 순서다.
    앞의 것이 이기면 뒤는 보지 않는다. 왜 이 순서인지는 위 "동시보유 접기" 주석 참조.

    Args:
        edges: 같은 이웃 자산에 붙은 엣지들. 각 항목에서 ``kind_code``·``confidence``·
            ``tier`` 를 읽는다. 비어 있으면 안 된다(호출자가 그룹을 만들어 넘긴다).

    Returns:
        ``(남길 엣지의 인덱스, 접힌 종류 코드 목록)``. 접힌 목록은 정렬돼 있고 남긴 종류는
        빠진다. 엣지가 하나뿐이면 ``(0, [])``.

    Raises:
        ValueError: ``edges`` 가 비었을 때. 조용히 넘기면 호출부의 그룹 구성 버그가 숨는다.
    """
    if not edges:
        raise ValueError("접을 엣지가 없다 — 호출부의 그룹 구성을 확인하라")
    if len(edges) == 1:
        return 0, []

    kinds = frozenset(str(e.get("kind_code") or "").strip().lower() for e in edges)
    preferred = FOLD_PREFERRED_KIND.get(kinds)

    def rank(item: tuple[int, Mapping[str, Any]]) -> tuple[int, int, float, str, int]:
        """정렬키 — 위 판정 순서 1~5를 그대로 튜플로 편다.

        Args:
            item: ``enumerate(edges)`` 가 주는 ``(입력 인덱스, 엣지)``. 인덱스는 최종
                tiebreak 로 쓰이므로 함께 받는다.

        Returns:
            오름차순 비교용 5튜플. **이기는 조건일수록 작은 값**이다.
        """
        idx, e = item
        kind = str(e.get("kind_code") or "").strip().lower()
        conf = e.get("confidence")
        # 오름차순 정렬로 첫 항목을 고르므로, 이기는 조건일수록 **작은 값**을 준다.
        return (
            # 사람이 승인한 것을 기계 규칙으로 버리면 그 결정을 무시하는 것이 된다.
            0 if str(e.get("tier") or "") == "strong" else 1,
            # 점수보다 앞선다 — 이름표가 둘 달린 상황에서는 점수가 신호가 아니다(위 주석).
            0 if preferred is not None and kind == preferred else 1,
            -(float(conf) if conf is not None else 0.0),
            # 아래 둘은 근거가 있어서가 아니라 같은 입력이면 같은 출력이어야 하기 때문이다.
            kind,
            idx,
        )

    keep_idx, keep = min(enumerate(edges), key=rank)
    keep_kind = str(keep.get("kind_code") or "").strip().lower()
    folded = sorted(k for k in kinds if k and k != keep_kind)
    return keep_idx, folded
