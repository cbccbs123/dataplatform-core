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

from collections.abc import Mapping, Sequence
from typing import Any

# 유사도 추론 계열 — 근거가 임베딩 유사도. 저신뢰 꼬리를 폐기·비노출 대상으로 본다.
SIMILARITY_KINDS: frozenset[str] = frozenset({"duplicate_near", "same_domain"})

# 명시적 근거 계열 — 인용·파생·연작. 저신뢰여도 폐기하지 않는다(위 모듈 docstring 참조).
EXPLICIT_KINDS: frozenset[str] = frozenset({"references", "derived_from", "same_series"})

# 노출·검토에서 "끝난 것"으로 보는 상태 — 사람이 내린 결정을 되살리지 않는다.
_TERMINAL_STATUSES: frozenset[str] = frozenset({"rejected", "expired"})

# ─────────────────────────────────────────────────────────────────────────────
# 🔴 동시보유 접기 (081 조각⑤ · 2026-08-07 채택) — 관계 이슈 조사 시 **먼저 볼 곳**
#
# 같은 자산 쌍에 이름표가 둘 이상 붙는다. 저장 키가 `(src_node, dst_node, relation_kind_id)`
# 라 **종류가 다르면 다른 행**이기 때문이고, 이는 결함이 아니라 스키마의 정상 동작이다.
# v4 실측 **180쌍**. 화면은 이웃 하나를 카드 하나로 그리므로 그대로 두면 한 카드에
# `["duplicate_near", "same_domain"]` 같은 **모순된 이름표 두 개**가 붙는다.
#
# DB 는 건드리지 않는다(두 행 유지) — **조회에서만** 하나를 고른다. 지우면 되돌릴 수 없고
# 판정 근거가 사라진다. 접기는 필터라 언제든 끌 수 있다.
#
# ── 어느 쪽을 남기나: 실측으로 정했다(추측 아님)
#
# 079 가 영속화한 판정 라벨을 v4 에 붙여 쟀다(180쌍 중 **167쌍**에 판정 존재 · LLM 재호출 0).
# 판정 루브릭이 이 질문을 그대로 묻고 있었다:
#     strong = 같은 **구체적 대상** → `duplicate_near` 의 주장이 맞다
#     weak   = 같은 **넓은 분야**일 뿐 → `same_domain` 의 주장이 맞다
#
#   조합                                 쌍    dup 맞음   same_domain 맞음
#   duplicate_near@0.7 + same_domain@0.9  84      25          52
#   duplicate_near@0.7 + same_domain@0.7  77      16          57
#   duplicate_near@0.9 + same_domain@0.9  15       3           9
#   duplicate_near@0.9 + same_domain@0.7   2       1           1
#   ────────────────────────────────────────────────────────────────
#   합계                                 178   45 (27.3%)  119 (72.1%)
#
# → **`same_domain` 을 남기면 72.1% 맞고 `duplicate_near` 를 남기면 27.3% 맞다(2.6배).**
#
# ⚠️ **스펙 원안을 실측으로 기각했다.** spec 081 은 *"점수 우선 → 동점이면 duplicate_near"*
#    였는데, 그 규칙은 동점 77쌍에서 정확한 이름표(same_domain 92.4%)를 버리고 절반 틀린 것
#    (dup 0.7 = 48.3%)을 남긴다. 원안 작성 시점엔 종류별 정확도가 아직 없었다.
#
# ⚠️ **`duplicate_near` 0.9 는 단독일 때만 믿을 수 있다.** 전체 정확도는 88.9% 인데
#    `same_domain` 과 **같이 붙은 15쌍에서는 3/13 = 23%** 다. 이름표가 둘 달렸다는 사실
#    자체가 "이 판정은 흔들렸다"는 신호다 — 그래서 점수가 아니라 조합으로 가른다.
#
# ── 🔎 관계 이슈가 생기면 여기를 의심하라
#
#   "관계가 화면에서 사라졌다"      → 접기가 삼켰을 수 있다. `folded_kind_codes` 를 보라
#                                     (접힌 관계는 **사라지지 않는다** — 이름표만 접힌다).
#   "DB 건수와 화면 건수가 다르다"  → 정상이다. 차이 = 동시보유 쌍 수(v4 기준 180).
#   "이름표가 기대와 다르다"        → 이 표를 보라. `same_domain` 이 이겼을 것이다.
#   "규칙을 바꿔야겠다"            → **재측정부터.** 아래 재측정 방법 참조.
#
# ── 재측정 방법 (LLM 재호출 0 · 079 SC-001 그대로)
#
#   문서 레포 `fixtures/relations/verdicts/*.json` 의 판정 중 키가 `<a>__<b>`(pair_key) 인
#   것만 v4 에 조인된다(edge_id 키는 재생성으로 소멸). 자산 쌍은 재생성돼도 같으므로 유효하다.
#   근거 문서: `docs/관계_품질_측정_20260728.md` · 설계이력 2026-08-07.
#
# ⚠️ 이 수치의 한계(079 §한계 그대로): 판정자가 LLM 이다. **절대 문턱으로 쓰지 말 것** —
#    "A 와 B 중 어느 쪽이 나은가"의 **상대 비교로만** 유효하다. 지금 쓰임이 정확히 그것이다.
# ─────────────────────────────────────────────────────────────────────────────

# 조합 → 남길 이름표. **조합 전체가 키로 일치할 때만** 적용한다(부분 일치 금지 — 세 종류가
# 붙은 미지의 경우까지 이 규칙으로 처리하면 근거 없는 판정이 된다).
# 표본이 있는 조합만 등재한다. 나머지(dup+references 1쌍·dup+same_series 1쌍)는 **표본 1이라
# 근거가 없어 넣지 않는다** — 점수·결정적 순서 규칙으로 넘긴다.
FOLD_PREFERRED_KIND: dict[frozenset[str], str] = {
    frozenset({"duplicate_near", "same_domain"}): "same_domain",
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

    표시 문구는 **"연관 자료"(확인됨) / "참고 자료"(확인 전)** 로 확정했다(사용자 결정 2026-08-07).
    ⚠️ 종전의 *"비슷한 주제"* 는 쓰지 않는다 — 자산 자기주제(`asset_topic`) 기반의 **주제 탭·주제
    패싯·같은주제 묶음**이 이미 "주제"라는 이름을 쓰고 있어, 근거가 다른 두 기능이 같은 것으로
    읽힌다. 두 칸의 실제 차이는 주제가 아니라 **사람이 확인했는가**다.

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


def choose_folded_edge(edges: Sequence[Mapping[str, Any]]) -> tuple[int, list[str]]:
    """같은 이웃에 붙은 엣지 여럿 중 **화면에 남길 하나**를 고른다(081 조각⑤).

    근거·실측·재측정 방법은 이 모듈 상단 **"동시보유 접기"** 주석 블록이 정본이다.
    관계 이슈 조사 시 그 표를 먼저 보라.

    **판정 순서**(앞선 규칙이 이기면 뒤는 보지 않는다):

    1. **``tier`` 우선**(``strong`` > ``weak``) — 사람이 승인한 것을 기계 규칙으로 버리면
       그 결정을 무시하는 것이 된다. ``exposure_tier`` 가 ``active`` 를 무조건 강칸에 두는
       것과 같은 이유다.
    2. **조합 규칙**(``FOLD_PREFERRED_KIND``) — 종류 집합이 표의 키와 **정확히 일치**하면
       그 이름표를 남긴다. 점수보다 앞선다: 실측상 ``duplicate_near`` 는 ``same_domain`` 과
       같이 붙었을 때 점수가 높아도(0.9) 23%만 맞았다 — **이름표가 둘 달렸다는 사실 자체가
       판정이 흔들렸다는 신호**라, 그 상황에서는 점수를 신호로 쓸 수 없다.
    3. **신뢰도 내림차순** — 조합 규칙이 없는 경우의 기본. spec 081 원안이기도 하다.
    4. **``kind_code`` 사전순 → 입력 순서** — 결정성을 위한 최종 tiebreak(헌법 3조).
       근거가 있어서가 아니라 **같은 입력이면 같은 출력**이어야 하기 때문이다.

    Args:
        edges: 같은 이웃 자산에 붙은 엣지들. 각 항목에서 ``kind_code``·``confidence``·
            ``tier`` 를 읽는다. **비어 있으면 안 된다**(호출자가 그룹을 만들어 넘긴다).

    Returns:
        ``(남길 엣지의 인덱스, 접힌 종류 코드 목록)``. 접힌 목록은 **정렬돼** 있고 남긴
        종류는 빠진다. 엣지가 하나뿐이면 ``(0, [])``.

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
            0 if str(e.get("tier") or "") == "strong" else 1,      # 1. 강칸 우선
            0 if preferred is not None and kind == preferred else 1,  # 2. 조합 규칙
            -(float(conf) if conf is not None else 0.0),           # 3. 신뢰도 내림차순
            kind,                                                  # 4. 사전순
            idx,                                                   # 5. 입력 순서
        )

    keep_idx, keep = min(enumerate(edges), key=rank)
    keep_kind = str(keep.get("kind_code") or "").strip().lower()
    folded = sorted(k for k in kinds if k and k != keep_kind)
    return keep_idx, folded
