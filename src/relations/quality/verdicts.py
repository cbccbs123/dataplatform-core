"""맹검 판정 라벨의 모델과 순수 JSON 변환 — **측정 행위의 기록**이다.

**흐름에서의 위치**: 판정 러너(``scripts/judge_relations.py``)가 채우고
``tests/fixtures/relations/verdicts/<measure_id>.json`` 으로 남는다. 재집계·재현은 이 파일만
읽는다(LLM 재호출 0).

⚠️ **버전관리하지 않는다 — 로컬 보관이다**(사용자 결정 2026-08-05 · ``.gitignore``).
라벨이 11파일 1.6MB·43,693줄로 커져 PR 리뷰를 덮었고, 측정의 **결론·수치는 보고서**
(``docs/관계_*.md``)에 남으므로 원천 라벨을 레포에 둘 이유가 없어졌다. 대신 **로컬 파일이 유일본**
이므로 지우면 소급 재채점이 불가능하다 — 프롬프트가 이미 바뀌어 LLM 재호출로도 같은 라벨이
복원되지 않는다. 측정을 이어갈 계획이면 별도 보관하라.

**설계 판단 — 왜 골든 경로가 아닌가**: 골든(``tests/golden/relations/``)은 사람이 검증한 *정답*이고
실 코퍼스 ``asset_id`` 종속이다(``specs/051-relation-golden-coverage/spec.md`` C4). 판정은 성격이
다르다 — 코퍼스가 아니라 *측정*의 기록이다. 경로를 갈라 두는 것이 **"판정을 골든으로 자동 승격하지
않는다"는 규율을 물리적으로 강제**한다. 둘 다 버전관리 대상은 아니지만 이 구분은 유효하다.

**깨지면 안 되는 것**
- 판정 레코드에 **요약 본문·파일 경로를 넣지 않는다**(개인정보 노출면). 테스트가 필드 집합을 봉인한다.
- ``sample_edge_ids`` 를 반드시 남긴다 — DB 가 자라면 같은 시드로도 표본이 달라지므로,
  재현할 때는 다시 뽑지 않고 이 목록을 그대로 쓴다.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# ``error`` 는 LLM 호출 실패를 뜻한다 — 판정값이 아니라 **측정 실패 표식**이라서
# 비율 계산의 분모에서 빠진다(error 를 weak 로 세면 품질이 좋아 보이는 사고가 난다).
VALID_VERDICTS = frozenset({"strong", "weak", "none", "error"})

_WHY_MAX = 20   # 사유 길이 상한 — 개인정보가 사유 문장에 실려 나가는 것을 막는다.


@dataclass(frozen=True)
class Verdict:
    """엣지 한 건의 맹검 판정.

    Args:
        edge_id: 판정 대상 엣지.
        verdict: ``VALID_VERDICTS`` 중 하나.
        why: 판정 사유(저장 시 20자로 잘린다).
        prompt_sha256: **실제로 보낸** 프롬프트의 SHA-256. 자산 내용이 바뀌면 값이 달라지므로
            "그때와 같은 조건인가"를 사후에 판별할 수 있다.
        judged_by_human: 사람 교차판정 결과. ``""`` 는 미판정.
    """
    edge_id: str
    verdict: str
    why: str
    prompt_sha256: str
    judged_by_human: str = ""


@dataclass(frozen=True)
class VerdictSet:
    """한 번의 측정(measure) 전체 — 메타 + 표본 + 판정들.

    Args:
        measure_id: 측정 식별자(``YYYYMMDD-<축>``). 파일명과 같다.
        method: 이 측정이 무엇을 반증하려 했는지 사람이 읽는 한 문단.
        rubric_version: ``rubric.RUBRIC_VERSION``.
        rubric_text: 프롬프트에 실제로 들어간 루브릭 원문(복사본).
        judge_model: 판정에 쓴 모델 ID.
        seed: 표본 추출 시드.
        strata: 층화 축(``kind``·``conf``·``cohort``·``kind-conf``).
        created_at: 생성 시각(ISO8601).
        sample_edge_ids: 표본 엣지 전체. 재현 시 재추출 대신 이걸 쓴다.
        verdicts: 판정 결과들.
    """
    measure_id: str
    method: str
    rubric_version: str
    rubric_text: str
    judge_model: str
    seed: int
    strata: str
    created_at: str
    sample_edge_ids: tuple[str, ...]
    verdicts: tuple[Verdict, ...]


def dump_verdicts(vs: VerdictSet) -> dict:
    """JSON 직렬화 가능한 dict 로 변환한다(순수 함수).

    Args:
        vs: 저장할 측정 묶음.

    Returns:
        ``{version, …메타, sample_edge_ids, verdicts}``. 판정 레코드는 화이트리스트 5필드뿐이다.
    """
    return {
        "version": 1,
        "measure_id": vs.measure_id,
        "method": vs.method,
        "rubric_version": vs.rubric_version,
        "rubric_text": vs.rubric_text,
        "judge_model": vs.judge_model,
        "seed": vs.seed,
        "strata": vs.strata,
        "created_at": vs.created_at,
        "sample_edge_ids": list(vs.sample_edge_ids),
        "verdicts": [
            {"edge_id": v.edge_id, "verdict": v.verdict, "why": v.why[:_WHY_MAX],
             "prompt_sha256": v.prompt_sha256, "judged_by_human": v.judged_by_human}
            for v in vs.verdicts
        ],
    }


def load_verdicts(d: dict) -> VerdictSet:
    """dict 를 검증해 ``VerdictSet`` 으로 복원한다(순수 함수).

    Args:
        d: ``dump_verdicts`` 가 만든 dict.

    Returns:
        복원된 묶음.

    Raises:
        ValueError: ``version`` 이 1이 아니거나 판정값이 ``VALID_VERDICTS`` 밖일 때.
    """
    if d.get("version") != 1:
        raise ValueError(f"verdicts version must be 1: {d.get('version')!r}")
    verdicts = []
    for r in d.get("verdicts", []):
        v = str(r.get("verdict", ""))
        if v not in VALID_VERDICTS:
            raise ValueError(f"알 수 없는 판정값: {v!r} (edge_id={r.get('edge_id')!r})")
        verdicts.append(Verdict(str(r["edge_id"]), v, str(r.get("why", "")),
                                str(r.get("prompt_sha256", "")),
                                str(r.get("judged_by_human", ""))))
    return VerdictSet(
        measure_id=str(d.get("measure_id", "")), method=str(d.get("method", "")),
        rubric_version=str(d.get("rubric_version", "")),
        rubric_text=str(d.get("rubric_text", "")),
        judge_model=str(d.get("judge_model", "")), seed=int(d.get("seed", 0)),
        strata=str(d.get("strata", "")), created_at=str(d.get("created_at", "")),
        sample_edge_ids=tuple(str(x) for x in d.get("sample_edge_ids", [])),
        verdicts=tuple(verdicts))


def verdict_counts(vs: VerdictSet, *, human: bool = False) -> dict[str, int]:
    """판정값별 건수.

    Args:
        vs: 집계 대상.
        human: 참이면 사람 판정(``judged_by_human``)을 센다. 미판정(``""``)은 제외한다.

    Returns:
        ``{판정값: 건수}``. 0건인 값은 키가 없다.
    """
    src = (v.judged_by_human for v in vs.verdicts) if human else (v.verdict for v in vs.verdicts)
    return dict(Counter(x for x in src if x))


def error_rate(vs: VerdictSet) -> float:
    """LLM 호출 실패 비율.

    5%를 넘으면 그 측정을 무효로 본다 — 실제로 전량 실패한 결과가 ``strong 0%`` 로 조용히
    집계돼 정상처럼 보인 적이 있다(spec 엣지케이스 1).

    Args:
        vs: 집계 대상.

    Returns:
        ``error`` 건수 / 전체. 판정이 0건이면 0.0.
    """
    if not vs.verdicts:
        return 0.0
    return sum(1 for v in vs.verdicts if v.verdict == "error") / len(vs.verdicts)
