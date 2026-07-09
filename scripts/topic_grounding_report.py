"""자기주제 근거율(grounding)·분포 리포트 — 스펙 근거 스크립트 정식화(spec 065·FR-602·SC-02/03).

무엇을 측정하나
    065 는 자산 주제를 관계 이웃 엣지 투영이 아니라 **자기 내용(summary/keywords)에서 확정한 정본**
    (``asset_topic``)으로 둔다. 이 리포트는 그 정본의 건강 지표를 **읽기전용**으로 관측한다:

    · 집계 A(분포) — ``topic_ko`` 별·``(topic_ko, subtopic_ko)`` 별 카운트 + 부여율(SC-03).
        미부여 사유 추정: ``no_text``(메타에 summary/keywords 없음) vs ``분류실패``(텍스트 있는데 행 없음).
    · 집계 B(근거율·SC-02) — 자산 주제(``topic_ko``·``subtopic_ko``)가 그 자산의 자기 텍스트에
        **문자열로 등장**하는 비율. 등장하면 grounded, 없으면 polluted(오염). 문자열 매칭은
        스펙과 동일한 **보수적 하한**(의미적 근거는 이보다 높다). SC-02 = 오염율 ≤ 5% 목표.
    · (참고) ``--compare-projection`` — 옛 이웃-엣지 투영 방식이었다면 같은 grounding 지표가
        얼마였을지 비교(``graph_edge.topic`` 재구성·다중 라벨 중 하나라도 근거 있으면 grounded).

읽기전용·결정성 (헌법 3조)
    ``asset_topic``/``asset_metadata``/``graph_edge`` 를 일절 변경하지 않는다(LLM 0·순수 집계).
    같은 DB 상태 → 같은 리포트. 집계는 순수 함수로 분리해 단위테스트로 덮고, DB 조회는 얇게 둔다.

실행 (백필 전/후로 실행해 오염율 변화를 비교 — FR-503 백필 실행 자체는 사람 게이트)
    conda activate AuroraFS
    python scripts/topic_grounding_report.py --env dev
    python scripts/topic_grounding_report.py --env dev --compare-projection
    python scripts/topic_grounding_report.py --env dev --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

_REPO_ROOT = Path(__file__).resolve().parents[1]
# 직접 실행(python scripts/...) 시 repo 루트를 경로에 올려 src 패키지 import 보장(다른 러너와 동형).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ────────────────────────────────────────────────────────────────────────────
# 1) 순수 집계 (실 DB/LLM 없이 단위테스트로 덮는다)
# ────────────────────────────────────────────────────────────────────────────
def topic_distribution(topic_rows: list[dict[str, Any]]) -> Counter:
    """``topic_ko`` 별 자산 수 분포(순수·빈 topic_ko 제외)."""
    c: Counter = Counter()
    for r in topic_rows:
        tk = str(r.get("topic_ko") or "").strip()
        if tk:
            c[tk] += 1
    return c


def pair_distribution(topic_rows: list[dict[str, Any]]) -> Counter:
    """``(topic_ko, subtopic_ko)`` 짝별 자산 수 분포(순수). subtopic 빈값은 None 으로 정규화."""
    c: Counter = Counter()
    for r in topic_rows:
        tk = str(r.get("topic_ko") or "").strip()
        if not tk:
            continue
        sub = r.get("subtopic_ko")
        sub = str(sub).strip() if sub and str(sub).strip() else None
        c[(tk, sub)] += 1
    return c


def _label_in_text(label: Any, text: str) -> bool:
    """라벨(topic_ko/subtopic_ko)이 자기 텍스트에 문자열로 등장하는지(순수·보수적 하한).

    한국어는 대소문자가 없어 그대로 부분문자열 매칭한다. 빈/None 라벨은 매칭 대상 아님.
    """
    if not label:
        return False
    s = str(label).strip()
    return bool(s) and s in text


def _asset_grounded(labels: list, self_text: str) -> bool:
    """자산 라벨 중 하나라도 자기 텍스트에 근거가 있으면 grounded(순수).

    ``labels`` 는 ``[(topic_ko, subtopic_ko), ...]``. 정본은 라벨 1개, 투영 비교는 여러 개다.
    각 라벨은 subtopic(더 구체) 또는 topic 이 텍스트에 등장하면 근거 있음으로 본다.
    """
    text = self_text or ""
    for topic_ko, subtopic_ko in labels:
        if _label_in_text(subtopic_ko, text) or _label_in_text(topic_ko, text):
            return True
    return False


def build_grounding_report(asset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """근거율/오염율 리포트(순수·집계 B·SC-02).

    Args:
        asset_rows: ``[{asset_id, labels:[(topic_ko, subtopic_ko)], self_text}]``.

    Returns:
        ``{n_assets, n_grounded, n_polluted, grounding_rate, pollution_rate,
        polluted_samples}``. ``pollution_rate`` = 오염 자산 / 주제 보유 자산(SC-02 측정치).
    """
    total = len(asset_rows)
    grounded = 0
    polluted_samples: list[dict[str, Any]] = []
    for r in asset_rows:
        if _asset_grounded(r.get("labels") or [], r.get("self_text") or ""):
            grounded += 1
        elif len(polluted_samples) < 20:  # 검수용 오염 샘플(결정적 앞 20)
            polluted_samples.append(
                {"asset_id": r.get("asset_id"), "labels": [list(x) for x in (r.get("labels") or [])]}
            )
    polluted = total - grounded
    return {
        "n_assets": total,
        "n_grounded": grounded,
        "n_polluted": polluted,
        "grounding_rate": round(grounded / total, 4) if total else 0.0,
        "pollution_rate": round(polluted / total, 4) if total else 0.0,
        "polluted_samples": polluted_samples,
    }


def build_distribution_report(
    *, topic_rows: list[dict[str, Any]], text_asset_ids: set[str], n_registered: int
) -> dict[str, Any]:
    """분포·부여율·미부여 사유 리포트(순수·집계 A·SC-03).

    Args:
        topic_rows: ``asset_topic`` 행 ``[{asset_id, topic_ko, subtopic_ko}]``.
        text_asset_ids: registered 중 자기 텍스트(summary/keywords) 보유 자산 id 집합.
        n_registered: registered 자산 총수.

    부여율(``assignment_rate``)은 **텍스트 보유 자산 대비**(SC-03: 텍스트 보유의 95%↑ 목표).
    미부여 사유: ``no_text`` = registered 인데 텍스트 없음, ``분류실패`` = 텍스트 있는데 주제 행 없음.
    """
    topic_ids = {str(r["asset_id"]) for r in topic_rows}
    n_text = len(text_asset_ids)
    assigned_with_text = len(topic_ids & text_asset_ids)
    return {
        "n_registered": n_registered,
        "n_with_text": n_text,
        "n_with_topic": len(topic_ids),
        "n_no_text": max(n_registered - n_text, 0),
        "n_classify_failed": len(text_asset_ids - topic_ids),
        "assignment_rate": round(assigned_with_text / n_text, 4) if n_text else 0.0,
        "topic_distribution": dict(topic_distribution(topic_rows).most_common()),
        "pair_distribution": {
            (f"{t}>{s}" if s else t): c
            for (t, s), c in pair_distribution(topic_rows).most_common()
        },
    }


def group_label_rows(flat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """평탄 행 → 자산별 라벨 묶음(순수·정본/투영 공용).

    입력 ``[{asset_id, topic_ko, subtopic_ko, self_text}]``(자산당 여러 행 가능·투영) →
    ``[{asset_id, labels:[(topic_ko, subtopic_ko)], self_text}]``. 자기 텍스트는 자산당 동일
    가정(첫 등장 보존). 등장 순서를 보존해 결정적(헌법 3조).
    """
    by_asset: dict[str, dict[str, Any]] = {}
    for r in flat_rows:
        aid = str(r["asset_id"])
        entry = by_asset.get(aid)
        if entry is None:
            entry = {"asset_id": aid, "labels": [], "self_text": r.get("self_text") or ""}
            by_asset[aid] = entry
        tk = r.get("topic_ko")
        if tk:
            entry["labels"].append((tk, r.get("subtopic_ko")))
    return list(by_asset.values())


# ────────────────────────────────────────────────────────────────────────────
# 1b) 분포 가드 지표 (068 FR-301·T401 — 순수 계산 · 집단 통계 불변식)
# ────────────────────────────────────────────────────────────────────────────
# 회귀는 **개별 자산 정답표(수동 학습化)를 늘리는 방식이 아니라 집단 통계 불변식**으로 잡는다
# (spec 068 Non-Goals·헌법 학습 0). 가드는 계통 붕괴(미부여 급증·소분류 과병합·"기타" 과다)만
# 감지하고, 개별 1건의 애매(경계 topic 등)는 수용한다.
#
# 임계는 상수로 두되 **dev 실측(068 G6 재백필 리포트) 후 확정**한다 — 아래 값은 spec 진단치
# (여행>관광지 64%·음식>음식 53%·내용있는 미부여 13%)를 근거로 한 초기치이며 캘리브레이션 대상.
# 레벨: "hard"(게이트 실패·계통 붕괴) / "warn"(주의·게이트는 통과).
_UNASSIGNED_RATE_HARD = 0.12  # 미부여 자산/registered 상한(무내용 정상 미부여 여유 포함·dev 실측 후 확정)
_SUBTOPIC_MAX_SHARE_WARN = 0.5   # topic별 최대 subtopic 점유율 경보(dev 실측 후 확정)
_SUBTOPIC_MAX_SHARE_HARD = 0.7   # topic별 최대 subtopic 점유율 하드(과병합·dev 실측 후 확정)
_MISC_SUBTOPIC_RATE_WARN = 0.35  # subtopic None/"기타" 비율 경보(시드 커버 공백·dev 실측 후 확정)
_MISC_SUBTOPIC_RATE_HARD = 0.60  # subtopic None/"기타" 비율 하드(dev 실측 후 확정)
# 점유율 지표를 적용할 topic 최소 자산 수 — 표본 과소 topic 의 점유율은 노이즈라 게이트 제외.
_MIN_TOPIC_ASSETS_FOR_SHARE = 10
# "기타"류 subtopic 라벨(닫힌 시드 커버 공백의 탈출구). None 도 미세분류 없음으로 함께 계산.
_MISC_SUBTOPIC_LABELS = {"기타", "기타·미분류", "미분류"}


def _norm_subtopic(sub: Any) -> str | None:
    """subtopic 정규화(빈/공백 → None). 순수."""
    if sub is None:
        return None
    s = str(sub).strip()
    return s or None


def _is_real_subtopic(sub: Any) -> bool:
    """실 subtopic 여부 — None/"기타"류가 아닌 구체 소분류만 True(순수)."""
    s = _norm_subtopic(sub)
    return s is not None and s not in _MISC_SUBTOPIC_LABELS


def subtopic_concentration(topic_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """topic별 소분류 집중도(순수·과병합 감지용).

    각 topic 에 대해 **실 subtopic**(None/"기타" 제외)의 최대 점유율을 topic 전체 자산 대비로 계산.
    여행>관광지 64% 같은 과코스닝(한 subtopic 이 topic 을 흡수)을 지표화한다.

    Returns:
        ``{topic_ko: {n_assets, dominant_subtopic, max_share, n_subtopics}}``.
        max_share = 최대 실 subtopic 자산수 / topic 전체 자산수(실 subtopic 없으면 0.0·dominant None).
    """
    by_topic: dict[str, dict[str, Any]] = {}
    for r in topic_rows:
        tk = str(r.get("topic_ko") or "").strip()
        if not tk:
            continue
        entry = by_topic.setdefault(tk, {"n_assets": 0, "real": Counter()})
        entry["n_assets"] += 1
        if _is_real_subtopic(r.get("subtopic_ko")):
            entry["real"][_norm_subtopic(r.get("subtopic_ko"))] += 1
    out: dict[str, dict[str, Any]] = {}
    for tk, e in by_topic.items():
        real: Counter = e["real"]
        if real:
            dom, dom_n = real.most_common(1)[0]  # 동수는 첫 등장 우선(결정적)
            max_share = round(dom_n / e["n_assets"], 4)
        else:
            dom, max_share = None, 0.0
        out[tk] = {
            "n_assets": e["n_assets"],
            "dominant_subtopic": dom,
            "max_share": max_share,
            "n_subtopics": len(real),
        }
    return out


def topic_count_drift(before: dict[str, int], after: dict[str, int]) -> dict[str, Any]:
    """백필 전후 topic 건수 변동(순수·옵션 드리프트·FR-301 ④).

    Returns:
        ``{per_topic: {topic: {before, after, delta}}, total_churn, total_before, total_after}``.
        ``total_churn`` = Σ|delta|(전체 이동량). topic 정렬 결정적(sorted).
    """
    keys = sorted(set(before) | set(after))
    per_topic: dict[str, dict[str, int]] = {}
    churn = 0
    for k in keys:
        b = int(before.get(k, 0))
        a = int(after.get(k, 0))
        per_topic[k] = {"before": b, "after": a, "delta": a - b}
        churn += abs(a - b)
    return {
        "per_topic": per_topic,
        "total_churn": churn,
        "total_before": sum(int(v) for v in before.values()),
        "total_after": sum(int(v) for v in after.values()),
    }


def build_guard_report(
    *,
    topic_rows: list[dict[str, Any]],
    n_registered: int,
    before_topic_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """분포 가드 리포트(순수·집단 통계·FR-301·T401).

    입력은 **정본 asset_topic**(자산당 1행) 가정. 실 DB 집계와 분리(순수 함수) — DB 조회는 얇은 래퍼.

    Args:
        topic_rows: ``[{asset_id, topic_ko, subtopic_ko}]``(주제 부여 자산).
        n_registered: registered 자산 총수(미부여율 분모).
        before_topic_counts: (옵션) 백필 전 topic별 건수 — 있으면 드리프트 포함.

    Returns:
        ``{metrics, violations, level, [drift]}``. ``violations`` 항목은
        ``{metric, scope, value, threshold, level}``(+concentration 은 dominant_subtopic).
        ``level`` = "hard"/"warn"/"ok"(하드>경보>정상).
    """
    rows_with_topic = [r for r in topic_rows if str(r.get("topic_ko") or "").strip()]
    topic_ids = {str(r["asset_id"]) for r in rows_with_topic}
    n_with_topic = len(topic_ids)
    n_unassigned = max(n_registered - n_with_topic, 0)
    unassigned_rate = round(n_unassigned / n_registered, 4) if n_registered else 0.0

    # "기타"/None subtopic 비율(주제 보유 자산 대비) — 닫힌 시드 커버 공백 신호.
    n_misc = sum(1 for r in rows_with_topic if not _is_real_subtopic(r.get("subtopic_ko")))
    misc_rate = round(n_misc / len(rows_with_topic), 4) if rows_with_topic else 0.0

    # 실 (topic, subtopic) pair 싱글턴 비율 — 소분류 파편화 참고 지표.
    real_pairs: Counter = Counter()
    for r in rows_with_topic:
        if _is_real_subtopic(r.get("subtopic_ko")):
            real_pairs[(str(r["topic_ko"]).strip(), _norm_subtopic(r.get("subtopic_ko")))] += 1
    n_real_pairs = len(real_pairs)
    n_singleton = sum(1 for c in real_pairs.values() if c == 1)
    singleton_rate = round(n_singleton / n_real_pairs, 4) if n_real_pairs else 0.0

    concentration = subtopic_concentration(rows_with_topic)

    violations: list[dict[str, Any]] = []
    if unassigned_rate >= _UNASSIGNED_RATE_HARD:
        violations.append({
            "metric": "unassigned_rate", "scope": "corpus",
            "value": unassigned_rate, "threshold": _UNASSIGNED_RATE_HARD, "level": "hard",
        })
    if misc_rate >= _MISC_SUBTOPIC_RATE_HARD:
        violations.append({
            "metric": "misc_subtopic_rate", "scope": "corpus",
            "value": misc_rate, "threshold": _MISC_SUBTOPIC_RATE_HARD, "level": "hard",
        })
    elif misc_rate >= _MISC_SUBTOPIC_RATE_WARN:
        violations.append({
            "metric": "misc_subtopic_rate", "scope": "corpus",
            "value": misc_rate, "threshold": _MISC_SUBTOPIC_RATE_WARN, "level": "warn",
        })
    # topic별 소분류 집중도 — 표본 충분한 topic 만(정렬로 결정적 출력).
    for tk in sorted(concentration):
        c = concentration[tk]
        if c["n_assets"] < _MIN_TOPIC_ASSETS_FOR_SHARE:
            continue
        share = c["max_share"]
        lvl = "hard" if share >= _SUBTOPIC_MAX_SHARE_HARD else (
            "warn" if share >= _SUBTOPIC_MAX_SHARE_WARN else None)
        if lvl:
            thr = _SUBTOPIC_MAX_SHARE_HARD if lvl == "hard" else _SUBTOPIC_MAX_SHARE_WARN
            violations.append({
                "metric": "subtopic_max_share", "scope": tk,
                "value": share, "threshold": thr, "level": lvl,
                "dominant_subtopic": c["dominant_subtopic"],
            })

    levels = {v["level"] for v in violations}
    overall = "hard" if "hard" in levels else ("warn" if "warn" in levels else "ok")

    report: dict[str, Any] = {
        "metrics": {
            "n_registered": n_registered,
            "n_with_topic": n_with_topic,
            "n_unassigned": n_unassigned,
            "unassigned_rate": unassigned_rate,
            "n_misc_subtopic": n_misc,
            "misc_subtopic_rate": misc_rate,
            "n_real_pairs": n_real_pairs,
            "n_singleton_pairs": n_singleton,
            "singleton_pair_rate": singleton_rate,
            "subtopic_concentration": concentration,
        },
        "violations": violations,
        "level": overall,
    }
    if before_topic_counts is not None:
        after_counts = dict(topic_distribution(rows_with_topic))
        report["drift"] = topic_count_drift(before_topic_counts, after_counts)
    return report


# ────────────────────────────────────────────────────────────────────────────
# 1c) 고정 스모크셋 (068 FR-302·T402 — 늘리지 않음·재백필 후 대조)
# ────────────────────────────────────────────────────────────────────────────
_DEFAULT_SMOKE_PATH = _REPO_ROOT / "tests" / "golden" / "topic_smoke.json"


def load_topic_smoke(path: Path | str = _DEFAULT_SMOKE_PATH) -> list[dict[str, Any]]:
    """고정 스모크 앵커 목록 로드(순수 I/O). ``anchors`` 배열만 반환."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("anchors", []))


def validate_topic_smoke(golden: list[dict[str, Any]], valid_topics: set[str]) -> list[str]:
    """스모크 골든 무결성 검증(순수) — 문제 문자열 목록(빈 목록=정상).

    검사: expected_topic 이 None 이거나 27집합 소속 · hint 고유 · distinct_from 참조 존재.
    """
    problems: list[str] = []
    hints = [e.get("hint") for e in golden]
    seen: set[Any] = set()
    for h in hints:
        if h in seen:
            problems.append(f"hint 중복: {h!r}")
        seen.add(h)
    hint_set = set(hints)
    for e in golden:
        et = e.get("expected_topic")
        if et is not None and et not in valid_topics:
            problems.append(f"미지의 expected_topic: {et!r}(hint={e.get('hint')!r})")
        for ref in e.get("distinct_from", []) or []:
            if ref not in hint_set:
                problems.append(f"distinct_from 참조 없음: {ref!r}(hint={e.get('hint')!r})")
    return problems


def compare_topic_smoke(
    golden: list[dict[str, Any]], actual: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """스모크 골든 vs 실제 분류 결과 대조(순수·재백필 후 실행용).

    Args:
        golden: ``load_topic_smoke()`` 결과.
        actual: ``{hint: {"topic_ko": str|None, "subtopic_ko": str|None}}``(실제 재분류 결과).

    규칙:
        · topic: expected_topic 과 정확 일치(둘 다 None 도 일치). None 기대인데 부여되면 미부여 회귀.
        · subtopic: expected_subtopic 가 문자열이면 정확 일치, None 이면 와일드카드(대조 안 함).
        · distinct_from: 같은 topic 내에서 서로 다른 subtopic 이어야 함(둘 다 None/동일이면 과병합 위반).

    Returns:
        ``{n, n_topic_ok, n_subtopic_checked, n_subtopic_ok, topic_mismatches,
        subtopic_mismatches, separation_violations, passed}``.
    """
    def _act(hint: Any) -> tuple[Any, Any]:
        a = actual.get(hint) or {}
        return a.get("topic_ko"), _norm_subtopic(a.get("subtopic_ko"))

    topic_mismatches: list[dict[str, Any]] = []
    subtopic_mismatches: list[dict[str, Any]] = []
    separation_violations: list[dict[str, Any]] = []
    n_topic_ok = n_sub_checked = n_sub_ok = 0

    for e in golden:
        hint = e.get("hint")
        exp_t = e.get("expected_topic")
        exp_s = e.get("expected_subtopic")
        act_t, act_s = _act(hint)
        # 정규화: 빈 문자열 topic 은 None 취급.
        act_t = act_t if (act_t is not None and str(act_t).strip()) else None
        if exp_t == act_t:
            n_topic_ok += 1
        else:
            topic_mismatches.append({"hint": hint, "expected": exp_t, "actual": act_t})
        if exp_s is not None:  # None 은 와일드카드(닫힌 시드 확정 전) — 대조 생략
            n_sub_checked += 1
            if exp_s == act_s:
                n_sub_ok += 1
            else:
                subtopic_mismatches.append({"hint": hint, "expected": exp_s, "actual": act_s})

    # 분리 제약: distinct_from 앵커끼리 subtopic 이 달라야 함(중복 위반 1회만 기록).
    recorded: set[frozenset] = set()
    for e in golden:
        hint = e.get("hint")
        _, s_a = _act(hint)
        for ref in e.get("distinct_from", []) or []:
            key = frozenset({hint, ref})
            if key in recorded:
                continue
            recorded.add(key)
            _, s_b = _act(ref)
            if s_a == s_b:  # 둘 다 None 이거나 같은 라벨 → 변별 실패(과병합)
                separation_violations.append(
                    {"pair": [hint, ref], "shared_subtopic": s_a}
                )

    passed = not (topic_mismatches or subtopic_mismatches or separation_violations)
    return {
        "n": len(golden),
        "n_topic_ok": n_topic_ok,
        "n_subtopic_checked": n_sub_checked,
        "n_subtopic_ok": n_sub_ok,
        "topic_mismatches": topic_mismatches,
        "subtopic_mismatches": subtopic_mismatches,
        "separation_violations": separation_violations,
        "passed": passed,
    }


def format_report_lines(report: dict[str, Any]) -> list[str]:
    """리포트 dict → 콘솔 줄(순수·사람 검수용)."""
    dist = report["distribution"]
    grd = report["grounding"]
    a_pct = 100.0 * dist["assignment_rate"]
    p_pct = 100.0 * grd["pollution_rate"]
    lines = [
        "[자기주제 grounding 리포트 · 065 · 읽기전용]",
        "  — 집계 A: 부여율·분포 —",
        f"  registered {dist['n_registered']} · 텍스트 보유 {dist['n_with_text']} · "
        f"주제 부여 {dist['n_with_topic']}",
        f"  부여율(텍스트 보유 대비): {a_pct:.1f}%  (SC-03 목표 95%↑)",
        f"  미부여 사유: no_text {dist['n_no_text']} · 분류실패 {dist['n_classify_failed']}",
        "  topic 분포:",
    ]
    for topic, cnt in dist["topic_distribution"].items():
        lines.append(f"      {cnt:>4}  {topic}")
    lines += [
        "  — 집계 B: 근거율/오염율(SC-02) —",
        f"  주제 보유 {grd['n_assets']} · 근거 있음 {grd['n_grounded']} · "
        f"오염 {grd['n_polluted']}",
        f"  오염율: {p_pct:.1f}%  (SC-02 목표 ≤5% · 문자열 매칭 보수적 하한)",
    ]
    proj = report.get("projection_grounding")
    if proj is not None:
        pj_pct = 100.0 * proj["pollution_rate"]
        lines += [
            "  — (참고) 옛 이웃-엣지 투영 방식 대비 —",
            f"  투영 주제 보유 {proj['n_assets']} · 오염율: {pj_pct:.1f}%"
            f"  (정본 {p_pct:.1f}% 와 비교)",
        ]
    guard = report.get("guard")
    if guard is not None:
        lines += format_guard_lines(guard)
    return lines


_GUARD_LEVEL_MARK = {"ok": "✅", "warn": "🟡", "hard": "🔴"}


def format_guard_lines(guard: dict[str, Any]) -> list[str]:
    """분포 가드 리포트 dict → 콘솔 줄(순수·068 FR-301·집단 통계)."""
    m = guard["metrics"]
    lines = [
        f"  — 분포 가드(068 FR-301 · 집단 통계 · 임계 dev 실측 후 확정) {_GUARD_LEVEL_MARK.get(guard['level'], '')} —",
        f"  미부여율: {100.0 * m['unassigned_rate']:.1f}%  "
        f"(미부여 {m['n_unassigned']}/{m['n_registered']} · 하드 <{100 * _UNASSIGNED_RATE_HARD:.0f}%)",
        f"  '기타'/None subtopic 비율: {100.0 * m['misc_subtopic_rate']:.1f}%  "
        f"(경보 {100 * _MISC_SUBTOPIC_RATE_WARN:.0f}%·하드 {100 * _MISC_SUBTOPIC_RATE_HARD:.0f}%)",
        f"  실 소분류 싱글턴 비율: {100.0 * m['singleton_pair_rate']:.1f}%  "
        f"({m['n_singleton_pairs']}/{m['n_real_pairs']} pair)",
    ]
    if guard["violations"]:
        lines.append("  위반:")
        for v in guard["violations"]:
            mark = _GUARD_LEVEL_MARK.get(v["level"], "")
            extra = f" [{v['dominant_subtopic']}]" if v.get("dominant_subtopic") else ""
            lines.append(
                f"      {mark} {v['metric']} @ {v['scope']}{extra}: "
                f"{v['value']:.3f} (임계 {v['threshold']})"
            )
    else:
        lines.append("  위반 없음(정상).")
    drift = guard.get("drift")
    if drift is not None:
        lines.append(f"  드리프트 총 이동량(백필 전후): {drift['total_churn']}")
    return lines


# ────────────────────────────────────────────────────────────────────────────
# 2) DB 경로 (읽기전용 — 얇게)
# ────────────────────────────────────────────────────────────────────────────
# 자기주제 정본 + 자기 텍스트 소스(summary/keywords/labels) — 결정적 정렬.
_GROUNDING_SQL = """
SELECT at.asset_id, at.topic_ko, at.subtopic_ko,
       m.ext_meta->>'summary' AS summary,
       m.ext_meta->'keywords' AS keywords,
       m.ext_meta->'labels'   AS labels
FROM asset_topic at
JOIN asset a ON a.asset_id = at.asset_id
LEFT JOIN asset_metadata m ON m.asset_id = at.asset_id
WHERE a.status = 'registered'
ORDER BY at.asset_id
"""

# registered 중 자기 텍스트(summary 비지 않음 OR keywords 비지 않은 배열) 보유 자산 id.
_TEXT_ASSET_SQL = """
SELECT a.asset_id
FROM asset a
JOIN asset_metadata m ON m.asset_id = a.asset_id
WHERE a.status = 'registered'
  AND (
    COALESCE(m.ext_meta->>'summary', '') <> ''
    OR (
      jsonb_typeof(m.ext_meta->'keywords') = 'array'
      AND jsonb_array_length(m.ext_meta->'keywords') > 0
    )
  )
"""

_REGISTERED_COUNT_SQL = "SELECT count(*) FROM asset WHERE status = 'registered'"

# (참고) 옛 이웃-엣지 투영 재구성 — active 엣지의 topic 을 양끝 자산에 투영(옛 project_asset_topics
# 와 동형: 엣지 하나가 src·dst 두 자산에 그 topic 을 준다). 의료(PHI) 제외(헌법 10조).
_PROJECTION_SQL = """
SELECT n.asset_id AS asset_id,
       ge.topic->>'topic_ko'    AS topic_ko,
       ge.topic->>'subtopic_ko' AS subtopic_ko,
       m.ext_meta->>'summary' AS summary,
       m.ext_meta->'keywords' AS keywords,
       m.ext_meta->'labels'   AS labels
FROM graph_edge ge
JOIN node sn ON sn.node_id = ge.src_node AND sn.node_kind = 'asset'
JOIN node dn ON dn.node_id = ge.dst_node AND dn.node_kind = 'asset'
JOIN LATERAL (VALUES (sn.asset_id), (dn.asset_id)) AS n(asset_id) ON TRUE
JOIN asset a ON a.asset_id = n.asset_id AND a.status = 'registered'
LEFT JOIN asset_metadata m ON m.asset_id = n.asset_id
WHERE ge.status = 'active'
  AND COALESCE(ge.topic->>'topic_ko', '') <> ''
  AND a.domain_label IS DISTINCT FROM 'medical'
ORDER BY n.asset_id
"""


def _build_self_text(row: dict[str, Any]) -> str:
    """행의 summary/keywords/labels → 자기 텍스트(분류 seam 과 동일 구성·중복 구현 금지)."""
    from src.classify.asset_topic import build_self_text

    return build_self_text(row.get("summary"), row.get("keywords"), row.get("labels"))


def fetch_grounding_rows(conn) -> list[dict[str, Any]]:
    """자기주제 정본 + 자기 텍스트 평탄 행(읽기전용). ``self_text`` 는 여기서 구성."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_GROUNDING_SQL)
        rows = cur.fetchall()
    return [
        {
            "asset_id": str(r["asset_id"]),
            "topic_ko": r["topic_ko"],
            "subtopic_ko": r["subtopic_ko"],
            "self_text": _build_self_text(r),
        }
        for r in rows
    ]


def fetch_text_asset_ids(conn) -> set[str]:
    """registered 중 자기 텍스트 보유 자산 id 집합(읽기전용·부여율 분모)."""
    with conn.cursor() as cur:
        cur.execute(_TEXT_ASSET_SQL)
        return {str(r[0]) for r in cur.fetchall()}


def fetch_registered_count(conn) -> int:
    """registered 자산 총수(읽기전용)."""
    with conn.cursor() as cur:
        cur.execute(_REGISTERED_COUNT_SQL)
        return int(cur.fetchone()[0])


def fetch_projection_rows(conn) -> list[dict[str, Any]]:
    """(참고) 옛 이웃-엣지 투영 재구성 평탄 행(읽기전용·--compare-projection)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_PROJECTION_SQL)
        rows = cur.fetchall()
    return [
        {
            "asset_id": str(r["asset_id"]),
            "topic_ko": r["topic_ko"],
            "subtopic_ko": r["subtopic_ko"],
            "self_text": _build_self_text(r),
        }
        for r in rows
    ]


def _load_before_topic_counts(before_json: str | None) -> dict[str, int] | None:
    """이전 리포트 JSON 에서 백필 전 topic 건수(distribution.topic_distribution) 로드(드리프트용)."""
    if not before_json:
        return None
    prev = json.loads(Path(before_json).read_text(encoding="utf-8"))
    dist = prev.get("distribution", {})
    return {str(k): int(v) for k, v in (dist.get("topic_distribution") or {}).items()}


def run_report(
    *, env: str, compare_projection: bool = False, before_json: str | None = None
) -> dict[str, Any]:
    """리포트 실행(읽기전용 DB·LLM 0). .env.{env} 로드 → init_settings → 집계."""
    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    dotenv_path = _REPO_ROOT / f".env.{env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(env)

    db = PostgresUtil()
    with db, db.connection() as conn:
        grounding_flat = fetch_grounding_rows(conn)
        text_ids = fetch_text_asset_ids(conn)
        n_reg = fetch_registered_count(conn)
        proj_flat = fetch_projection_rows(conn) if compare_projection else None

    # topic_rows 는 grounding_flat 에서 파생(2차 쿼리 회피).
    topic_rows = [
        {"asset_id": r["asset_id"], "topic_ko": r["topic_ko"], "subtopic_ko": r["subtopic_ko"]}
        for r in grounding_flat
    ]
    report: dict[str, Any] = {
        "env": env,
        "distribution": build_distribution_report(
            topic_rows=topic_rows, text_asset_ids=text_ids, n_registered=n_reg
        ),
        "grounding": build_grounding_report(group_label_rows(grounding_flat)),
        # 068 FR-301 분포 가드(순수 build_guard_report 의 얇은 실DB 래퍼).
        "guard": build_guard_report(
            topic_rows=topic_rows, n_registered=n_reg,
            before_topic_counts=_load_before_topic_counts(before_json),
        ),
    }
    if proj_flat is not None:
        report["projection_grounding"] = build_grounding_report(group_label_rows(proj_flat))
    return report


def main() -> int:
    p = argparse.ArgumentParser(
        description="자기주제 근거율(grounding)·분포 리포트(spec 065·FR-602·읽기전용)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument(
        "--compare-projection", dest="compare_projection", action="store_true",
        help="(참고) 옛 이웃-엣지 투영 방식이었다면 오염율이 얼마였을지 비교",
    )
    p.add_argument("--json", dest="json_out", default=None, help="리포트 JSON 저장 경로(선택)")
    p.add_argument(
        "--before-json", dest="before_json", default=None,
        help="(068 드리프트) 백필 전 리포트 JSON 경로 — topic 건수 변동 비교",
    )
    p.add_argument(
        "--fail-on-hard", dest="fail_on_hard", action="store_true",
        help="(068 게이트) 분포 가드가 하드 임계 위반이면 비정상 종료코드(1) 반환",
    )
    args = p.parse_args()

    report = run_report(
        env=args.env, compare_projection=args.compare_projection, before_json=args.before_json
    )
    print("\n".join(format_report_lines(report)))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  리포트 JSON 저장: {args.json_out}")
    if args.fail_on_hard and report.get("guard", {}).get("level") == "hard":
        print("🔴 분포 가드 하드 임계 위반 — 재백필/시드 점검 필요(068 FR-301).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
