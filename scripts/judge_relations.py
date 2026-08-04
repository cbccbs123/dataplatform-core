"""관계 맹검 판정 러너 — 층화 표본을 뽑아 제3자 LLM 에게 판정시키고 파일로 남긴다.

**흐름에서의 위치**: `graph_edge` 를 **읽기만** 하고, 판정 결과를
``tests/fixtures/relations/verdicts/<measure_id>.json`` 으로 쓴다. DB 에는 아무것도 쓰지 않는다.

**맹검이란**: 판정자에게 시스템이 붙인 ``relation_kind``·``confidence``·``reason`` 을 **주지
않는다**. 양끝 자산 내용만 보고 "이 둘이 실제로 관련 있나"를 답하게 한다. 시스템 로직을 그대로
되읽는 순환 평가를 막기 위해서다.

**설계 판단 — 왜 `graph_query` seam 을 쓰지 않는가**: 그 seam 은 *특정 자산의 이웃*을 찾을 때
필요하다(대칭 엣지가 dst 로 접혀 있어 단방향 조회로는 누락된다). 여기서는 **엣지 테이블 전체를
표본추출**하는데, 대칭 엣지도 행은 하나뿐이라 누락이 발생하지 않는다. 그래서 직접 조회가 맞다.

**깨지면 안 되는 것**
- 표본은 ``seed`` 로 재현된다. SQL 은 ``ORDER BY ge.edge_id`` 로 전순서를 확정한다 —
  이게 없으면 같은 시드라도 행 순서가 달라져 표본이 흔들린다.
- 판정 실패율이 5%를 넘으면 그 측정을 **무효**로 본다(전량 실패가 0%로 조용히 집계된 전례).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from psycopg.rows import dict_row

from src.config.settings import get_current_settings, init_settings
from src.database.postgres_util import PostgresUtil
from src.llm.client import complete_json
from src.relations.quality import VERDICTS_DIR
from src.relations.quality.metrics import cohen_kappa
from src.relations.quality.rubric import RUBRIC_KO_V1, RUBRIC_VERSION
from src.relations.quality.verdicts import (
    VALID_VERDICTS,
    Verdict,
    VerdictSet,
    dump_verdicts,
    error_rate,
    load_verdicts,
    verdict_counts,
)

# 온프레미스 LLM 부하를 고려한 보수적 동시성. 올리기 전에 판정 지연을 먼저 재라.
_CONCURRENCY = 6

# 판정 실패가 이 비율을 넘으면 측정 자체를 무효로 본다.
# (실제로 366건 전량이 RuntimeError 였는데 집계는 "strong 0% weak 0%" 로 나와 정상처럼 보였다.)
ERROR_RATE_MAX = 0.05

# confidence 구간 — 실측에서 의미가 갈린 지점을 그대로 경계로 삼는다
# (0.90/0.92 는 표본상 구분되지 않아 한 칸으로 묶고, 0.95 는 확실히 다르므로 가른다).
CONF_BAND_SQL = """CASE
    WHEN ge.confidence < 0.60 THEN 'c<0.60'
    WHEN ge.confidence < 0.75 THEN 'c0.60-0.75'
    WHEN ge.confidence < 0.90 THEN 'c0.75-0.90'
    WHEN ge.confidence < 0.95 THEN 'c0.90-0.95'
    ELSE 'c>=0.95' END"""

# 층화 축 → 셀 라벨 SQL 식. 셀 단위로 같은 수만큼 뽑아 소수 종류가 묻히지 않게 한다.
CELL_EXPR = {
    "kind": "rk.kind_code",
    "conf": CONF_BAND_SQL,
    "cohort": "ge.created_at::date::text",
    "kind-conf": f"rk.kind_code || '/' || ({CONF_BAND_SQL})",
}

# 표본 행 — 판정에 필요한 양끝 자산 내용까지 한 번에 읽는다(읽기 전용).
# ``{cell}`` 만 치환한다. status 는 파라미터로 넘긴다.
SAMPLE_SQL = """
    SELECT ge.edge_id::text AS edge_id, ({cell}) AS cell,
           rk.kind_code, ge.confidence, ge.created_at::date::text AS created_on,
           sa.modality AS a_mod, regexp_replace(sa.fs_path,'^.*/','') AS a_name,
           coalesce(sat.topic_ko,'(없음)') AS a_topic,
           left(coalesce(sam.ext_meta->>'summary',''), 300) AS a_sum,
           coalesce(sam.ext_meta->>'keywords','[]') AS a_kw,
           da.modality AS b_mod, regexp_replace(da.fs_path,'^.*/','') AS b_name,
           coalesce(dat.topic_ko,'(없음)') AS b_topic,
           left(coalesce(dam.ext_meta->>'summary',''), 300) AS b_sum,
           coalesce(dam.ext_meta->>'keywords','[]') AS b_kw
    FROM graph_edge ge
    JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
    -- node_kind='asset' 가드는 레포 관례다(graph_query·review·asset_topic_query 동일).
    -- entity 노드는 asset_id 가 NULL 이라(chk_node_kind 가 asset 일 때만 NOT NULL 강제)
    -- 가드가 없으면 asset 조인에 기대 "우연히" 걸러지는 상태가 된다 — 명시해 둔다.
    JOIN node sn ON sn.node_id = ge.src_node AND sn.node_kind = 'asset'
    JOIN asset sa ON sa.asset_id = sn.asset_id
    JOIN node dn ON dn.node_id = ge.dst_node AND dn.node_kind = 'asset'
    JOIN asset da ON da.asset_id = dn.asset_id
    LEFT JOIN asset_metadata sam ON sam.asset_id = sa.asset_id
    LEFT JOIN asset_metadata dam ON dam.asset_id = da.asset_id
    LEFT JOIN asset_topic sat ON sat.asset_id = sa.asset_id
    LEFT JOIN asset_topic dat ON dat.asset_id = da.asset_id
    WHERE ge.status = %s
    ORDER BY ge.edge_id
"""

# ⚠️ 시스템이 붙인 kind·confidence·reason 은 **넣지 않는다**(맹검). 양끝 내용만 준다.
JUDGE_PROMPT = """두 자산의 내용만 보고 **실제로 관련이 있는지** 판정해라. 시스템이 왜 이었는지는 모른다.

{rubric}

자산 A
  파일: {a_name}
  종류: {a_mod}
  주제: {a_topic}
  요약: {a_sum}
  키워드: {a_kw}

자산 B
  파일: {b_name}
  종류: {b_mod}
  주제: {b_topic}
  요약: {b_sum}
  키워드: {b_kw}

JSON 하나만 출력: {{"verdict":"strong|weak|none","why":"20자 이내 근거"}}"""


def group_and_sample(rows: list[dict], *, per_cell: int, seed: int) -> list[dict]:
    """셀별로 최대 ``per_cell`` 건씩 뽑는다(순수 함수·시드 재현).

    풀이 ``per_cell`` 보다 작으면 그 셀은 **전수**를 쓴다 — 비율만 보고하면 n 이 작은 셀이
    큰 셀과 같은 무게로 읽히므로, 호출자는 실제 n 을 함께 보고해야 한다.

    Args:
        rows: ``cell`` 과 ``edge_id`` 키를 가진 표본 후보 행들.
        per_cell: 셀당 표본 상한.
        seed: 난수 시드. 같은 값이면 같은 표본이 나온다.

    Returns:
        ``edge_id`` 오름차순으로 정렬된 표본 행 목록.
    """
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(str(r["cell"]), []).append(r)
    rnd = random.Random(seed)
    picked: list[dict] = []
    # 셀 순회 순서도 고정한다 — dict 삽입 순서에 기대면 입력 정렬이 바뀔 때 표본이 흔들린다.
    for cell in sorted(by_cell):
        pool = sorted(by_cell[cell], key=lambda r: str(r["edge_id"]))
        picked += rnd.sample(pool, min(per_cell, len(pool)))
    return sorted(picked, key=lambda r: str(r["edge_id"]))


def build_judge_prompt(row: dict, rubric: str) -> str:
    """표본 행 하나를 맹검 판정 프롬프트로 만든다(순수 함수).

    Args:
        row: ``SAMPLE_SQL`` 이 낸 행. 양끝 자산의 이름·모달리티·주제·요약·키워드를 쓴다.
        rubric: 판정 기준 원문(``rubric.RUBRIC_KO_V1``).

    Returns:
        완성된 프롬프트. **``kind_code``·``confidence`` 는 의도적으로 빠져 있다.**
    """
    return JUDGE_PROMPT.format(
        rubric=rubric,
        a_name=str(row["a_name"])[:70], a_mod=row["a_mod"], a_topic=row["a_topic"],
        a_sum=row["a_sum"] or "(없음)", a_kw=str(row["a_kw"])[:150],
        b_name=str(row["b_name"])[:70], b_mod=row["b_mod"], b_topic=row["b_topic"],
        b_sum=row["b_sum"] or "(없음)", b_kw=str(row["b_kw"])[:150])


def judge_one(row: dict, *, rubric: str, llm: Callable[[str], dict[str, Any]]) -> Verdict:
    """한 쌍을 맹검 판정한다(LLM 1회).

    한 건이 실패해도 측정 전체를 멈추지 않는다 — 대신 ``error`` 로 남겨 상위가 실패율을
    집계할 수 있게 한다(그래야 "전량 실패인데 0%로 보이는" 사고가 드러난다).

    Args:
        row: 표본 행.
        rubric: 판정 기준 원문.
        llm: 프롬프트를 받아 dict 를 주는 함수. 운영은 ``src.llm.client.complete_json``,
            테스트는 고정 응답 함수를 넣는다.

    Returns:
        ``Verdict``. 실패 시 ``verdict="error"`` 이고 ``why`` 는 예외 타입명이다
        (예외 메시지는 경로·본문을 담을 수 있어 넣지 않는다).
    """
    prompt = build_judge_prompt(row, rubric)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    try:
        out = llm(prompt)
        raw = str(out.get("verdict", "")).strip().lower()
        ok = raw in (VALID_VERDICTS - {"error"})
        return Verdict(str(row["edge_id"]), raw if ok else "error",
                       str(out.get("why", ""))[:20], digest)
    except Exception as exc:  # noqa: BLE001 — 한 건 실패가 측정 전체를 멈추지 않게
        return Verdict(str(row["edge_id"]), "error", type(exc).__name__, digest)


def human_kappa(vs: VerdictSet) -> tuple[float, int]:
    """LLM 판정과 사람 교차판정의 일치도.

    ``error``(호출 실패)와 사람 미판정은 비교 대상에서 뺀다 — 실패를 불일치로 세면 κ 가
    실제보다 나빠 보인다.

    Args:
        vs: 사람 판정이 일부 채워진 측정 묶음.

    Returns:
        ``(κ, 비교한 건수)``. 비교할 게 없으면 ``(0.0, 0)``.
    """
    pairs = [(v.verdict, v.judged_by_human) for v in vs.verdicts
             if v.judged_by_human and v.verdict != "error"]
    return (cohen_kappa(pairs), len(pairs))


def run_judge(
    db: PostgresUtil, *, measure_id: str, method: str, status: str, strata: str,
    per_cell: int, seed: int, out_path: str,
) -> VerdictSet:
    """표본을 뽑아 맹검 판정하고 파일로 저장한다.

    Args:
        db: DB 핸들(읽기 전용으로만 쓴다).
        measure_id: 측정 식별자. 파일명이 된다.
        method: 이 측정이 무엇을 반증하려 하는지 한 문단.
        status: 표본 대상 엣지 상태(``active``·``proposed``).
        strata: 층화 축(``CELL_EXPR`` 의 키).
        per_cell: 셀당 표본 상한.
        seed: 표본 시드.
        out_path: 저장 경로.

    Returns:
        저장한 ``VerdictSet``.

    Raises:
        ValueError: ``strata`` 가 ``CELL_EXPR`` 에 없을 때.
    """
    if strata not in CELL_EXPR:
        raise ValueError(f"알 수 없는 층화 축: {strata!r} (가능: {sorted(CELL_EXPR)})")
    with db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(SAMPLE_SQL.format(cell=CELL_EXPR[strata]), (status,))
        rows = cur.fetchall()
    sample = group_and_sample(rows, per_cell=per_cell, seed=seed)
    print(f"표본 {len(sample)}건({strata}·{status}) — 판정 시작(동시 {_CONCURRENCY})", flush=True)
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as ex:
        results = list(ex.map(
            lambda r: judge_one(r, rubric=RUBRIC_KO_V1, llm=complete_json), sample))
    vs = VerdictSet(
        measure_id=measure_id, method=method,
        rubric_version=RUBRIC_VERSION, rubric_text=RUBRIC_KO_V1,
        judge_model=get_current_settings().meta_model, seed=seed, strata=strata,
        created_at=datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        sample_edge_ids=tuple(str(r["edge_id"]) for r in sample),
        verdicts=tuple(results))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(dump_verdicts(vs), ensure_ascii=False, indent=1), encoding="utf-8")
    return vs


def run_human(path: str, *, limit: int) -> tuple[float, int]:
    """저장된 판정 파일에서 앞 ``limit`` 건을 사람에게 물어 κ 를 낸다.

    ⚠️ **LLM 판정을 화면에 보여주지 않는다** — 보여주면 사람이 그 답에 끌려가(anchoring)
    일치도가 부풀려진다. 양쪽 다 맹검이어야 κ 가 의미를 갖는다.

    Args:
        path: 판정 파일 경로.
        limit: 물어볼 건수.

    Returns:
        ``(κ, 비교 건수)``.
    """
    path_obj = Path(path)
    vs = load_verdicts(json.loads(path_obj.read_text(encoding="utf-8")))
    # 대상은 edge_id 순 앞에서부터 — 무작위로 고르면 재실행 때 다른 건을 묻게 된다.
    todo = [v for v in sorted(vs.verdicts, key=lambda v: v.edge_id)
            if v.verdict != "error" and not v.judged_by_human][:limit]
    if not todo:
        print("사람이 판정할 건이 없다(이미 다 했거나 표본이 전부 error).")
        return human_kappa(vs)

    rows = _fetch_display_rows(tuple(v.edge_id for v in todo))
    # 내용을 못 읽은 엣지는 화면이 '?' 로만 채워진다 — 그 상태로 판정하면 κ 가 무의미해지므로
    # 조용히 넘기지 않고 알린다(엣지가 삭제됐거나 다른 DB 를 보고 있다는 신호다).
    if len(rows) < len(todo):
        missing = [v.edge_id for v in todo if v.edge_id not in rows]
        print(f"⚠️ 표시 내용을 못 읽은 엣지 {len(missing)}건 — 내용 없이 판정하지 말고 원인을 먼저 "
              f"확인하라(예: {missing[:3]}). 다른 DB 를 보고 있거나 엣지가 삭제된 경우다.")
        todo = [v for v in todo if v.edge_id in rows]
        if not todo:
            return human_kappa(vs)
    answers: dict[str, str] = {}
    key_to_verdict = {"s": "strong", "w": "weak", "n": "none"}
    for i, v in enumerate(todo, 1):
        r = rows.get(v.edge_id, {})
        print(f"\n[{i}/{len(todo)}]")
        print(f"  A: {r.get('a_name','?')} ({r.get('a_mod','?')})  {r.get('a_sum','')[:80]}")
        print(f"  B: {r.get('b_name','?')} ({r.get('b_mod','?')})  {r.get('b_sum','')[:80]}")
        print("  s=strong  w=weak  n=none  ?=보류(건너뜀)")
        while True:
            key = input("  > ").strip().lower()
            if key in key_to_verdict:
                answers[v.edge_id] = key_to_verdict[key]
                break
            if key == "?":
                break            # 보류 — judged_by_human 을 비워 둔다
            print("  s / w / n / ? 중 하나를 입력하라.")

    updated = tuple(
        Verdict(v.edge_id, v.verdict, v.why, v.prompt_sha256,
                answers.get(v.edge_id, v.judged_by_human))
        for v in vs.verdicts)
    vs2 = replace(vs, verdicts=updated)
    path_obj.write_text(json.dumps(dump_verdicts(vs2), ensure_ascii=False, indent=1),
                        encoding="utf-8")
    kappa, n = human_kappa(vs2)
    print(f"\nκ = {kappa:.3f}  (비교 {n}건)")
    if n and kappa < 0.4:
        print("⚠️ κ<0.4 — 맹검 LLM 판정을 품질 판정 수단으로 쓰는 것 자체를 재검토하라.")
    return (kappa, n)


# 표시용 조회 — 요청한 edge_id 만 **상태와 무관하게** 집어 온다.
# ⚠️ SAMPLE_SQL 을 재사용하면 안 된다. 그쪽은 `WHERE ge.status = %s` 라, `--status proposed` 로
#    만든 판정 파일(M5·M6)을 `--human` 으로 열면 한 건도 못 찾고 화면이 '?' 로만 채워진다.
#    예외도 경고도 없이 사람이 **내용 없이 판정하게 되는** 조용한 실패다(리뷰 지적).
DISPLAY_SQL = """
    SELECT ge.edge_id::text AS edge_id,
           sa.modality AS a_mod, regexp_replace(sa.fs_path,'^.*/','') AS a_name,
           coalesce(sat.topic_ko,'(없음)') AS a_topic,
           left(coalesce(sam.ext_meta->>'summary',''), 300) AS a_sum,
           da.modality AS b_mod, regexp_replace(da.fs_path,'^.*/','') AS b_name,
           coalesce(dat.topic_ko,'(없음)') AS b_topic,
           left(coalesce(dam.ext_meta->>'summary',''), 300) AS b_sum
    FROM graph_edge ge
    JOIN node sn ON sn.node_id = ge.src_node AND sn.node_kind = 'asset'
    JOIN asset sa ON sa.asset_id = sn.asset_id
    JOIN node dn ON dn.node_id = ge.dst_node AND dn.node_kind = 'asset'
    JOIN asset da ON da.asset_id = dn.asset_id
    LEFT JOIN asset_metadata sam ON sam.asset_id = sa.asset_id
    LEFT JOIN asset_metadata dam ON dam.asset_id = da.asset_id
    LEFT JOIN asset_topic sat ON sat.asset_id = sa.asset_id
    LEFT JOIN asset_topic dat ON dat.asset_id = da.asset_id
    WHERE ge.edge_id = ANY(%s)
"""


def _fetch_display_rows(edge_ids: tuple[str, ...]) -> dict[str, dict]:
    """사람에게 보여줄 양끝 자산 내용을 읽는다(읽기 전용·상태 무관).

    판정 파일에는 본문을 저장하지 않으므로(개인정보) 화면 표시용으로 그때그때 DB 에서 읽는다.
    ``active``·``proposed`` 어느 표본이든 열 수 있어야 하므로 **status 를 조건에 넣지 않고**
    ``edge_id`` 로 직접 집는다.

    Args:
        edge_ids: 표시할 엣지들.

    Returns:
        ``{edge_id: 표시용 행}``. 못 찾은 엣지는 키가 없다(호출부가 경고한다).
    """
    db = PostgresUtil()
    with db, db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(DISPLAY_SQL, (list(edge_ids),))
        # ⚠️ dict_row 커서다 — dict(cur.fetchall()) 로 감싸면 키/값이 뒤집힌다(실제로 겪은 사고).
        return {r["edge_id"]: r for r in cur.fetchall()}


def main(argv: list[str] | None = None) -> int:
    """관계 맹검 판정 CLI 진입점 — 판정 실행(기본)과 사람 교차판정(``--human``)을 분기한다.

    ⚠️ 무엇을 하든 **``init_settings`` 를 먼저 부른다**. 빠뜨리면 LLM 호출마다 ``RuntimeError``
    가 나는데, 건별 실패는 ``error`` 로 흡수되므로 전량 실패가 "strong 0%" 로 조용히 집계된다
    (실제로 366건을 그렇게 날린 적이 있다).

    Args:
        argv: 명령행 인자. ``None`` 이면 실제 인자를 읽는다(테스트 주입용).

    Returns:
        0=정상. **2 = error 율이 ``ERROR_RATE_MAX`` 를 넘어 측정이 무효**(파일은 증거로 남긴다).
    """
    p = argparse.ArgumentParser(description="관계 맹검 판정 러너 (spec 079) — DB 읽기 전용")
    p.add_argument("--env", choices=["dev", "prod"], default="dev",
                   help="설정 프로파일(기본: dev). .env.<env> 를 읽어 초기화한다")
    p.add_argument("--human", action="store_true",
                   help="사람 교차판정 모드 — 저장된 판정 파일에 judged_by_human 을 채우고 κ 를 낸다")
    p.add_argument("--file", default=None, help="[--human] 판정 파일 경로")
    p.add_argument("--limit", type=int, default=30,
                   help="[--human] 사람에게 물어볼 건수 상한(기본: 30)")
    p.add_argument("--measure-id", dest="measure_id", default=None,
                   help="측정 식별자(YYYYMMDD-<축>). 저장 파일명이 된다")
    p.add_argument("--method", default=None,
                   help="이 측정이 무엇을 반증하려 하는지 한 문단(판정 파일에 그대로 남는다)")
    p.add_argument("--status", choices=["active", "proposed"], default="active",
                   help="표본 대상 엣지 상태(기본: active)")
    p.add_argument("--strata", choices=sorted(CELL_EXPR), default="kind",
                   help="층화 축 — 셀마다 같은 수를 뽑아 소수 종류가 묻히지 않게 한다(기본: kind)")
    p.add_argument("--per-cell", dest="per_cell", type=int, default=30,
                   help="셀당 표본 상한. 풀이 더 작으면 그 셀은 전수(기본: 30)")
    p.add_argument("--seed", type=int, default=20260728,
                   help="표본 추출 시드 — 같은 값이면 같은 표본이 나온다(기본: 20260728)")
    p.add_argument("--out", default=None,
                   help=f"저장 경로(미지정: {VERDICTS_DIR}/<measure-id>.json)")
    args = p.parse_args(argv)

    # 설정 초기화가 먼저다(위 ⚠️ 참조). 사람 모드도 DB 를 읽으므로 예외 없이 초기화한다.
    dotenv_path = Path(__file__).resolve().parents[1] / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    if args.human:
        if not args.file:
            p.error("--human 에는 --file <verdicts.json> 이 필요하다")
        run_human(args.file, limit=args.limit)
        return 0

    if not args.measure_id or not args.method:
        p.error("--measure-id 와 --method 는 필수다(측정의 정체를 파일에 남긴다)")
    out_path = args.out or str(Path(VERDICTS_DIR) / f"{args.measure_id}.json")

    db = PostgresUtil()
    with db:
        vs = run_judge(db, measure_id=args.measure_id, method=args.method,
                       status=args.status, strata=args.strata,
                       per_cell=args.per_cell, seed=args.seed, out_path=out_path)

    # 집계 요약 — ``err`` 열은 생략하지 않는다. 이 열이 없으면 전량 실패를 "strong 0%" 로
    # 읽는 사고가 재발한다(spec 엣지케이스 1).
    counts = verdict_counts(vs)
    err = error_rate(vs)
    total = len(vs.verdicts)
    rated = total - counts.get("error", 0)
    print(f"\n저장: {out_path}")
    print("strong  weak  none   err     n  strong율(판정된 것 기준)")
    print(f"{counts.get('strong', 0):6d}{counts.get('weak', 0):6d}{counts.get('none', 0):6d}"
          f"{counts.get('error', 0):6d}{total:6d}  "
          f"{(counts.get('strong', 0) / rated if rated else 0.0):.1%}")
    if err > ERROR_RATE_MAX:
        print(f"⚠️ error 율 {err:.1%} > {ERROR_RATE_MAX:.0%} — **이 측정은 무효다**. "
              "설정 초기화·LLM 접속을 확인하고 다시 돌려라(파일은 증거로 남긴다).")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
