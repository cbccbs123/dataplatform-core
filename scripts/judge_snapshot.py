"""스냅샷 제안 엣지의 맹검 판정 — shadow A/B 의 strong 절대수 비교용 (spec 079 T505).

**흐름에서의 위치**: `measure_relation_quality snapshot` 이 얼린 두 스냅샷(대조군/실험군)을 받아,
제안된 (소스, 타깃) 자산 쌍을 **한 배치에서** 판정하고 `verdicts` 파일로 남긴다. DB 는 읽기만 한다.

**왜 `judge_relations.py` 로 안 되는가**: 그쪽은 `graph_edge` 에 실재하는 엣지를 `edge_id` 로
판정한다. 스냅샷의 제안 엣지는 **DB 에 없다**(shadow 실험이라 기록하지 않으므로). 그래서 자산 쌍을
직접 먹이는 경로가 따로 필요하다.

**같은 것을 재사용한다** — 루브릭(`RUBRIC_KO_V1`), 프롬프트 조립(`build_judge_prompt`), 판정
호출과 실패 흡수(`judge_one`)를 `judge_relations` 에서 그대로 가져온다. 판정 기준이 갈라지면
A/B 결과를 기존 측정과 비교할 수 없다.

**맹검 이중화**: 판정자에게 ① 시스템이 붙인 `kind`·`confidence` 를 숨기고(`build_judge_prompt` 가
이미 그렇게 만든다) ② **어느 팔(A/B)인지도 숨긴다.** 두 스냅샷의 쌍을 섞어 한 배치로 돌리므로
판정자는 자기가 대조군을 보는지 실험군을 보는지 알 수 없다.

**깨지면 안 되는 것**
- 같은 쌍이 A·B 양쪽에 나오면 **각각 따로 판정하지 않는다** — 같은 자산 쌍은 판정이 같아야 하므로
  한 번만 판정하고 결과를 양쪽에 공유한다(LLM 호출 절감 + 판정 흔들림 제거).
- `edge_id` 자리에 스냅샷 쌍 키(`<source>__<target>`)를 넣는다. DB 엣지가 아니므로 실제
  `edge_id` 와 섞이지 않게 구분 가능한 형태를 쓴다.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from psycopg.rows import dict_row
from scripts.judge_relations import _CONCURRENCY, judge_one

from src.config.settings import get_current_settings, init_settings
from src.database.postgres_util import PostgresUtil
from src.llm.client import complete_json
from src.relations.quality.rubric import RUBRIC_KO_V1, RUBRIC_VERSION
from src.relations.quality.snapshot import Snapshot, load_snapshot
from src.relations.quality.verdicts import (
    VerdictSet,
    dump_verdicts,
    error_rate,
    verdict_counts,
)

# 판정에 필요한 자산 내용 — `judge_relations.DISPLAY_SQL` 과 같은 필드를 자산 단위로 읽는다.
ASSET_SQL = """
    SELECT a.asset_id::text AS asset_id, a.modality,
           regexp_replace(a.fs_path,'^.*/','') AS name,
           coalesce(t.topic_ko,'(없음)') AS topic,
           left(coalesce(m.ext_meta->>'summary',''), 300) AS summary,
           coalesce(m.ext_meta->>'keywords','[]') AS keywords
    FROM asset a
    LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id
    LEFT JOIN asset_topic t ON t.asset_id = a.asset_id
    WHERE a.asset_id = ANY(%s)
"""


def pair_key(source: str, target: str) -> str:
    """스냅샷 쌍의 판정 키 — 방향 무관하게 같은 쌍이 같은 키를 갖는다.

    대칭 정렬을 하는 이유: A 팔에서 (x→y) 로, B 팔에서 (y→x) 로 제안돼도 **같은 자산 쌍**이라
    판정이 같아야 한다. 정렬하지 않으면 같은 쌍을 두 번 판정해 결과가 흔들릴 수 있다.

    Args:
        source: 소스 자산 id.
        target: 타깃 자산 id.

    Returns:
        ``"<작은id>__<큰id>"``. DB ``edge_id``(UUID 단일)와 형태가 달라 섞이지 않는다.
    """
    a, b = sorted((str(source), str(target)))
    return f"{a}__{b}"


def collect_pairs(snap: Snapshot) -> dict[str, tuple[str, str]]:
    """스냅샷의 제안 엣지를 판정 대상 자산 쌍으로 모은다(순수 함수).

    Args:
        snap: 동결 스냅샷.

    Returns:
        ``{쌍 키: (소스 자산 id, 타깃 자산 id)}``. 같은 쌍이 여러 소스에서 나와도 하나로 접힌다.
    """
    out: dict[str, tuple[str, str]] = {}
    for sid, ss in snap.sources.items():
        for e in ss.proposed:
            out.setdefault(pair_key(sid, e.target), (sid, e.target))
    return out


def fetch_assets(db: PostgresUtil, asset_ids: list[str]) -> dict[str, dict]:
    """판정에 쓸 자산 내용을 읽는다(읽기 전용).

    Args:
        db: DB 핸들.
        asset_ids: 읽을 자산 id 목록.

    Returns:
        ``{asset_id: 행}``. 못 찾은 자산은 키가 없다(호출부가 그 쌍을 건너뛴다).
    """
    with db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(ASSET_SQL, (asset_ids,))
        # ⚠️ dict_row 커서다 — dict(cur.fetchall()) 로 감싸면 키/값이 뒤집힌다.
        return {r["asset_id"]: r for r in cur.fetchall()}


def to_judge_row(key: str, a: dict, b: dict) -> dict:
    """자산 두 건을 `judge_relations.build_judge_prompt` 가 먹는 행 모양으로 맞춘다.

    Args:
        key: 쌍 키(판정 결과의 ``edge_id`` 자리에 들어간다).
        a: 소스 자산 행.
        b: 타깃 자산 행.

    Returns:
        ``build_judge_prompt`` 가 요구하는 키를 갖춘 dict.
    """
    return {"edge_id": key,
            "a_name": a["name"], "a_mod": a["modality"], "a_topic": a["topic"],
            "a_sum": a["summary"], "a_kw": a["keywords"],
            "b_name": b["name"], "b_mod": b["modality"], "b_topic": b["topic"],
            "b_sum": b["summary"], "b_kw": b["keywords"]}


def judge_snapshots(db: PostgresUtil, snaps: dict[str, Snapshot]) -> tuple[VerdictSet, dict]:
    """여러 스냅샷의 제안 쌍을 **합집합으로 한 번씩** 판정한다.

    같은 쌍이 두 팔에 모두 있으면 한 번만 판정해 양쪽이 같은 결과를 공유한다 — 판정을 두 번 하면
    같은 쌍에 다른 라벨이 붙어 A/B 차이가 판정 잡음으로 오염된다.

    Args:
        db: DB 핸들(읽기 전용).
        snaps: ``{팔 이름: 스냅샷}``.

    Returns:
        ``(판정 묶음, {팔 이름: 그 팔의 쌍 키 집합})``.
    """
    per_arm = {arm: collect_pairs(s) for arm, s in snaps.items()}
    union: dict[str, tuple[str, str]] = {}
    for pairs in per_arm.values():
        union.update(pairs)

    need = sorted({x for pair in union.values() for x in pair})
    assets = fetch_assets(db, need)
    missing = [k for k, (s, t) in union.items() if s not in assets or t not in assets]
    if missing:
        print(f"⚠️ 자산 내용을 못 읽은 쌍 {len(missing)}건 — 판정에서 제외한다(예: {missing[:2]}).")

    rows = [to_judge_row(k, assets[s], assets[t]) for k, (s, t) in sorted(union.items())
            if s in assets and t in assets]
    print(f"판정 대상 쌍 {len(rows)}건(합집합) — 시작(동시 {_CONCURRENCY})", flush=True)
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as ex:
        results = list(ex.map(
            lambda r: judge_one(r, rubric=RUBRIC_KO_V1, llm=complete_json), rows))

    vs = VerdictSet(
        measure_id="20260728-shadow-ab",
        method=("shadow A/B — 관계 종류 힌트의 순환 지시 제거(prompt.py:98 duplicate_near · "
                ":135 anti_dup) 효과 측정. 두 팔의 제안 쌍을 합집합으로 한 번씩 판정하고, "
                "판정자에게는 kind·confidence 는 물론 **어느 팔인지도** 숨긴다."),
        rubric_version=RUBRIC_VERSION, rubric_text=RUBRIC_KO_V1,
        judge_model=get_current_settings().meta_model, seed=0, strata="shadow-ab-pair",
        created_at=datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        sample_edge_ids=tuple(r["edge_id"] for r in rows), verdicts=tuple(results))
    return vs, {arm: set(pairs) for arm, pairs in per_arm.items()}


def compare_arms(snaps: dict[str, Snapshot], vs: VerdictSet, arm_pairs: dict[str, set]) -> dict:
    """팔별 4개 지표를 낸다 — spec 폐기 기준 1·2·3 판정 근거.

    Args:
        snaps: ``{팔 이름: 스냅샷}``.
        vs: 합집합 판정 결과.
        arm_pairs: ``{팔 이름: 쌍 키 집합}``.

    Returns:
        ``{팔 이름: {edge_count, kind_dist, strong_count, weak_count, rated, assets_with_edge}}``.
    """
    verdict_of = {v.edge_id: v.verdict for v in vs.verdicts}
    out: dict[str, dict] = {}
    for arm, snap in snaps.items():
        kinds: dict[str, int] = {}
        assets: set[str] = set()
        n_edges = 0
        for sid, ss in snap.sources.items():
            for e in ss.proposed:
                n_edges += 1
                kinds[e.kind] = kinds.get(e.kind, 0) + 1
                assets.add(sid)
                assets.add(e.target)
        judged = [verdict_of[k] for k in arm_pairs[arm]
                  if verdict_of.get(k) in ("strong", "weak", "none")]
        out[arm] = {
            "edge_count": n_edges,
            "pair_count": len(arm_pairs[arm]),
            "kind_dist": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
            "rated": len(judged),
            "strong_count": judged.count("strong"),
            "weak_count": judged.count("weak"),
            "none_count": judged.count("none"),
            "assets_with_edge": len(assets),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """두 스냅샷을 맹검 판정하고 팔별 지표를 출력한다.

    Args:
        argv: 명령행 인자. ``None`` 이면 실제 인자를 읽는다(테스트 주입용).

    Returns:
        0=정상, 2=판정 실패율이 임계를 넘어 측정 무효.
    """
    ap = argparse.ArgumentParser(description="스냅샷 제안 엣지 맹검 판정(spec 079 T505)")
    ap.add_argument("--env", choices=["dev", "prod"], default="dev",
                    help="설정 프로파일(기본: dev). .env.<env> 를 읽어 초기화한다")
    ap.add_argument("--arm", action="append", metavar="이름=경로", required=True,
                    help="비교할 팔(예: --arm A=/tmp/snap_A.json --arm B=/tmp/snap_B.json)")
    ap.add_argument("--out", required=True, help="판정 결과 저장 경로")
    args = ap.parse_args(argv)

    dotenv_path = Path(__file__).resolve().parents[1] / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    # ⚠️ LLM seam 은 활성 설정에서 클라이언트를 만든다 — 이걸 빠뜨리면 호출마다 RuntimeError 다.
    init_settings(args.env)

    snaps: dict[str, Snapshot] = {}
    for spec in args.arm:
        if "=" not in spec:
            ap.error(f"--arm 은 이름=경로 형식이다: {spec!r}")
        name, path = spec.split("=", 1)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # measure_relation_quality 는 {snapshot, key_to_id, missing_keys} 로 감싸 저장한다.
        snaps[name] = load_snapshot(payload.get("snapshot", payload))

    db = PostgresUtil()
    with db:
        vs, arm_pairs = judge_snapshots(db, snaps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dump_verdicts(vs), ensure_ascii=False, indent=1),
                              encoding="utf-8")

    err = error_rate(vs)
    counts = verdict_counts(vs)
    print(f"\n저장: {args.out}")
    print(f"합집합 판정: {counts}  err율 {100*err:.1f}%")

    stats = compare_arms(snaps, vs, arm_pairs)
    print(f"\n{'팔':6} {'제안':>5} {'쌍':>5} {'판정':>5} {'strong':>7} {'weak':>5} "
          f"{'none':>5} {'자산':>5}  kind 분포")
    for arm, s in stats.items():
        print(f"{arm:6} {s['edge_count']:5} {s['pair_count']:5} {s['rated']:5} "
              f"{s['strong_count']:7} {s['weak_count']:5} {s['none_count']:5} "
              f"{s['assets_with_edge']:5}  {s['kind_dist']}")

    if err > 0.05:
        print(f"\n🔴 판정 실패율 {100*err:.1f}% > 5% — 이 측정은 무효다.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
