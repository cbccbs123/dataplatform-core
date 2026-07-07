"""기존 active ``graph_edge.topic`` 을 시드된 정본으로 **결정적 재작성**(spec 058 G6 · T601~T605).

목적 (FR-501/502/503 · SC-01/03/04/06/07)
    G5 에서 사람 검수·적재된 정본 레지스트리(``topic_registry``)·해소 캐시(``topic_alias``)를
    이용해, 이미 저장된 active 엣지의 ``topic`` jsonb 를 정본으로 일괄 재작성한다. 생성시 배선
    (G4·``graph_persist``)이 앞으로의 데이터를 정규화한다면, 백필은 **현 2,775 엣지**를 같은 규칙으로
    소급 정리한다("생성시 = 백필 = 수집" 일치).

결정성·헌법 (LLM 0)
    재작성은 **alias 룩업(정확일치) + canonicalize_subtopic 규칙(모달리티/계층)** 만 쓴다 —
    kNN 도 judge(LLM) 도 register(레지스트리 변형) 도 **호출하지 않는다**(G5 시드가 이미 커버).
      · topic_ko  → ``lookup_alias``(정본). alias 미스(정본 없는 topic)면 **원본 유지**(비파괴)·리포트.
      · topic_en  → ``_lookup_topic_en``(정본 영문). None 이면 기존 topic_en 보존(빈 라벨 방지).
      · subtopic  → ``canonicalize_subtopic``(정본_topic_ko, raw_sub): 모달리티어/계층(정본 topic) → 비움.
    같은 입력 → 같은 결과(헌법 3조). LLM/임베딩 실호출 0.

세 모드
    ``--dry-run``(기본): 재작성 **계산만**(쓰기 0). 변경 엣지 수 · SC-07(distinct topic 120→N) ·
        SC-03 미리보기(재작성 후 topic∩subtopic 라벨 수) · SC-04(subtopic 모달리티 잔존) ·
        alias 미스 목록을 리포트.
    ``--apply``: ① **백업 먼저**(active 엣지의 ``edge_id, topic`` 원본을 복원 가능하게 백업 테이블
        ``graph_edge_topic_bak_058`` 에 덤프) → ② 변경 엣지의 ``topic`` jsonb 배치 UPDATE → ③ 커밋.
        이미 백업이 있으면(재실행) 클로버 방지로 중단(먼저 ``--restore`` 하거나 백업 테이블 삭제).
    ``--restore``: 백업 테이블에서 ``topic`` 원복(골든 회귀 시 되돌리기). 복원 수 리포트.

주의
    - **dev 만**. 프로덕션 백필·OS 재색인은 별도 사람 게이트(plan G6·🔴).
    - ``TOPIC_CANONICALIZE_ENABLED`` 플래그와 **무관** — 백필은 플래그 게이트를 거치지 않고 직접 재작성한다.
    - OS 재색인(T603)은 별도: ``python -m src.app.run_opensearch_resync --env dev`` (topics/subtopics
      keyword 재투영·056). 재작성 → 재색인 순서.

실행
    conda activate AuroraFS
    python scripts/backfill_topic_canonical.py --env dev --dry-run   # 계산만(기본)
    python scripts/backfill_topic_canonical.py --env dev --apply     # 백업 후 재작성·커밋
    python scripts/backfill_topic_canonical.py --env dev --restore   # 백업에서 원복
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from psycopg.rows import dict_row

_REPO_ROOT = Path(__file__).resolve().parents[1]
# 직접 실행(python scripts/...) 시 repo 루트를 경로에 올려 src 패키지 import 보장(measure_* 러너 동형).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 모달리티 블랙리스트 단일 출처(SC-04 판정) — topic_canonicalize 와 공유(중복 정의 금지).
from src.relations.topic_canonicalize import _MODALITY_BLACKLIST  # noqa: E402

# 백업 테이블(058 전용) — active 엣지 원본 topic 을 복원 가능하게 보관. --restore 가 여기서 되돌린다.
_BACKUP_TABLE = "graph_edge_topic_bak_058"

# topic jsonb 4필드(graph_persist 저장 계약과 동일 형태).
_TOPIC_KEYS = ("topic_ko", "subtopic_ko", "topic_en", "subtopic_en")

# resolve_topic_fn: (topic_ko, topic_en) -> (canonical_ko, canonical_en, alias_miss)
ResolveTopicFn = Callable[[str, str], tuple[str, str | None, bool]]
# resolve_subtopic_fn: (canonical_ko, subtopic_ko) -> 정본 subtopic 라벨 or None(비움)
ResolveSubtopicFn = Callable[[str, str], str | None]


# ────────────────────────────────────────────────────────────────────────────
# 1) 순수 재작성 계산 (결정적·주입 seam·단위 테스트)
# ────────────────────────────────────────────────────────────────────────────
def rewrite_topic_row(
    old: dict[str, Any],
    resolve_topic_fn: ResolveTopicFn,
    resolve_subtopic_fn: ResolveSubtopicFn,
) -> tuple[dict[str, str], bool, dict[str, bool]]:
    """엣지 하나의 topic jsonb → 정본 재작성(순수·주입 seam). 반환 ``(new, changed, flags)``.

    - ``resolve_topic_fn``/``resolve_subtopic_fn`` 은 해소 seam(DB 경로=alias 룩업+canonicalize_subtopic,
      테스트=가짜). 이 함수는 그 결과를 조립만 하므로 **주입에 대해 순수·결정적**이다.
    - graph_persist 배선과 **동형** 규칙: subtopic 이 비면(None) subtopic_ko/en 둘 다 ``""`` 로 비운다.
    - alias 미스면 topic 원본 유지(비파괴). topic_en 은 정본 en 없으면 기존값 보존(빈 라벨 방지).
    """
    topic_ko = str(old.get("topic_ko") or "")
    subtopic_ko = str(old.get("subtopic_ko") or "")
    topic_en = str(old.get("topic_en") or "")
    subtopic_en = str(old.get("subtopic_en") or "")

    canonical_ko, canonical_en, alias_miss = resolve_topic_fn(topic_ko, topic_en)
    new_topic_ko = canonical_ko if canonical_ko else topic_ko
    new_topic_en = str(canonical_en) if canonical_en else topic_en

    new_sub = resolve_subtopic_fn(new_topic_ko, subtopic_ko)
    if new_sub is None:
        # 모달리티어/계층 규칙으로 비운 경우 en 도 함께 비운다(계층 일관·FR-301/302·graph_persist 동형).
        new_subtopic_ko, new_subtopic_en = "", ""
    else:
        new_subtopic_ko = new_sub
        new_subtopic_en = subtopic_en  # subtopic_en 정본화(영문)는 후속 여지(1차는 ko 라벨만)

    new = {
        "topic_ko": new_topic_ko,
        "subtopic_ko": new_subtopic_ko,
        "topic_en": new_topic_en,
        "subtopic_en": new_subtopic_en,
    }
    changed = any(new[k] != str(old.get(k) or "") for k in _TOPIC_KEYS)
    flags = {
        "topic_changed": new_topic_ko != topic_ko,
        "topic_en_changed": new_topic_en != topic_en,
        "subtopic_cleared": subtopic_ko != "" and new_subtopic_ko == "",
        "subtopic_changed": new_subtopic_ko != subtopic_ko,
        "alias_miss": alias_miss,
    }
    return new, changed, flags


def build_plan(
    rows: list[dict[str, Any]],
    resolve_topic_fn: ResolveTopicFn,
    resolve_subtopic_fn: ResolveSubtopicFn,
) -> list[dict[str, Any]]:
    """active 엣지 행 목록 → 재작성 계획(순수·결정적). 각 항목 ``{edge_id, old, new, changed, flags}``.

    ``rows`` 각 항목은 ``{edge_id, topic_ko, subtopic_ko, topic_en, subtopic_en}``(topic jsonb 평탄화).
    """
    plan: list[dict[str, Any]] = []
    for r in rows:
        old = {k: r.get(k) for k in _TOPIC_KEYS}
        new, changed, flags = rewrite_topic_row(old, resolve_topic_fn, resolve_subtopic_fn)
        plan.append(
            {
                "edge_id": r.get("edge_id"),
                "old": {k: str(old.get(k) or "") for k in _TOPIC_KEYS},
                "new": new,
                "changed": changed,
                "flags": flags,
            }
        )
    return plan


# ────────────────────────────────────────────────────────────────────────────
# 2) SC 판정 (순수·단위 테스트) — 재작성 후 라벨 집합 기준
# ────────────────────────────────────────────────────────────────────────────
def _distinct_labels(dicts: list[dict[str, str]], key: str) -> set[str]:
    """비어있지 않은 라벨 distinct 집합(순수)."""
    return {str(d.get(key) or "") for d in dicts if str(d.get(key) or "").strip()}


def sc07_distinct_topics(dicts: list[dict[str, str]]) -> int:
    """SC-07: distinct topic_ko 수(범위 축소 측정)."""
    return len(_distinct_labels(dicts, "topic_ko"))


def sc03_topic_subtopic_overlap(dicts: list[dict[str, str]]) -> list[str]:
    """SC-03: topic 이자 subtopic 인 라벨(계층 불일치). 재작성 후 0 이어야 한다."""
    topics = _distinct_labels(dicts, "topic_ko")
    subs = _distinct_labels(dicts, "subtopic_ko")
    return sorted(topics & subs)


def sc04_modality_subtopics(dicts: list[dict[str, str]]) -> list[str]:
    """SC-04: subtopic 이 매체어(텍스트/오디오/영상/이미지+en)인 라벨. 재작성 후 0 이어야 한다."""
    subs = _distinct_labels(dicts, "subtopic_ko")
    return sorted(s for s in subs if s.lower() in _MODALITY_BLACKLIST)


def summarize_plan(plan: list[dict[str, Any]]) -> dict[str, Any]:
    """재작성 계획 → 리포트 dict(순수). 전/후 SC 지표·변경 통계·alias 미스."""
    olds = [p["old"] for p in plan]
    news = [p["new"] for p in plan]
    changed = [p for p in plan if p["changed"]]
    alias_miss = sorted(
        {p["old"]["topic_ko"] for p in plan if p["flags"]["alias_miss"] and p["old"]["topic_ko"]}
    )
    return {
        "n_edges": len(plan),
        "n_changed": len(changed),
        "n_topic_changed": sum(1 for p in plan if p["flags"]["topic_changed"]),
        "n_topic_en_changed": sum(1 for p in plan if p["flags"]["topic_en_changed"]),
        "n_subtopic_cleared": sum(1 for p in plan if p["flags"]["subtopic_cleared"]),
        "n_subtopic_changed": sum(1 for p in plan if p["flags"]["subtopic_changed"]),
        "sc07_before": sc07_distinct_topics(olds),
        "sc07_after": sc07_distinct_topics(news),
        "sc03_before": sc03_topic_subtopic_overlap(olds),
        "sc03_after": sc03_topic_subtopic_overlap(news),
        "sc04_before": sc04_modality_subtopics(olds),
        "sc04_after": sc04_modality_subtopics(news),
        "alias_miss": alias_miss,
    }


def format_report_lines(report: dict[str, Any], *, mode: str) -> list[str]:
    """리포트 dict → 콘솔 줄(순수·사람 검수용)."""
    lines = [
        f"[백필 topic 정규화 · {mode}] active 엣지 {report['n_edges']}건",
        f"  변경 엣지: {report['n_changed']}건 "
        f"(topic_ko {report['n_topic_changed']} · topic_en {report['n_topic_en_changed']} · "
        f"subtopic 변경 {report['n_subtopic_changed']}[비움 {report['n_subtopic_cleared']}])",
        f"  SC-07 distinct topic: {report['sc07_before']} → {report['sc07_after']}",
        f"  SC-03 topic∩subtopic 라벨: {len(report['sc03_before'])} → {len(report['sc03_after'])}"
        + (f"  ⚠ 잔존: {report['sc03_after']}" if report["sc03_after"] else "  (0·계층 일관)"),
        f"  SC-04 subtopic 모달리티: {len(report['sc04_before'])} → {len(report['sc04_after'])}"
        + (f"  ⚠ 잔존: {report['sc04_after']}" if report["sc04_after"] else "  (0·모달리티 정리)"),
        f"  alias 미스(정본 없는 topic·원본 유지): {len(report['alias_miss'])}"
        + (f" → {report['alias_miss']}" if report["alias_miss"] else " (없음·전 topic 커버)"),
    ]
    return lines


# ────────────────────────────────────────────────────────────────────────────
# 3) DB 경로 — 해소 seam 결선(alias 룩업·canonicalize_subtopic·LLM 0) · 백업 · 재작성 · 복원
# ────────────────────────────────────────────────────────────────────────────
_ACTIVE_EDGES_SQL = """
SELECT edge_id,
       topic->>'topic_ko'    AS topic_ko,
       topic->>'subtopic_ko' AS subtopic_ko,
       topic->>'topic_en'    AS topic_en,
       topic->>'subtopic_en' AS subtopic_en
FROM graph_edge
WHERE status = 'active'
ORDER BY edge_id
"""


def fetch_active_edges(conn) -> list[dict[str, Any]]:
    """active 엣지의 (edge_id + topic 4필드) 평탄화 목록(결정적 정렬·edge_id str 강제)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_ACTIVE_EDGES_SQL)
        rows = cur.fetchall()
    return [
        {
            "edge_id": str(r["edge_id"]),
            "topic_ko": r["topic_ko"],
            "subtopic_ko": r["subtopic_ko"],
            "topic_en": r["topic_en"],
            "subtopic_en": r["subtopic_en"],
        }
        for r in rows
    ]


def make_db_resolvers(conn) -> tuple[ResolveTopicFn, ResolveSubtopicFn]:
    """DB 해소 seam 결선(메모이즈) — ``lookup_alias``·``_lookup_topic_en``·``canonicalize_subtopic``.

    LLM/kNN/register 는 **부르지 않는다**(백필 규율): topic 은 alias 정확일치만(미스=원본 유지),
    subtopic 은 canonicalize_subtopic(모달리티/계층·순수 룩업). distinct 라벨 단위 메모이즈로 쿼리 절감.
    """
    from src.relations.topic_canonicalize import (
        _lookup_topic_en,
        canonicalize_subtopic,
        lookup_alias,
    )

    topic_cache: dict[tuple[str, str], tuple[str, str | None, bool]] = {}
    sub_cache: dict[tuple[str, str], str | None] = {}

    def resolve_topic(topic_ko: str, topic_en: str) -> tuple[str, str | None, bool]:
        key = (topic_ko, topic_en)
        if key in topic_cache:
            return topic_cache[key]
        if not topic_ko or not topic_ko.strip():
            res = (topic_ko, topic_en or None, False)  # 빈 topic → 그대로(미스 아님)
        else:
            hit = lookup_alias(conn, topic_ko)  # 정확일치(캐시)·LLM 0
            alias_miss = hit is None
            canonical_ko = hit if hit is not None else topic_ko
            canonical_en = _lookup_topic_en(conn, canonical_ko) or topic_en
            res = (canonical_ko, canonical_en, alias_miss)
        topic_cache[key] = res
        return res

    def resolve_subtopic(canonical_ko: str, subtopic_ko: str) -> str | None:
        key = (canonical_ko, subtopic_ko)
        if key in sub_cache:
            return sub_cache[key]
        # canonicalize_subtopic 은 모달리티/계층 판정에 registry/alias 순수 룩업만 쓴다(LLM 0).
        res = canonicalize_subtopic(conn, canonical_ko, subtopic_ko)
        sub_cache[key] = res
        return res

    return resolve_topic, resolve_subtopic


def compute_plan(conn) -> list[dict[str, Any]]:
    """DB 에서 active 엣지 읽고 재작성 계획 산출(seam 결선 + 순수 build_plan)."""
    rows = fetch_active_edges(conn)
    rt, rs = make_db_resolvers(conn)
    return build_plan(rows, rt, rs)


def backup_row_count(conn) -> int | None:
    """백업 테이블 행 수(없으면 None) — --apply 클로버 방지·--restore 대상 확인."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (_BACKUP_TABLE,))
        if cur.fetchone()[0] is None:
            return None
        cur.execute(f"SELECT count(*) FROM {_BACKUP_TABLE}")
        return int(cur.fetchone()[0])


def create_backup(conn) -> int:
    """active 엣지의 (edge_id, topic) 원본을 백업 테이블에 덤프. 백업 행 수 반환.

    복원 가능성 우선: 백업 테이블을 새로 만들고 active 엣지 topic 전부를 그대로 담는다.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE {_BACKUP_TABLE} (
                edge_id uuid PRIMARY KEY,
                topic jsonb NOT NULL,
                backed_up_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            INSERT INTO {_BACKUP_TABLE} (edge_id, topic)
            SELECT edge_id, topic FROM graph_edge WHERE status = 'active'
            """
        )
        cur.execute(f"SELECT count(*) FROM {_BACKUP_TABLE}")
        return int(cur.fetchone()[0])


def apply_rewrite(conn, plan: list[dict[str, Any]]) -> int:
    """변경 엣지의 ``topic`` jsonb 를 정본으로 배치 UPDATE. 재작성 행 수 반환(topic 만 갱신·복원 대칭)."""
    changed = [p for p in plan if p["changed"]]
    if not changed:
        return 0
    params = [
        (json.dumps(p["new"], ensure_ascii=False), p["edge_id"]) for p in changed
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE graph_edge SET topic = %s::jsonb WHERE edge_id = %s",
            params,
        )
    return len(changed)


def restore_from_backup(conn) -> int:
    """백업 테이블에서 ``topic`` 원복(--restore). 복원 행 수 반환."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE graph_edge ge
            SET topic = b.topic
            FROM {_BACKUP_TABLE} b
            WHERE ge.edge_id = b.edge_id
            """
        )
        return cur.rowcount


# ────────────────────────────────────────────────────────────────────────────
# 4) 실행(IO) — 모드별 오케스트레이션
# ────────────────────────────────────────────────────────────────────────────
def run_dry_run(db) -> dict[str, Any]:
    """재작성 계산만(쓰기 0). 리포트 dict 반환."""
    with db, db.connection() as conn:
        plan = compute_plan(conn)
        conn.rollback()  # 읽기 전용 보장(canonicalize_subtopic 은 순수 룩업이나 방어적 롤백)
    return summarize_plan(plan)


def run_apply(db) -> dict[str, Any]:
    """① 백업(클로버 방지) → ② 재작성 → ③ 커밋. 리포트 + 백업/재작성 수 반환."""
    with db, db.connection() as conn:
        existing = backup_row_count(conn)
        if existing is not None:
            raise SystemExit(
                f"백업 테이블 {_BACKUP_TABLE} 이 이미 존재(행 {existing}). 클로버 방지로 중단.\n"
                f"  되돌리려면 --restore, 다시 백필하려면 먼저 백업 테이블을 삭제하라\n"
                f"  (DROP TABLE {_BACKUP_TABLE};)."
            )
        plan = compute_plan(conn)
        report = summarize_plan(plan)
        n_backup = create_backup(conn)
        n_rewrite = apply_rewrite(conn, plan)
        conn.commit()
    report["n_backup"] = n_backup
    report["n_rewrite"] = n_rewrite
    report["backup_table"] = _BACKUP_TABLE
    return report


def run_restore(db) -> int:
    """백업에서 topic 원복·커밋. 복원 수 반환."""
    with db, db.connection() as conn:
        existing = backup_row_count(conn)
        if existing is None:
            raise SystemExit(f"백업 테이블 {_BACKUP_TABLE} 이 없다 — 복원 불가.")
        n = restore_from_backup(conn)
        conn.commit()
    return n


def main() -> int:
    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    p = argparse.ArgumentParser(
        description="graph_edge.topic 백필 — 시드 정본으로 결정적 재작성(spec 058 G6)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="재작성 계산만(기본·쓰기 0)")
    mode.add_argument("--apply", action="store_true", help="백업 후 재작성·커밋")
    mode.add_argument("--restore", action="store_true", help="백업에서 topic 원복")
    args = p.parse_args()

    # 프로덕션 백필은 사람 게이트(plan G6·🔴) — 스크립트에서 실수 차단.
    if args.env == "prod":
        raise SystemExit("프로덕션 백필은 별도 사람 게이트다. 이 스크립트는 dev 만 지원한다.")

    dotenv_path = _REPO_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    db = PostgresUtil()

    if args.apply:
        report = run_apply(db)
        print("\n".join(format_report_lines(report, mode="apply")))
        print(
            f"  백업: {report['backup_table']} 에 {report['n_backup']}행 · "
            f"재작성: {report['n_rewrite']}행 커밋 완료."
        )
        return 0
    if args.restore:
        n = run_restore(db)
        print(f"[복원] 백업 {_BACKUP_TABLE} 에서 topic {n}행 원복·커밋 완료.")
        return 0

    report = run_dry_run(db)
    print("\n".join(format_report_lines(report, mode="dry-run")))
    print("  (dry-run·쓰기 0. 적용하려면 --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
