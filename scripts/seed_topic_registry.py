"""topic 정본 레지스트리 시드 스크립트 (spec 058 G5 · T501).

목적
    현 코퍼스의 자유기입 topic 라벨(distinct ≈120)을 **LLM 1회 동의어 클러스터링**으로 묶어
    정본(canonical) topic + alias 초안을 산출한다(요리/음식/식품→요리 등). 산출물은 **사람 검수용
    초안**이며(T502 사람 게이트), 검수 후 ``--apply`` 로 ``topic_registry``/``topic_alias`` 에 적재한다.

두 모드
    ``--dry-run``(기본): **DB 쓰기 0**. graph_edge(active·topic_ko 비지 않음)에서 topic_ko 빈도 +
        관측 topic_en 변형을 집계 → LLM(``complete_json``·temp=0) 1회로 동의어 그룹핑 → 후처리 검증
        (누락/중복/정본존재) → 초안 JSON 저장 + 콘솔 요약. LLM 실호출.
    ``--apply <검수파일.json>``: 검수된 초안을 적재. 그룹별 ``register_topic``(임베딩 계산·멱등) +
        각 member 에 ``topic_alias`` INSERT(``decided_by='seed'``·ON CONFLICT DO NOTHING). 적재 수 리포트.

헌법·불변식
    - **LLM 단일 seam**(2조): ``src.llm.client.complete_json`` 만 사용·temp=0·zero-shot(1회 시드).
    - **결정성**(3조): 클러스터링은 1회 시드→검수 후 동결. 입력 정렬·그룹 정렬에 타이브레이커를 고정해
      같은 입력에 같은 산출(빈도 desc→라벨 asc). LLM 판정 결과는 검수본으로 재현.
    - **학습 0**(2조): 임베딩은 등록 시 추론만(``register_topic``). 여기 dry-run 은 임베딩도 계산 안 함.
    - dry-run 은 조회만(``topic_query`` 의 graph_edge topic jsonb 관례 재사용·조회행 str() 강제).

실행(T501)
    conda activate AuroraFS
    python -m scripts.seed_topic_registry --env dev --dry-run
      → specs/058-relation-topic-canonicalization/seed_topic_draft.json + 콘솔 요약.
    ※ ``--apply`` 는 사람 검수(T502) 후에만. T501 에서는 실행하지 않는다.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

# 초안 기본 저장 경로(repo 내·사람 검수용). spec 058 디렉터리에 둔다.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DRAFT_PATH = (
    _REPO_ROOT
    / "specs"
    / "058-relation-topic-canonicalization"
    / "seed_topic_draft.json"
)

# LLM 프롬프트에 노출할 topic_en 변형 상한(프롬프트 비대화 방지·관측 상위만).
_MAX_EN_VARIANTS_SHOWN = 4
# canonical_en 미상 시 폴백(register_topic 의 topic_en NOT NULL 관례와 동형).
_DEFAULT_EN = "general"


# ────────────────────────────────────────────────────────────────────────────
# 1) 조회: graph_edge active 엣지의 topic_ko 빈도 + topic_en 변형 집계
# ────────────────────────────────────────────────────────────────────────────
# topic jsonb 접근 관례는 src/relations/topic_query.py 와 동일:
#   ge.topic->>'topic_ko' / ge.topic->>'topic_en', status='active' 리터럴, 빈 topic_ko 배제.
_TOPIC_STATS_SQL = """
SELECT ge.topic->>'topic_ko' AS topic_ko,
       ge.topic->>'topic_en' AS topic_en,
       COUNT(*) AS freq
FROM graph_edge ge
WHERE ge.status = 'active'
  AND COALESCE(ge.topic->>'topic_ko', '') <> ''
GROUP BY ge.topic->>'topic_ko', ge.topic->>'topic_en'
ORDER BY topic_ko ASC, freq DESC
"""


def fetch_topic_stats(conn) -> tuple[dict[str, int], dict[str, dict[str, int]], int]:
    """active 엣지에서 (topic_ko 빈도, topic_ko별 topic_en 변형 빈도, 총 엣지 수) 집계.

    Returns:
        freq_by_ko: {topic_ko: 총빈도}
        en_variants: {topic_ko: {topic_en: 빈도}}  (빈/None en 은 제외)
        n_edges: topic 보유 active 엣지 총수(= 모든 freq 합)
    조회행 계약(graph_query 관례): topic_ko/topic_en 은 ``str()`` 로 강제.
    """
    freq_by_ko: dict[str, int] = defaultdict(int)
    en_variants: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_edges = 0
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_TOPIC_STATS_SQL)
        for r in cur.fetchall():
            ko = str(r["topic_ko"])
            freq = int(r["freq"])
            freq_by_ko[ko] += freq
            n_edges += freq
            en = r["topic_en"]
            if en is not None and str(en).strip():
                en_variants[ko][str(en)] += freq
    # 일반 dict 로 고정(defaultdict 누수 방지·직렬화 안전)
    return (
        dict(freq_by_ko),
        {ko: dict(v) for ko, v in en_variants.items()},
        n_edges,
    )


def _best_en(en_freq: dict[str, int] | None) -> str:
    """topic_en 변형 중 대표 1개 — 빈도 desc → 알파벳 asc(결정적). 없으면 폴백."""
    if not en_freq:
        return _DEFAULT_EN
    return sorted(en_freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _en_variants_sorted(en_freq: dict[str, int] | None) -> list[str]:
    """topic_en 변형을 빈도 desc → 알파벳 asc 로 정렬한 라벨 목록(프롬프트 표시용)."""
    if not en_freq:
        return []
    return [en for en, _ in sorted(en_freq.items(), key=lambda kv: (-kv[1], kv[0]))]


def build_topic_input(
    freq_by_ko: dict[str, int], en_variants: dict[str, dict[str, int]]
) -> list[dict[str, Any]]:
    """LLM/검증에 쓸 입력 목록 — 빈도 desc → topic_ko asc(결정적 정렬).

    각 항목: {topic_ko, freq, en_variants:[...]}. en_variants 는 관측 상위만.
    """
    items = sorted(freq_by_ko.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[dict[str, Any]] = []
    for ko, freq in items:
        out.append(
            {
                "topic_ko": ko,
                "freq": freq,
                "en_variants": _en_variants_sorted(en_variants.get(ko)),
            }
        )
    return out


# ────────────────────────────────────────────────────────────────────────────
# 2) LLM 동의어 클러스터링(단일 seam·temp=0·zero-shot 1회)
# ────────────────────────────────────────────────────────────────────────────
_CLUSTER_PROMPT_HEADER = """너는 한국어 주제(topic) 라벨을 동의어/같은 개념끼리 묶어 정본 어휘로 수렴시키는 클러스터링기다.

아래는 코퍼스(관계 그래프)에서 관측된 topic 라벨 목록이다. 각 줄은 "한국어 라벨 (freq=빈도) | en: 영문변형들" 이다.

작업: 위 topic_ko 들을 동의어/같은 개념끼리 그룹으로 묶어라.

규칙(엄수):
- canonical_ko = 각 그룹의 대표 라벨. 그 그룹에서 **가장 흔하고 가장 일반적인** 라벨을 고른다. 반드시 members 중 하나여야 한다.
- members = 그 정본으로 묶일 코퍼스 내 동의어 전부(**정본 자신 포함**).
- 입력 topic_ko 는 **정확히 한 그룹에만** 속한다(누락·중복 금지). 모든 입력 라벨이 어딘가에 정확히 1번 나와야 한다.
- **진짜 동의어/같은 개념만 병합**한다. 관련은 있지만 다른 개념이면 각자 별도 그룹으로 둔다(**과병합 금지**). 동의어가 없으면 그 라벨은 members 가 자기 자신 1개인 단독 그룹이다.
- canonical_en = 그룹 대표 영문 1개(관측된 en 변형 중 대표).
- members 는 반드시 위 입력 topic_ko 목록에 있는 라벨만 쓴다(새 라벨 창작 금지).

출력: JSON 객체 하나만. 코드블록·설명 문장 금지.
형식: {"groups": [{"canonical_ko": "...", "canonical_en": "...", "members": ["...", ...]}, ...]}

topic 목록:
"""


def build_clustering_prompt(topic_input: list[dict[str, Any]]) -> str:
    """입력 목록 → LLM 클러스터링 프롬프트(후보 목록을 한 줄씩 나열)."""
    lines = [_CLUSTER_PROMPT_HEADER]
    for it in topic_input:
        en = ", ".join(it["en_variants"][:_MAX_EN_VARIANTS_SHOWN]) or "-"
        lines.append(f"- {it['topic_ko']} (freq={it['freq']}) | en: {en}")
    return "\n".join(lines)


def parse_groups(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """LLM 응답 dict → 그룹 목록(형태 방어). 비정상 항목은 스킵.

    각 그룹: {canonical_ko:str, canonical_en:str, members:[str,...]}. members 는 문자열만 남긴다.
    """
    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, list):
        return []
    out: list[dict[str, Any]] = []
    for g in groups_raw:
        if not isinstance(g, dict):
            continue
        canonical_ko = g.get("canonical_ko")
        members = g.get("members")
        if not isinstance(canonical_ko, str) or not canonical_ko.strip():
            continue
        if not isinstance(members, list):
            continue
        clean_members = [str(m).strip() for m in members if str(m).strip()]
        canonical_en = g.get("canonical_en")
        out.append(
            {
                "canonical_ko": canonical_ko.strip(),
                "canonical_en": (
                    str(canonical_en).strip()
                    if isinstance(canonical_en, str) and canonical_en.strip()
                    else ""
                ),
                "members": clean_members,
            }
        )
    return out


# ────────────────────────────────────────────────────────────────────────────
# 3) 후처리 검증 + 정규화(결정적) — 누락/중복/정본존재 리포트, 초안을 적재가능 형태로
# ────────────────────────────────────────────────────────────────────────────
def validate_and_normalize_groups(
    groups: list[dict[str, Any]],
    freq_by_ko: dict[str, int],
    en_variants: dict[str, dict[str, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """LLM 그룹을 검증·정규화해 **적재가능·완전한** 그룹 목록 + 검증 리포트를 만든다.

    정규화(결정적):
      1. 입력에 없는 member(창작) 제거 → ``members_not_in_input`` 리포트.
      2. 여러 그룹에 걸친 중복 member 는 **첫 그룹만 유지**(그룹 순서 결정적) → ``duplicated_members`` 리포트.
      3. canonical_ko 가 members 에 없으면(입력엔 있음) members 앞에 추가 → ``canonical_not_in_members``.
         canonical_ko 가 입력 자체에 없으면 → ``canonical_not_in_input`` 리포트(사람 검수 대상·그대로 보존).
      4. member 가 하나도 안 남은 그룹은 드롭.
      5. LLM 이 누락한 입력 topic 은 **단독 그룹**(canonical=자신·en=관측 대표)으로 보강 → ``missing_topics``.
      6. total_freq = members 빈도 합. 정렬 = 병합그룹(members>1) 우선 → total_freq desc → canonical_ko asc.
    """
    input_set = set(freq_by_ko)
    seen: set[str] = set()
    members_not_in_input: list[str] = []
    duplicated_members: list[str] = []
    canonical_not_in_members: list[str] = []
    canonical_not_in_input: list[str] = []

    normalized: list[dict[str, Any]] = []
    for g in groups:
        canonical_ko = g["canonical_ko"]
        kept: list[str] = []
        for m in g["members"]:
            if m not in input_set:
                members_not_in_input.append(m)
                continue
            if m in seen:
                duplicated_members.append(m)
                continue
            seen.add(m)
            kept.append(m)

        # canonical 이 members 에 없으면(입력엔 존재) 보강
        if canonical_ko not in kept:
            if canonical_ko in input_set:
                if canonical_ko not in seen:
                    seen.add(canonical_ko)
                    kept.insert(0, canonical_ko)
                    canonical_not_in_members.append(canonical_ko)
                # canonical 이 이미 다른 그룹에 흡수됐다면 여기선 못 넣음(중복 방지) → 검수 대상
                else:
                    canonical_not_in_members.append(canonical_ko)
            else:
                canonical_not_in_input.append(canonical_ko)

        if not kept:
            continue  # 남은 member 없음 → 드롭

        canonical_en = g["canonical_en"] or _best_en(en_variants.get(canonical_ko))
        total_freq = sum(freq_by_ko.get(m, 0) for m in kept)
        normalized.append(
            {
                "canonical_ko": canonical_ko,
                "canonical_en": canonical_en,
                "members": sorted(kept, key=lambda m: (-freq_by_ko.get(m, 0), m)),
                "total_freq": total_freq,
            }
        )

    # LLM 이 누락한 입력 topic → 단독 그룹 보강(초안을 완전하게)
    missing_topics = sorted(input_set - seen, key=lambda m: (-freq_by_ko.get(m, 0), m))
    for m in missing_topics:
        normalized.append(
            {
                "canonical_ko": m,
                "canonical_en": _best_en(en_variants.get(m)),
                "members": [m],
                "total_freq": freq_by_ko.get(m, 0),
            }
        )

    # 정렬: 병합그룹(members>1) 우선 → total_freq desc → canonical_ko asc(결정적)
    normalized.sort(
        key=lambda g: (0 if len(g["members"]) > 1 else 1, -g["total_freq"], g["canonical_ko"])
    )

    report = {
        "n_input": len(input_set),
        "n_groups": len(normalized),
        "n_merge_groups": sum(1 for g in normalized if len(g["members"]) > 1),
        "n_singleton_groups": sum(1 for g in normalized if len(g["members"]) == 1),
        "missing_topics": missing_topics,
        "duplicated_members": sorted(set(duplicated_members)),
        "members_not_in_input": sorted(set(members_not_in_input)),
        "canonical_not_in_members": sorted(set(canonical_not_in_members)),
        "canonical_not_in_input": sorted(set(canonical_not_in_input)),
        # 커버리지 불변식: 모든 입력이 정확히 1개 그룹에 속함(단독 보강 후 항상 True 여야)
        "coverage_complete": len(seen | set(missing_topics)) == len(input_set),
    }
    return normalized, report


def build_draft(
    normalized_groups: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    n_topics: int,
    n_edges: int,
) -> dict[str, Any]:
    """검수용 초안 JSON 구조. spec 지정 구조 + 검증 리포트(``_validation``·검수 보조)."""
    return {
        "generated_from": {"n_topics": n_topics, "n_edges": n_edges},
        "groups": normalized_groups,
        "_validation": report,
        "_NOTE": (
            "LLM 동의어 클러스터링 초안 — 사람 검수 필수(T502). 오병합 교정·정본 확정 후 "
            "`--apply <이 파일>` 로 topic_registry/topic_alias 에 적재. members>1 이 실제 병합."
        ),
    }


# ────────────────────────────────────────────────────────────────────────────
# 4) 콘솔 요약
# ────────────────────────────────────────────────────────────────────────────
def summarize_lines(draft: dict[str, Any]) -> list[str]:
    """콘솔 요약 줄 목록 — N→M·병합 그룹 목록(canonical ← members·빈도)·단독 수."""
    gen = draft["generated_from"]
    groups = draft["groups"]
    rep = draft["_validation"]
    n_merge = rep["n_merge_groups"]
    n_singleton = rep["n_singleton_groups"]
    lines = [
        f"[시드 초안] {gen['n_topics']} topic (active 엣지 {gen['n_edges']}) "
        f"→ {len(groups)} 정본 (병합 {n_merge} · 단독 {n_singleton})",
        "",
        f"── 병합 그룹({n_merge}) — canonical(총빈도) ← members ──",
    ]
    for g in groups:
        if len(g["members"]) <= 1:
            continue
        members_str = ", ".join(
            f"{m}" for m in g["members"] if m != g["canonical_ko"]
        )
        lines.append(
            f"  {g['canonical_ko']}({g['total_freq']}) [{g['canonical_en']}] ← {members_str}"
        )
    # 검증 경고
    warn = []
    if rep["missing_topics"]:
        warn.append(f"누락(단독보강) {len(rep['missing_topics'])}")
    if rep["duplicated_members"]:
        warn.append(f"중복member(첫그룹유지) {len(rep['duplicated_members'])}")
    if rep["members_not_in_input"]:
        warn.append(f"입력밖member(제거) {len(rep['members_not_in_input'])}")
    if rep["canonical_not_in_input"]:
        warn.append(f"정본이입력에없음 {len(rep['canonical_not_in_input'])}")
    lines.append("")
    lines.append("검증: " + ("정상(커버리지 완전)" if rep["coverage_complete"] else "커버리지 불완전!"))
    if warn:
        lines.append("경고: " + " · ".join(warn))
    return lines


# ────────────────────────────────────────────────────────────────────────────
# 5) dry-run / apply
# ────────────────────────────────────────────────────────────────────────────
def run_dry_run(db, out_path: Path, *, client=None) -> dict[str, Any]:
    """조회 → LLM 클러스터링 1회 → 검증·정규화 → 초안 저장. **DB 쓰기 0**.

    ``client`` 주입 시 그 LLM 클라이언트로(테스트·재현용), 미주입이면 운영 온프레미스 seam.
    """
    from src.llm.client import complete_json

    with db.transaction() as conn:
        freq_by_ko, en_variants, n_edges = fetch_topic_stats(conn)

    topic_input = build_topic_input(freq_by_ko, en_variants)
    prompt = build_clustering_prompt(topic_input)
    raw = complete_json(prompt, client=client)  # ★ LLM 1회(temp=0)
    groups = parse_groups(raw)
    normalized, report = validate_and_normalize_groups(groups, freq_by_ko, en_variants)
    draft = build_draft(
        normalized, report, n_topics=len(freq_by_ko), n_edges=n_edges
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(draft, f, ensure_ascii=False, indent=2)
    return draft


def run_apply(db, review_path: Path) -> dict[str, int]:
    """검수된 초안 적재 — 그룹별 register_topic + member alias INSERT(decided_by='seed').

    **T502 사람 게이트 이후에만 실행**. register_topic 은 임베딩 계산·멱등, alias 는 ON CONFLICT DO NOTHING.
    """
    from src.relations.topic_canonicalize import register_topic

    with open(review_path, encoding="utf-8") as f:
        payload = json.load(f)
    groups = payload.get("groups", [])

    n_registered = 0
    n_alias = 0
    with db.transaction() as conn:
        for g in groups:
            canonical_ko = str(g["canonical_ko"])
            canonical_en = str(g.get("canonical_en") or _DEFAULT_EN)
            # 정본 등록(임베딩 계산·멱등). 시드 출처 표식.
            register_topic(conn, canonical_ko, canonical_en, source="seed")
            n_registered += 1
            for m in g.get("members", []):
                # 각 member(정본 자신 포함) → alias 동결(decided_by='seed'·멱등).
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO topic_alias (raw_ko, canonical_ko, decided_by)
                        VALUES (%s, %s, 'seed')
                        ON CONFLICT (raw_ko) DO NOTHING
                        """,
                        (str(m), canonical_ko),
                    )
                    n_alias += cur.rowcount
    return {"registered": n_registered, "alias_inserted": n_alias}


def main() -> int:
    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    p = argparse.ArgumentParser(description="topic 정본 레지스트리 시드(spec 058 T501)")
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="DB 쓰기 0 — LLM 클러스터링 초안 산출(기본)"
    )
    mode.add_argument(
        "--apply", metavar="검수파일.json", help="검수된 초안 적재(T502 사람 게이트 이후)"
    )
    p.add_argument(
        "--out", default=str(_DEFAULT_DRAFT_PATH), help="초안 저장 경로(dry-run)"
    )
    args = p.parse_args()

    dotenv_path = _REPO_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    db = PostgresUtil()
    with db:
        if args.apply:
            out = run_apply(db, Path(args.apply))
            print(json.dumps(out, ensure_ascii=False))
            return 0
        # 기본 = dry-run
        draft = run_dry_run(db, Path(args.out))
    print("\n".join(summarize_lines(draft)))
    print(f"\n초안 저장: {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
