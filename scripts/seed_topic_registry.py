"""topic 정본 레지스트리 시드 스크립트 — **닫힌 분류체계(taxonomy) 적재** (spec 058 v2 · T902).

v2 개정(2026-07-07·닫힌 분류체계 전환·ADR `2026-07-07-topic-closed-taxonomy-pivot.md`)
    v1 은 현 코퍼스의 자유기입 topic 을 런타임 canonicalize 로 재생(replay)해 정본/alias 초안을
    산출하고 사람이 검수·적재했다. 그러나 "열린 어휘 + 쌍별 유사판정 + 전이 병합"은 안정해가
    없음이 실측으로 확인돼(광역 흡수↔충돌 잔존의 진동), topic 층을 **닫힌 통제어휘 27+미분류**
    (`taxonomy_draft.md` §1 정본 → `taxonomy_seed.json`)로 전환했다. 이 스크립트는 그 시드 파일을
    **그대로** topic_registry 에 적재한다(replay/from-draft 로직 폐기).

무엇을 하나
    `taxonomy_seed.json`(topic 층 닫힌 분류체계 정본·버전 기록)의 각 topic 을
    `topic_registry` 에 **parent_topic=NULL(topic 층)·source='taxonomy'** 로 등록한다. 등록은
    `register_topic` 을 재사용해 라벨 임베딩(st_bge·1536D)을 계산·저장한다(kNN 후보 회수용).
    subtopic 층(부모 스코프·열린 성장)은 이 시드가 다루지 않는다(G10 canonicalize·G12 백필 몫).
    닫힌 분류체계는 쌍별 병합이 없으므로 시드 결과 **alias 는 0**이다.

두 모드
    ``--dry-run``(기본): 파싱·요약만 출력(DB 미접촉).
    ``--apply``: **기존 topic 층 시드 정리 후 재적재** — 닫힌 분류체계는 topic 정본 집합이 통째로
        갈리므로 ``DELETE ... WHERE parent_topic IS NULL``(topic 층만·alias 먼저·읽기 순서) 후
        taxonomy 28행을 적재하고 커밋한다. register_topic 의 ``ON CONFLICT ... DO NOTHING`` 으로
        재적용도 멱등(28행 유지).

재적용 멱등의 범위(🔴 PR #81 code-review)
    멱등·재적재는 **topic 층(parent_topic IS NULL)에 한정**한다. v297 로 자란 **subtopic 층
    (parent_topic IS NOT NULL·백필 성장 레이어·결정성 캐시)은 삭제하지 않고 보존**한다 —
    과거 ``TRUNCATE`` 는 두 층을 모두 날려 governance §4('전역 재빌드 없음')와 상충했다.

헌법·불변식
    - **학습 0(2조)**: 임베딩은 추론만(register_topic·st_bge). 규칙은 결정적.
    - **결정성(3조)**: 시드 파일이 정본 — 같은 파일 → 같은 registry(재현 가능). LLM/kNN 재실행 0.
    - **1536D/pgvector(4조)**: register_topic 이 라벨 임베딩 1536D 저장(0-노름 거부).
    - 조회/적재행 라벨은 ``str()`` 강제(graph_query 관례).

실행
    conda activate AuroraFS
    python -m scripts.seed_topic_registry --env dev --dry-run     # 파싱·요약(DB 미접촉)
    python -m scripts.seed_topic_registry --env dev --apply       # topic 층 스코프 삭제→28행 적재·커밋
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# 시드 정본 경로. taxonomy 시드는 **src/relations 패키지 내부**가 단일 출처다(PR #81 이관·prompt.py
# 와 동일 파일). src-only 패키징(pyproject include=["src*"]) 시에도 런타임 로드가 되도록 specs/ 가
# 아닌 src/ 에 둔다. alias 선시드는 시드 전용(런타임 미참조)이라 specs/ 에 남는다.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC_DIR = _REPO_ROOT / "specs" / "058-relation-topic-canonicalization"
_DEFAULT_SEED_PATH = _REPO_ROOT / "src" / "relations" / "taxonomy_seed.json"
# alias 선시드 정본(§3 커버리지 매핑·raw_ko→canonical). registry 시드 직후 topic 층 alias 로 적재.
_DEFAULT_ALIAS_SEED_PATH = _SPEC_DIR / "taxonomy_alias_seed.json"
# subtopic 시드 정본(spec 068 G2·G3). topic 시드와 동거(src/relations)해 src-only 패키징에서도
# 런타임 로드된다. 각 subtopic 은 부모 topic 스코프(parent_topic=<topic_ko>)로 가산 적재된다.
_DEFAULT_SUBTOPIC_SEED_PATH = _REPO_ROOT / "src" / "relations" / "subtopic_seed.json"


# ────────────────────────────────────────────────────────────────────────────
# 1) 시드 파싱·정본 행 추출 (결정적 순수 함수)
# ────────────────────────────────────────────────────────────────────────────
def load_taxonomy_seed(path: Path | str = _DEFAULT_SEED_PATH) -> dict[str, Any]:
    """taxonomy_seed.json 을 읽어 dict 로 반환(순수 I/O·파싱만). ``{version, topics:[...]}``."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def taxonomy_registry_entries(seed: dict[str, Any]) -> list[dict[str, str]]:
    """시드 → registry 적재 행 ``[{topic_ko, topic_en}]`` (파일 순서 보존·라벨 str() 강제).

    닫힌 분류체계이므로 각 행이 곧 정본 topic(병합·alias 없음). 순서는 taxonomy_draft §1 표 순서.
    """
    return [
        {"topic_ko": str(t["topic_ko"]), "topic_en": str(t["topic_en"])}
        for t in seed["topics"]
    ]


def load_alias_seed(path: Path | str = _DEFAULT_ALIAS_SEED_PATH) -> dict[str, Any]:
    """taxonomy_alias_seed.json 을 읽어 dict 로 반환(순수 I/O·파싱만). ``{version, aliases:[...]}``."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def alias_seed_entries(alias_seed: dict[str, Any]) -> list[dict[str, str]]:
    """alias 시드 → 적재 행 ``[{raw_ko, canonical_ko}]`` (파일 순서 보존·라벨 str() 강제).

    §3 커버리지 매핑(raw 120 → 27+미분류)에서 자기참조(raw==canonical·음악/과학/동물)는 이미 제외됐다
    (닫힌 집합 정확일치가 처리). 각 행은 topic 층(parent NULL) alias 로 동결된다.
    """
    return [
        {"raw_ko": str(a["raw_ko"]), "canonical_ko": str(a["canonical_ko"])}
        for a in alias_seed["aliases"]
    ]


# ────────────────────────────────────────────────────────────────────────────
# 1b) subtopic 시드 파싱·정본 행 추출 (spec 068 G2 · FR-201/204 · 결정적 순수 함수)
# ────────────────────────────────────────────────────────────────────────────
def load_subtopic_seed(path: Path | str = _DEFAULT_SUBTOPIC_SEED_PATH) -> dict[str, Any]:
    """subtopic_seed.json 을 읽어 dict 로 반환(순수 I/O·파싱만). ``{version, source, subtopics:[...]}``.

    topic 시드(``load_taxonomy_seed``)와 대칭. 정본은 src/relations/subtopic_seed.json 단일 출처다.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _valid_parent_topics() -> set[str]:
    """유효 부모 topic 집합 — taxonomy 27 정본(**미분류 제외**).

    subtopic 은 실제 topic 아래에만 달린다. catch-all '미분류' 에는 subtopic 을 두지 않으므로
    부모 검증에서 제외한다(미분류 부모 → ValueError). taxonomy_seed.json 이 단일 출처(정합 보장).
    """
    return {
        e["topic_ko"]
        for e in taxonomy_registry_entries(load_taxonomy_seed())
        if e["topic_ko"] != "미분류"
    }


def subtopic_registry_entries(seed: dict[str, Any]) -> list[dict[str, Any]]:
    """subtopic 시드 → 적재 행 ``[{parent_topic, subtopic_ko, subtopic_en}]`` (파일 순서 보존).

    닫힌 시드이므로 각 행이 곧 정본 subtopic(런타임 성장 없음·재발 불가). 두 가지 기계적 규칙을
    적용한다(FR-204 — 무의미 소분류 원천 차단):
      ① **부모 검증**: parent_topic 이 27 정본(미분류 제외) 밖이면 ``ValueError`` (오탈자·오소속 차단).
      ② **부모명 반복 배제**: subtopic_ko 가 부모 topic 명과 동일하거나 그 부분문자열이면 제외한다
         (예: '음식·요리' 밑 '음식'). 닫힌 목록이라 런타임 재발이 없고, 부모명과 겹치는 소분류는
         변별력이 없으므로 시드 단계에서 걸러낸다.
    라벨은 ``str()`` 강제(graph_query 관례). ``subtopic_en`` 은 None 을 그대로 보존한다(정본
    미확정 여지 — register_topic 이 topic_en=None 허용). str('None') 로 오염시키지 않는다.
    """
    valid_parents = _valid_parent_topics()
    entries: list[dict[str, Any]] = []
    for s in seed["subtopics"]:
        parent = str(s["topic_ko"])
        if parent not in valid_parents:
            raise ValueError(
                f"subtopic 부모 topic 이 27 정본(미분류 제외) 밖: {parent!r} (subtopic={s.get('subtopic_ko')!r})"
            )
        sub_ko = str(s["subtopic_ko"])
        # ② 부모명 반복(동일/부분문자열) 배제 — 부모명이 subtopic 을 포함하는 방향만 검사한다.
        #    (역방향 '부모 ⊂ subtopic' 은 정상 소분류 '동물행동·생태' 등을 오배제하므로 검사 안 함.)
        if sub_ko in parent:
            continue
        sub_en = s.get("subtopic_en")
        entries.append(
            {
                "parent_topic": parent,
                "subtopic_ko": sub_ko,
                "subtopic_en": str(sub_en) if sub_en is not None else None,
            }
        )
    return entries


# ────────────────────────────────────────────────────────────────────────────
# 2) 적재 (register_topic 재사용·parent_topic=NULL·source='taxonomy')
# ────────────────────────────────────────────────────────────────────────────
def apply_taxonomy_seed(
    conn, seed: dict[str, Any], *, register_fn=None
) -> dict[str, int]:
    """시드의 각 topic 을 topic 층(parent_topic=None)으로 registry 등록. 적재 수 ``{n_registry, n_alias}``.

    - 각 topic: ``register_topic(conn, topic_ko, topic_en, source='taxonomy', parent_topic=None)``
      — 라벨 임베딩 계산·부분 유니크 인덱스(parent NULL) ON CONFLICT 로 멱등.
    - alias 는 쓰지 않는다(닫힌 분류체계는 쌍별 병합 없음 → n_alias=0). 커밋은 호출부 책임.
    - ``register_fn`` 주입 가능(단위 테스트용·기본은 topic_canonicalize seam).
    """
    if register_fn is None:
        # 지연 import(모듈 기동 경량 유지). 무거운 임베더는 register_topic 내부에서만 로드.
        from src.relations.topic_canonicalize import register_topic

        register_fn = register_topic

    entries = taxonomy_registry_entries(seed)
    for e in entries:
        register_fn(
            conn, e["topic_ko"], e["topic_en"], source="taxonomy", parent_topic=None
        )
    return {"n_registry": len(entries), "n_alias": 0}


def apply_alias_seed(
    conn, alias_seed: dict[str, Any], *, freeze_fn=None
) -> dict[str, int]:
    """§3 커버리지 매핑을 topic 층 alias(parent NULL·decided_by='seed')로 동결. 적재 수 ``{n_alias}``.

    - 각 alias: ``_freeze_alias(conn, raw_ko, canonical_ko, 'seed', parent_topic=None)`` — 부분 유니크
      ON CONFLICT 로 멱등. **registry(taxonomy 28) 적재 직후** 호출해야 한다(canonical 정본이 먼저 있어야
      canonicalize_topic 의 alias 히트 경로가 registry en 을 조회할 수 있음).
    - 효과: 백필 ``canonicalize_topic`` 이 off-list raw(에너지·천문 등)를 **LLM 재분류 없이** alias 히트로
      결정적 해소(SC-04v2·헌법 3조). subtopic 층은 이 시드가 다루지 않는다(열린 성장·LLM).
    - ``freeze_fn`` 주입 가능(단위 테스트용·기본은 topic_canonicalize seam).
    """
    if freeze_fn is None:
        # 지연 import — alias 동결 seam(ON CONFLICT 스코프 로직)을 재사용해 중복 정의를 피한다.
        from src.relations.topic_canonicalize import _freeze_alias

        freeze_fn = _freeze_alias

    entries = alias_seed_entries(alias_seed)
    for e in entries:
        freeze_fn(conn, e["raw_ko"], e["canonical_ko"], "seed", parent_topic=None)
    return {"n_alias": len(entries)}


def apply_subtopic_seed(
    conn, seed: dict[str, Any], *, register_fn=None
) -> dict[str, int]:
    """subtopic 시드의 각 행을 **부모 topic 스코프**(parent_topic=<topic_ko>)로 registry 등록.

    - 각 subtopic: ``register_topic(conn, subtopic_ko, subtopic_en, source='taxonomy',
      parent_topic=parent_topic)`` — 라벨 임베딩 1536D 계산·부분 유니크 인덱스
      (parent_topic, topic_ko) WHERE parent_topic IS NOT NULL 로 ON CONFLICT 멱등.
    - topic 층(parent NULL) 시드와 **독립**이며 subtopic 층을 삭제하지 않는다(가산 적재·
      governance §4 '전역 재빌드 없음'). 커밋은 호출부 책임.
    - ``register_fn`` 주입 가능(단위 테스트용·기본은 topic_canonicalize seam).
    Returns ``{"n_subtopic": N}``.
    """
    if register_fn is None:
        # 지연 import(모듈 기동 경량 유지). 무거운 임베더는 register_topic 내부에서만 로드.
        from src.relations.topic_canonicalize import register_topic

        register_fn = register_topic

    entries = subtopic_registry_entries(seed)
    for e in entries:
        register_fn(
            conn,
            e["subtopic_ko"],
            e["subtopic_en"],
            source="taxonomy",
            parent_topic=e["parent_topic"],
        )
    return {"n_subtopic": len(entries)}


# ────────────────────────────────────────────────────────────────────────────
# 3) 콘솔 요약 (순수)
# ────────────────────────────────────────────────────────────────────────────
def summarize_lines(seed: dict[str, Any]) -> list[str]:
    """콘솔 요약 줄 — 버전·정본 수·topic_ko 목록."""
    entries = taxonomy_registry_entries(seed)
    lines = [
        f"[taxonomy 시드] version={seed.get('version')} · 정본 topic {len(entries)}개"
        " (parent_topic=NULL·source='taxonomy'·alias 0)",
        "── topic_ko [topic_en] ──",
    ]
    lines += [f"  {e['topic_ko']} [{e['topic_en']}]" for e in entries]
    return lines


def summarize_subtopic_lines(seed: dict[str, Any]) -> list[str]:
    """subtopic 시드 콘솔 요약 — 버전·출처·총 subtopic 수·부모별 개수(결정적 순서)."""
    entries = subtopic_registry_entries(seed)
    per_parent: dict[str, int] = {}
    for e in entries:  # 파일 순서 보존(부모 첫 등장 순).
        per_parent[e["parent_topic"]] = per_parent.get(e["parent_topic"], 0) + 1
    lines = [
        f"[subtopic 시드] version={seed.get('version')} · source={seed.get('source')}",
        f"  총 subtopic {len(entries)}개(parent_topic=<topic>·source='taxonomy'·부모 {len(per_parent)}개 커버)",
        "── parent_topic: subtopic 수 ──",
    ]
    lines += [f"  {p}: {n}" for p, n in per_parent.items()]
    return lines


# ────────────────────────────────────────────────────────────────────────────
# 4) 실행 (dry-run 파싱만 / apply = TRUNCATE 후 재적재·커밋)
# ────────────────────────────────────────────────────────────────────────────
def _delete_topic_layer(conn) -> None:
    """기존 topic 층 시드 정리 — **parent_topic IS NULL(topic 층)만** 스코프 삭제.

    🔴 (PR #81 code-review) 과거 ``TRUNCATE topic_alias, topic_registry`` 는 v297 로 자란
    **subtopic 층(parent_topic IS NOT NULL·백필 성장 레이어·결정성 캐시)까지 통째로** 날려
    governance §4 '전역 재빌드 없음'과 상충하고, 재실행 시 프로덕션 subtopic 을 소실시켰다.
    → 삭제를 ``WHERE parent_topic IS NULL`` 로 스코프해 **subtopic 층은 보존**한다.

    FK 는 v297 에서 완화(드롭)됐으나, 읽기 순서(자식 alias → 부모 registry)를 유지해 명시한다.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM topic_alias WHERE parent_topic IS NULL")
        cur.execute("DELETE FROM topic_registry WHERE parent_topic IS NULL")


def run_seed(
    db, seed: dict[str, Any], alias_seed: dict[str, Any] | None = None, *, apply: bool = False
) -> dict[str, int]:
    """taxonomy 시드 + alias 선시드 적재(apply) 또는 dry-run(파싱만).

    - ``apply=False``: DB 미접촉(파싱·요약은 호출부). 카운트만 계산해 반환.
    - ``apply=True``: 한 트랜잭션에서 **topic 층만 스코프 삭제**(``_delete_topic_layer`` ·
      parent_topic IS NULL) → registry 28행 등록 → **직후 alias 선시드**(있으면·§3 매핑) → 커밋.
      registry→alias 순서를 지켜 alias 히트가 정본 en 을 조회할 수 있게 한다. subtopic 층
      (parent NOT NULL)은 삭제하지 않고 **보존**한다(governance §4·재실행 시 subtopic 무사).
    """
    if not apply:
        n_alias = len(alias_seed_entries(alias_seed)) if alias_seed is not None else 0
        return {"n_registry": len(taxonomy_registry_entries(seed)), "n_alias": n_alias}
    with db.connection() as conn:
        _delete_topic_layer(conn)
        counts = apply_taxonomy_seed(conn, seed)
        if alias_seed is not None:
            # registry 정본 적재 직후 alias 선시드(decided_by='seed'·parent NULL).
            counts["n_alias"] = apply_alias_seed(conn, alias_seed)["n_alias"]
        conn.commit()  # taxonomy + alias 실적재(T902·G12 driver)
    return counts


def run_subtopic_seed(
    db, subtopic_seed: dict[str, Any], *, apply: bool = False, register_fn=None
) -> dict[str, int]:
    """subtopic 시드 적재(apply) 또는 dry-run(카운트만) — **topic 시드와 독립**(spec 068 G3).

    - ``apply=False``: DB 미접촉(파싱·요약은 호출부). subtopic 행 수만 계산해 반환.
    - ``apply=True``: 한 트랜잭션에서 ``apply_subtopic_seed`` 로 subtopic 층에 **가산 적재** 후 커밋.
      topic 층 정리(``_delete_topic_layer``)를 호출하지 않는다 — subtopic 층은 삭제 없이 ON CONFLICT
      멱등으로만 재적용된다(governance §4 전역 재빌드 없음). 기존 topic 시드 경로는 불변.
    - ``register_fn`` 주입 가능(단위 테스트용 — 실 register_topic 임베딩 없이 배선 검증).
    """
    if not apply:
        return {"n_subtopic": len(subtopic_registry_entries(subtopic_seed))}
    with db.connection() as conn:
        counts = apply_subtopic_seed(conn, subtopic_seed, register_fn=register_fn)
        conn.commit()  # subtopic 층 가산 적재(G6 driver)
    return counts


def main() -> int:
    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    p = argparse.ArgumentParser(
        description="topic 정본 레지스트리 시드 — 닫힌 분류체계(taxonomy) 적재(spec 058 v2 T902)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument(
        "--seed", default=str(_DEFAULT_SEED_PATH), help="taxonomy 시드 JSON 경로(정본)"
    )
    p.add_argument(
        "--alias-seed",
        default=str(_DEFAULT_ALIAS_SEED_PATH),
        help="alias 선시드 JSON 경로(§3 커버리지 매핑·raw_ko→canonical)",
    )
    p.add_argument(
        "--no-alias-seed",
        action="store_true",
        help="alias 선시드를 건너뛴다(registry 만 적재·과거 동작).",
    )
    p.add_argument(
        "--subtopics",
        action="store_true",
        help="subtopic 시드도 함께 처리(spec 068·부모 topic 스코프 가산 적재·topic 층과 독립·"
        "subtopic 층 삭제 안 함). --apply 와 함께면 실적재, 아니면 dry-run 요약.",
    )
    p.add_argument(
        "--subtopic-seed",
        default=str(_DEFAULT_SUBTOPIC_SEED_PATH),
        help="subtopic 시드 JSON 경로(정본·src/relations/subtopic_seed.json).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="파싱·요약만 출력(DB 미접촉·기본)."
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="topic 층 스코프 삭제(parent NULL) 후 taxonomy 28행 + alias 선시드 적재·커밋"
        "(T902·G12·subtopic 층 보존).",
    )
    args = p.parse_args()

    dotenv_path = _REPO_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    seed = load_taxonomy_seed(Path(args.seed))
    alias_seed = None if args.no_alias_seed else load_alias_seed(Path(args.alias_seed))
    # subtopic 시드는 --subtopics 일 때만 로드(기존 topic 시드 경로에 영향 없음·독립).
    subtopic_seed = load_subtopic_seed(Path(args.subtopic_seed)) if args.subtopics else None

    if args.apply:
        db = PostgresUtil()
        with db:
            counts = run_seed(db, seed, alias_seed, apply=True)
            # subtopic 층 가산 적재(topic 층 삭제·재적재와 독립·같은 db 풀 재사용).
            sub_counts = (
                run_subtopic_seed(db, subtopic_seed, apply=True)
                if args.subtopics
                else None
            )
        print(
            f"[APPLY] taxonomy 시드 적재 완료: {args.seed}\n"
            f"  정본(register_topic) {counts['n_registry']}개(parent_topic=NULL·source='taxonomy') · "
            f"alias(선시드·decided_by='seed') {counts['n_alias']}개 "
            "(topic 층 스코프 삭제 후 재적재·커밋·subtopic 층 보존)."
        )
        if sub_counts is not None:
            print(
                f"[APPLY] subtopic 시드 적재 완료: {args.subtopic_seed}\n"
                f"  subtopic {sub_counts['n_subtopic']}개(parent_topic=<topic>·source='taxonomy'·"
                "가산 적재·ON CONFLICT 멱등·subtopic 층 삭제 안 함)."
            )
        return 0

    # dry-run(기본): DB 미접촉·파싱 요약만.
    print("\n".join(summarize_lines(seed)))
    n_alias = len(alias_seed_entries(alias_seed)) if alias_seed is not None else 0
    print(f"\n(dry-run·DB 미접촉) alias 선시드 {n_alias}개 대기. 적용하려면 --apply.")
    print(f"  시드 파일: {args.seed}\n  alias 시드: {args.alias_seed if not args.no_alias_seed else '(생략)'}")
    if subtopic_seed is not None:
        print()
        print("\n".join(summarize_subtopic_lines(subtopic_seed)))
        print(f"\n(dry-run·DB 미접촉) subtopic 적용하려면 --apply --subtopics. 시드 파일: {args.subtopic_seed}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
