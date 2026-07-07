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
    ``--apply``: **기존 v1 시드 정리 후 재적재** — 닫힌 분류체계는 정본 집합이 통째로 갈리므로
        ``TRUNCATE topic_alias, topic_registry``(alias 먼저·읽기 순서) 후 taxonomy 28행을 적재하고
        커밋한다. register_topic 의 ``ON CONFLICT ... DO NOTHING`` 으로 재적용도 멱등(28행 유지).

헌법·불변식
    - **학습 0(2조)**: 임베딩은 추론만(register_topic·st_bge). 규칙은 결정적.
    - **결정성(3조)**: 시드 파일이 정본 — 같은 파일 → 같은 registry(재현 가능). LLM/kNN 재실행 0.
    - **1536D/pgvector(4조)**: register_topic 이 라벨 임베딩 1536D 저장(0-노름 거부).
    - 조회/적재행 라벨은 ``str()`` 강제(graph_query 관례).

실행
    conda activate AuroraFS
    python -m scripts.seed_topic_registry --env dev --dry-run     # 파싱·요약(DB 미접촉)
    python -m scripts.seed_topic_registry --env dev --apply       # TRUNCATE→28행 적재·커밋
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# 시드 정본 경로(repo 내). spec 058 디렉터리의 taxonomy_seed.json.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC_DIR = _REPO_ROOT / "specs" / "058-relation-topic-canonicalization"
_DEFAULT_SEED_PATH = _SPEC_DIR / "taxonomy_seed.json"
# alias 선시드 정본(§3 커버리지 매핑·raw_ko→canonical). registry 시드 직후 topic 층 alias 로 적재.
_DEFAULT_ALIAS_SEED_PATH = _SPEC_DIR / "taxonomy_alias_seed.json"


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


# ────────────────────────────────────────────────────────────────────────────
# 4) 실행 (dry-run 파싱만 / apply = TRUNCATE 후 재적재·커밋)
# ────────────────────────────────────────────────────────────────────────────
def _truncate_topic_tables(conn) -> None:
    """기존 v1 시드 정리 — alias·registry 를 TRUNCATE(닫힌 분류체계 통째 재적재 전).

    FK 는 v297 에서 완화(드롭)됐으나, 읽기 순서(자식 alias → 부모 registry)를 유지해 명시한다.
    한 문장 TRUNCATE 로 두 테이블을 함께 비운다(원자적).
    """
    with conn.cursor() as cur:
        cur.execute("TRUNCATE topic_alias, topic_registry")


def run_seed(
    db, seed: dict[str, Any], alias_seed: dict[str, Any] | None = None, *, apply: bool = False
) -> dict[str, int]:
    """taxonomy 시드 + alias 선시드 적재(apply) 또는 dry-run(파싱만).

    - ``apply=False``: DB 미접촉(파싱·요약은 호출부). 카운트만 계산해 반환.
    - ``apply=True``: 한 트랜잭션에서 ``TRUNCATE topic_alias, topic_registry`` → registry 28행 등록
      → **직후 alias 선시드**(있으면·§3 매핑) → 커밋. registry→alias 순서를 지켜 alias 히트가
      정본 en 을 조회할 수 있게 한다.
    """
    if not apply:
        n_alias = len(alias_seed_entries(alias_seed)) if alias_seed is not None else 0
        return {"n_registry": len(taxonomy_registry_entries(seed)), "n_alias": n_alias}
    with db.connection() as conn:
        _truncate_topic_tables(conn)
        counts = apply_taxonomy_seed(conn, seed)
        if alias_seed is not None:
            # registry 정본 적재 직후 alias 선시드(decided_by='seed'·parent NULL).
            counts["n_alias"] = apply_alias_seed(conn, alias_seed)["n_alias"]
        conn.commit()  # taxonomy + alias 실적재(T902·G12 driver)
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
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="파싱·요약만 출력(DB 미접촉·기본)."
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="기존 시드 TRUNCATE 후 taxonomy 28행 + alias 선시드 적재·커밋(T902·G12).",
    )
    args = p.parse_args()

    dotenv_path = _REPO_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    seed = load_taxonomy_seed(Path(args.seed))
    alias_seed = None if args.no_alias_seed else load_alias_seed(Path(args.alias_seed))

    if args.apply:
        db = PostgresUtil()
        with db:
            counts = run_seed(db, seed, alias_seed, apply=True)
        print(
            f"[APPLY] taxonomy 시드 적재 완료: {args.seed}\n"
            f"  정본(register_topic) {counts['n_registry']}개(parent_topic=NULL·source='taxonomy') · "
            f"alias(선시드·decided_by='seed') {counts['n_alias']}개 (TRUNCATE 후 재적재·커밋)."
        )
        return 0

    # dry-run(기본): DB 미접촉·파싱 요약만.
    print("\n".join(summarize_lines(seed)))
    n_alias = len(alias_seed_entries(alias_seed)) if alias_seed is not None else 0
    print(f"\n(dry-run·DB 미접촉) alias 선시드 {n_alias}개 대기. 적용하려면 --apply.")
    print(f"  시드 파일: {args.seed}\n  alias 시드: {args.alias_seed if not args.no_alias_seed else '(생략)'}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
