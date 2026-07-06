"""topic 정본 레지스트리 시드 스크립트 — **런타임 replay 방식** (spec 058 G5 · T501).

목적
    현 코퍼스의 자유기입 topic 라벨(distinct ≈120)을 **실제 수집 경로(``canonicalize_topic``)로
    재생**해 정본(canonical) topic + alias 초안을 산출한다. 기존 holistic 1회 클러스터링을 걷어내고
    "시드 = 백필 = 수집" 일치를 보장한다(같은 seam·같은 kNN·같은 judge). 산출물은 **사람 검수용
    초안**이며(T502 사람 게이트), 검수 후 ``--replay --apply`` 로 dev registry/alias 에 적재한다.

replay 설계
    distinct topic 을 **빈도 desc → topic_ko asc**(결정적) 순으로 하나씩 실제
    ``canonicalize_topic(conn, topic_ko, topic_en)`` 에 통과시킨다. 이는 런타임 수집이 하는 바 그대로다:
      · 첫 topic → kNN 빈 → NEW 등록(LLM 0).
      · 이후 동의어 → kNN 후보 + judge(LLM·temp=0) → 기존 정본 매칭 또는 NEW.
    registry/alias 가 replay 로 성장하며, 그 결과를 역집계(alias raw→canonical)해 그룹 초안을 만든다.

두 모드
    ``--dry-run``(기본): **하나의 커넥션/트랜잭션** 안에서 전체 replay 를 수행한다
        (register_topic·alias INSERT 가 같은 txn 내 kNN 에 보임 → 순차 성장 재현). replay 후
        registry 스냅샷을 뜬 뒤 **롤백**(dev DB 미오염). 결과를 초안 JSON 에 기록 + 콘솔 요약.
        LLM/임베딩은 **실호출**되지만(캡처는 메모리), DB 변경은 롤백으로 0. 검수 재생성용.
    ``--from-draft <file>``(= ``--apply``): **검수본 초안 JSON 을 그대로 적재**(T502). replay 를
        재실행하지 않고(LLM/kNN 0) group→register_topic(source='seed')·member→alias(decided_by='seed')
        만 결정적으로 커밋한다. **검수본이 정본** — 사람 수정(정본 뒤집기·분리)이 그대로 반영된다
        (replay-commit 은 검수 수정을 무시하므로 폐기). 경로 생략 시 ``--out`` 사용.

헌법·불변식
    - **LLM 단일 seam**(2조): judge 는 ``src.llm.client.complete_json`` 만·temp=0·zero-shot·후보 K개만.
    - **결정성**(3조): 처리 순서(빈도 desc→ko asc)·temp=0·판정 결과 alias 동결 → 동일 코퍼스·동일
      순서 → 동일 결과. 재실행은 캐시 히트로 LLM 급감.
    - **학습 0**(2조): 임베딩은 추론만(``register_topic``·kNN), 규칙은 결정적.
    - 조회행 topic_ko/topic_en 은 ``str()`` 강제(graph_query 관례).

실행
    conda activate AuroraFS
    # T501 초안 생성(검수용·DB 미오염):
    python -m scripts.seed_topic_registry --env dev --dry-run
      → specs/058-relation-topic-canonicalization/seed_topic_draft.json + 콘솔 요약.
    # T502 검수본 적재(사람 검수 완료 후·커밋·LLM 0):
    python -m scripts.seed_topic_registry --env dev \
        --from-draft specs/058-relation-topic-canonicalization/seed_topic_draft.json
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


# ────────────────────────────────────────────────────────────────────────────
# 2) replay 순서·역집계·통계·초안 (결정적 순수 함수)
# ────────────────────────────────────────────────────────────────────────────
def build_replay_order(freq_by_ko: dict[str, int]) -> list[str]:
    """replay 처리 순서 — 빈도 desc → topic_ko asc(결정적·런타임 수집 순서 근사).

    빈도 높은 라벨이 먼저 처리돼 정본(canonical)이 되고, 이후 저빈도 동의어가 그 정본에 흡수된다.
    타이브레이커(ko asc)를 고정해 같은 입력에 같은 순서·같은 결과(헌법 3조)를 보장한다.
    """
    return [ko for ko, _ in sorted(freq_by_ko.items(), key=lambda kv: (-kv[1], kv[0]))]


def build_replay_groups(
    resolutions: list[dict[str, str]],
    freq_by_ko: dict[str, int],
    en_variants: dict[str, dict[str, int]],
    registry_en: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """replay 해소 결과(raw→canonical)를 **정본별 그룹으로 역집계**.

    Args:
        resolutions: 처리 순서대로의 ``[{raw_ko, canonical_ko}]`` — 각 원본이 어느 정본으로 해소됐는지.
        freq_by_ko: 원본별 빈도(총빈도·정렬용).
        en_variants: 원본별 관측 topic_en 변형(canonical_en 폴백용).
        registry_en: replay 후 ``topic_registry`` 스냅샷 ``{canonical_ko: topic_en}``. 있으면 우선.

    반환 각 그룹: ``{canonical_ko, canonical_en, members:[raw...], total_freq, decided_by}``.
      - members = 그 정본으로 해소된 raw 전부(정본 자신 포함)·빈도 desc → 라벨 asc.
      - decided_by = 병합그룹(members>1) → ``"llm"``(judge 로 흡수) · 단독 → ``"new"``(신규 등록만).
      - total_freq = members 빈도 합.
    정렬 = 병합그룹(members>1) 우선 → total_freq desc → canonical_ko asc(결정적).
    """
    registry_en = registry_en or {}
    members_by_canonical: dict[str, list[str]] = {}
    order: list[str] = []  # canonical 최초 등장 순서(결정적 보조)
    for r in resolutions:
        raw = str(r["raw_ko"])
        cano = str(r["canonical_ko"])
        if cano not in members_by_canonical:
            members_by_canonical[cano] = []
            order.append(cano)
        members_by_canonical[cano].append(raw)

    groups: list[dict[str, Any]] = []
    for cano in order:
        members = members_by_canonical[cano]
        members_sorted = sorted(members, key=lambda m: (-freq_by_ko.get(m, 0), m))
        total_freq = sum(freq_by_ko.get(m, 0) for m in members)
        canonical_en = registry_en.get(cano) or _best_en(en_variants.get(cano))
        decided_by = "llm" if len(members) > 1 else "new"
        groups.append(
            {
                "canonical_ko": cano,
                "canonical_en": canonical_en,
                "members": members_sorted,
                "total_freq": total_freq,
                "decided_by": decided_by,
            }
        )

    # 정렬: 병합그룹 우선 → total_freq desc → canonical_ko asc(결정적)
    groups.sort(
        key=lambda g: (0 if len(g["members"]) > 1 else 1, -g["total_freq"], g["canonical_ko"])
    )
    return groups


def build_stats(groups: list[dict[str, Any]], llm_calls: int) -> dict[str, int]:
    """replay 요약 통계 — 정본 수·병합 그룹 수·단독 수·LLM 판정 호출 수."""
    return {
        "n_canonical": len(groups),
        "n_merged_groups": sum(1 for g in groups if len(g["members"]) > 1),
        "n_singleton": sum(1 for g in groups if len(g["members"]) == 1),
        "llm_calls": llm_calls,
    }


def build_draft(
    groups: list[dict[str, Any]],
    stats: dict[str, int],
    *,
    n_topics: int,
    n_edges: int,
) -> dict[str, Any]:
    """검수용 초안 JSON 구조(replay 모드). spec 지정 구조 + 검수 NOTE."""
    return {
        "mode": "replay",
        "generated_from": {"n_topics": n_topics, "n_edges": n_edges},
        "groups": groups,
        "stats": stats,
        "_NOTE": (
            "런타임 canonicalize replay 초안 — 사람 검수 필수(T502). 오병합/누락(under-merge) 교정 후 "
            "`--replay --apply` 로 dev registry/alias 에 적재(같은 replay 를 커밋). members>1 이 실제 병합."
        ),
    }


# ────────────────────────────────────────────────────────────────────────────
# 3) 콘솔 요약
# ────────────────────────────────────────────────────────────────────────────
def summarize_lines(draft: dict[str, Any]) -> list[str]:
    """콘솔 요약 줄 목록 — N→M·LLM 호출 수·병합 그룹 목록(canonical(총빈도)[en] ← members)."""
    gen = draft["generated_from"]
    groups = draft["groups"]
    stats = draft["stats"]
    n_merge = stats["n_merged_groups"]
    n_singleton = stats["n_singleton"]
    lines = [
        f"[replay 시드 초안] {gen['n_topics']} topic (active 엣지 {gen['n_edges']}) "
        f"→ {stats['n_canonical']} 정본 (병합 {n_merge} · 단독 {n_singleton})",
        f"LLM 판정(judge) 호출: {stats['llm_calls']}회",
        "",
        f"── 병합 그룹({n_merge}) — canonical(총빈도)[en] ← members ──",
    ]
    for g in groups:
        if len(g["members"]) <= 1:
            continue
        members_str = ", ".join(m for m in g["members"] if m != g["canonical_ko"])
        lines.append(
            f"  {g['canonical_ko']}({g['total_freq']}) [{g['canonical_en']}] ← {members_str}"
        )
    return lines


# ────────────────────────────────────────────────────────────────────────────
# 4) LLM 호출 카운팅 래퍼 — 실제 judge(complete_json) 호출 수를 정확히 계측
# ────────────────────────────────────────────────────────────────────────────
class _CountingCompletions:
    """``client.chat.completions.create`` 를 감싸 호출 수를 센다(다른 인자는 그대로 위임)."""

    def __init__(self, parent: _CountingClient) -> None:
        self._parent = parent

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._parent.calls += 1
        return self._parent._inner.chat.completions.create(*args, **kwargs)


class _CountingChat:
    def __init__(self, parent: _CountingClient) -> None:
        self.completions = _CountingCompletions(parent)


class _CountingClient:
    """온프레미스 LLM 클라이언트 얇은 래퍼 — judge 실호출 수 계측(``complete_json`` seam 경유).

    ``judge_topic`` 은 kNN 후보가 있을 때만 ``complete_json(prompt, client=...)`` 을 부른다.
    이 래퍼를 canonicalize 에 주입하면 실제 LLM 호출만 정확히 카운트한다(초기 NEW=후보 0=미호출).
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0
        self.chat = _CountingChat(self)


def _snapshot_registry_en(conn) -> dict[str, str | None]:
    """replay 후(롤백 전) ``topic_registry`` 스냅샷 ``{topic_ko: topic_en}``(조회행 str() 강제)."""
    out: dict[str, str | None] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT topic_ko, topic_en FROM topic_registry")
        for r in cur.fetchall():
            ko = str(r["topic_ko"])
            en = r["topic_en"]
            out[ko] = str(en) if en is not None else None
    return out


# ────────────────────────────────────────────────────────────────────────────
# 5) replay 실행 (dry-run 롤백 / apply 커밋)
# ────────────────────────────────────────────────────────────────────────────
def run_replay(db, out_path: Path, *, apply: bool = False, client=None) -> dict[str, Any]:
    """distinct topic 을 실제 ``canonicalize_topic`` 으로 재생 → 초안 산출.

    **하나의 커넥션/트랜잭션** 안에서 순차 처리한다(autocommit off — INSERT 가 같은 txn 내 kNN 에
    보여 registry 가 순차 성장). replay 후 registry 스냅샷을 떠서 초안을 만들고:
      - ``apply=False``(dry-run): ``conn.rollback()`` — DB 변경 0(LLM/임베딩만 실호출·캡처는 메모리).
      - ``apply=True``: ``conn.commit()`` — dev registry/alias 실적재(T502 게이트 이후).

    ``client`` 미주입이면 운영 온프레미스 seam(``get_llm_client``)을 래핑해 judge 호출 수를 계측한다.
    """
    from src.llm.client import get_llm_client
    from src.relations.topic_canonicalize import canonicalize_topic

    counting = _CountingClient(client or get_llm_client())

    with db.connection() as conn:
        freq_by_ko, en_variants, n_edges = fetch_topic_stats(conn)
        order = build_replay_order(freq_by_ko)

        resolutions: list[dict[str, str]] = []
        for ko in order:
            topic_en = _best_en(en_variants.get(ko))
            # ★ 런타임 수집과 동일 경로: alias 정확일치 → kNN 후보 → judge(LLM·temp=0) → 등록/동결.
            res = canonicalize_topic(conn, ko, topic_en, client=counting)
            resolutions.append(
                {"raw_ko": ko, "canonical_ko": str(res["canonical_ko"])}
            )

        # 롤백 전에 registry 스냅샷(정본 topic_en 확정본) 확보 → 초안 구성.
        registry_en = _snapshot_registry_en(conn)
        groups = build_replay_groups(resolutions, freq_by_ko, en_variants, registry_en)
        stats = build_stats(groups, counting.calls)
        draft = build_draft(groups, stats, n_topics=len(freq_by_ko), n_edges=n_edges)

        if apply:
            conn.commit()  # dev registry/alias 실적재(T502 이후)
        else:
            conn.rollback()  # dry-run: DB 미오염(replay 부작용 폐기)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(draft, f, ensure_ascii=False, indent=2)
    return draft


# ────────────────────────────────────────────────────────────────────────────
# 6) 검수본 draft 적재 (--from-draft) — 결정적·LLM/kNN 재실행 0 (T502)
# ────────────────────────────────────────────────────────────────────────────
# 왜 replay-commit 이 아니라 draft 적재인가(T502 결정):
#   replay(--apply)는 LLM judge 를 재실행·커밋이라 **사람 검수 수정을 무시**한다(천문학→천문 뒤집기,
#   레저/여가·서예/캘리그라피 분리는 replay 로 재현 불가). 검수본 JSON 을 정본으로 삼아 **그대로**
#   registry/alias 에 적재하면 LLM 0·결정적·재현 가능하다(plan "검수본 파일로 재현" 부합).
def read_draft(path: Path | str) -> dict[str, Any]:
    """검수본 초안 JSON 을 읽어 dict 로 반환(순수 I/O·파싱만)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def draft_registry_entries(draft: dict[str, Any]) -> list[dict[str, str]]:
    """검수본 → 정본(registry) 적재 행 ``[{canonical_ko, canonical_en}]`` (group 당 1개·draft 순서 보존).

    조회/적재행 계약(graph_query 관례)에 맞춰 라벨은 ``str()`` 강제.
    """
    return [
        {"canonical_ko": str(g["canonical_ko"]), "canonical_en": str(g["canonical_en"])}
        for g in draft["groups"]
    ]


def draft_alias_entries(draft: dict[str, Any]) -> list[dict[str, str]]:
    """검수본 → alias 적재 행 ``[{raw_ko, canonical_ko}]`` (group 의 각 member 를 raw→canonical 로).

    정본 자신도 members 에 포함되므로 self-alias(raw=canonical)로 자연히 처리된다. 분리 그룹은
    각자 자신만 member 이므로 상호 alias 가 생기지 않는다(병합 아님). draft 순서 보존.
    """
    rows: list[dict[str, str]] = []
    for g in draft["groups"]:
        cano = str(g["canonical_ko"])
        for m in g["members"]:
            rows.append({"raw_ko": str(m), "canonical_ko": cano})
    return rows


def apply_draft(
    conn, draft: dict[str, Any], *, register_fn=None, alias_fn=None
) -> dict[str, int]:
    """검수본을 registry/alias 에 **결정적 적재**(LLM/kNN 재실행 0). 적재 수 ``{n_registry, n_alias}``.

    - 각 group: ``register_topic(conn, canonical_ko, canonical_en, source='seed')`` — 라벨 임베딩
      계산·``ON CONFLICT DO NOTHING`` 멱등(0-노름 가드 유지).
    - 각 member: ``_freeze_alias(conn, raw_ko, canonical_ko, 'seed')`` — ``ON CONFLICT DO NOTHING``.
    - ``register_fn``/``alias_fn`` 주입 가능(단위 테스트용·기본은 topic_canonicalize seam). 커밋은
      호출부(``run_from_draft``) 책임.
    """
    if register_fn is None or alias_fn is None:
        # 지연 import(모듈 기동 경량 유지·기존 lazy 관례). 무거운 임베더는 register_topic 내부에서만.
        from src.relations.topic_canonicalize import _freeze_alias, register_topic

        register_fn = register_fn or register_topic
        alias_fn = alias_fn or _freeze_alias

    reg = draft_registry_entries(draft)
    ali = draft_alias_entries(draft)
    for e in reg:
        register_fn(conn, e["canonical_ko"], e["canonical_en"], source="seed")
    for e in ali:
        alias_fn(conn, e["raw_ko"], e["canonical_ko"], "seed")
    return {"n_registry": len(reg), "n_alias": len(ali)}


def run_from_draft(db, path: Path) -> dict[str, int]:
    """검수본 파일을 dev registry/alias 에 적재하고 **커밋**(T502). 적재 수 반환.

    LLM/kNN 재실행 없음 — 검수본이 정본. 한 트랜잭션으로 register/alias 적재 후 commit.
    """
    draft = read_draft(path)
    with db.connection() as conn:
        counts = apply_draft(conn, draft)
        conn.commit()  # 검수 완료분 실적재(T502)
    return counts


def main() -> int:
    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    p = argparse.ArgumentParser(
        description="topic 정본 레지스트리 시드 — 런타임 replay(spec 058 T501)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    # replay 는 유일한 방식(holistic 제거). 플래그는 명시성·호출 관례용(있어도 없어도 replay).
    p.add_argument(
        "--replay",
        action="store_true",
        help="런타임 canonicalize 재생 방식(기본·유일). 명시성용 플래그.",
    )
    # --from-draft: 검수본 JSON 을 결정적 적재(LLM/kNN 재실행 0·T502). 경로 미지정 시 --out 사용.
    p.add_argument(
        "--from-draft",
        default=None,
        metavar="PATH",
        help="검수본 초안 JSON 을 그대로 registry/alias 에 적재(결정적·LLM/kNN 0·커밋). "
        "경로 생략 시 --out 을 사용. --apply 도 이 경로로 동작(replay-commit 아님).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="트랜잭션 롤백 — DB 미오염·replay 초안 산출(기본). LLM/임베딩은 실호출(검수 재생성용).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="검수본 draft 적재 모드(=--from-draft, 경로는 --from-draft 또는 --out). "
        "LLM 재실행 없이 검수본을 dev registry/alias 에 커밋(T502).",
    )
    p.add_argument(
        "--out", default=str(_DEFAULT_DRAFT_PATH), help="초안 저장/적재 경로"
    )
    args = p.parse_args()

    dotenv_path = _REPO_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    # 적재 모드 여부: --from-draft 지정 or --apply. (--apply 는 이제 replay-commit 이 아니라 draft 적재.)
    load_mode = args.apply or (args.from_draft is not None)

    db = PostgresUtil()
    with db:
        if load_mode:
            draft_path = Path(args.from_draft or args.out)
            counts = run_from_draft(db, draft_path)
            print(
                f"[FROM-DRAFT] 검수본 적재 완료: {draft_path}\n"
                f"  정본(register_topic) {counts['n_registry']}개 · "
                f"alias {counts['n_alias']}개 (LLM/kNN 재실행 0·커밋)."
            )
            return 0
        draft = run_replay(db, Path(args.out), apply=False)

    print("\n".join(summarize_lines(draft)))
    print(f"\n초안 저장: {args.out} (dry-run·롤백·DB 미오염)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
