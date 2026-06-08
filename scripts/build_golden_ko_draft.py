#!/usr/bin/env python3
"""한국어 골든셋 초안 생성 — registered 자산 ext_meta 에서 질의·정답 초안을 뽑는다 (017 G4 / T010).

017 A/B(KoSimCSE vs BGE-M3) 하니스(`tests/test_embedding_ab_kpi.py`)는 **사람이 검수·확정한**
한국어 골든셋(`tests/fixtures/search/golden_ko.json`)으로 두 채널 retrieval 품질을 비교한다.
사람이 질의를 100% 손으로 만드는 부담을 줄이기 위해, 이 스크립트는 자산의 ``ext_meta``
(``summary``/``keywords``)에서 **질의 초안 + 정답 자산 id(자기 자산)** 를 **결정적**으로 뽑아
``tests/fixtures/search/golden_ko.draft.json`` 에 기록한다.

⚠️ **초안은 noisy 하다 — 사람 확정본만 하니스가 사용한다.**
  - 자동 라벨(정답=자기 자산 1건)은 누락·과잉이 있을 수 있다. 같은 주제의 다른 자산이
    실제로는 정답인데 빠지거나, 질의가 너무 일반적이라 부정확할 수 있다.
  - 사람이 이 초안을 열어 ① 질의 문장을 자연스럽게 다듬고 ② 정답 자산 id 를 보정(추가/제거)해
    ``golden_ko.json`` 으로 확정한다(질의 20~50건 권장, plan R-3). 하니스는 ``golden_ko.json``
    이 **존재할 때만** 돈다(초안 파일은 입력으로 쓰지 않는다).

결정성(헌법 3조): 대상 자산을 ``ORDER BY asset_id`` 로 고정 순서 조회하고, 질의 생성 규칙은
시드 없는 순수 함수(``_draft_query_from_ext_meta``)다 → 같은 DB 상태면 2회 실행 동일 초안.
읽기 전용(SELECT 만, 쓰기·스키마 0). 학습 0(LLM·모델 미사용 — 메타에서 규칙 추출만).

실행
    conda activate AuroraFS
    python scripts/build_golden_ko_draft.py --env dev --limit 30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

_LOG = logging.getLogger("meta_extract.build_golden_ko_draft")

# 초안 출력 경로(사람이 검수해 golden_ko.json 으로 확정). 초안 파일 자체는 커밋 대상 아님.
_DRAFT_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "search" / "golden_ko.draft.json"

# 질의 문장 길이 상한(요약 기반 질의가 너무 길어지지 않게). 검수자가 다듬는 출발점.
_QUERY_MAX_LEN = 60
# 키워드 기반 질의에 묶는 키워드 수(너무 많으면 질의가 부자연).
_KEYWORDS_IN_QUERY = 3
# 요약 첫 문장 분리 경계(한국어/영문 종결부호·개행).
_SENTENCE_BREAKS = (". ", "。", "!", "?", "\n")


def _configure_logging() -> None:
    if _LOG.handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


def _draft_query_from_ext_meta(ext_meta: dict[str, Any]) -> str | None:
    """``ext_meta`` 에서 한국어 질의 초안 문자열을 **결정적**으로 만든다(없으면 None).

    규칙(순서 고정 — 시드·무작위 없음):
      1) ``keywords`` 가 있으면 앞 ``_KEYWORDS_IN_QUERY`` 개 비어있지 않은 문자열을 공백으로 묶는다.
         (키워드 질의는 retrieval 평가에 가장 자연스러운 짧은 질의.)
      2) 없으면 ``summary`` 의 **첫 문장**(종결부호/개행 경계)을 ``_QUERY_MAX_LEN`` 자로 잘라 쓴다.
      3) 둘 다 없으면 None(이 자산은 초안에서 제외 — 사람이 따로 추가).
    """
    # 1) keywords 우선 — list[str] 또는 단일 str 모두 허용(방어적).
    keywords = ext_meta.get("keywords")
    terms: list[str] = []
    if isinstance(keywords, str):
        terms = [keywords.strip()] if keywords.strip() else []
    elif isinstance(keywords, list):
        terms = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
    if terms:
        return " ".join(terms[:_KEYWORDS_IN_QUERY])

    # 2) summary 첫 문장 폴백.
    summary = ext_meta.get("summary")
    if isinstance(summary, str) and summary.strip():
        text = summary.strip()
        cut = len(text)
        for brk in _SENTENCE_BREAKS:
            pos = text.find(brk)
            if pos != -1:
                cut = min(cut, pos)
        first = text[:cut].strip() or text.strip()
        return first[:_QUERY_MAX_LEN].strip()

    return None


def _fetch_assets(conn: Any, *, limit: int | None) -> list[dict[str, Any]]:
    """registered 자산 중 ``summary``/``keywords`` 가 있는 자산의 asset_id·ext_meta 를 결정적 순서로.

    ``ORDER BY a.asset_id`` 로 고정 순서(2회 실행 동일·``--limit`` 단계적 추출이 같은 앞부분).
    """
    sql = (
        "SELECT a.asset_id, am.ext_meta "
        "FROM asset a "
        "JOIN asset_metadata am ON am.asset_id = a.asset_id "
        "WHERE a.status = 'registered' "
        "AND (am.ext_meta ? 'summary' OR am.ext_meta ? 'keywords') "
        "ORDER BY a.asset_id"
    )
    params: list[Any] = []
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def build_drafts(conn: Any, *, limit: int | None) -> list[dict[str, Any]]:
    """대상 자산을 조회해 질의 초안 목록 ``[{"query", "relevant_asset_ids"}]`` 을 만든다.

    질의를 못 만든 자산(summary·keywords 둘 다 빈약)은 건너뛴다(사람이 따로 보강).
    정답은 **자기 자산 1건**(noisy — 사람이 같은 주제 자산을 추가/보정).
    """
    rows = _fetch_assets(conn, limit=limit)
    drafts: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        ext_meta = row.get("ext_meta") or {}
        query = _draft_query_from_ext_meta(ext_meta if isinstance(ext_meta, dict) else {})
        if not query:
            skipped += 1
            continue
        drafts.append(
            {"query": query, "relevant_asset_ids": [str(row["asset_id"])]}
        )
    _LOG.info("초안 생성: %s건 (조회 %s건, 질의 생성 불가 skip %s건)", len(drafts), len(rows), skipped)
    return drafts


# ── 부트스트랩(backfill_bge_embeddings 와 동일 순서) ─────────────────────────────
# 1) load_dotenv(.env.{env}, override=False) → 2) init_settings(env) → 3) PostgresUtil() + `with` →
# 4) build_drafts → 5) golden_ko.draft.json 기록. 읽기 전용(SELECT 만).
def main() -> int:
    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    parser = argparse.ArgumentParser(
        description="한국어 골든셋 초안 생성 (017 A/B 하니스용; 사람 검수 전 초안)"
    )
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="초안 대상 자산 수 상한")
    args = parser.parse_args()

    _configure_logging()
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    db = PostgresUtil()
    with db, db.transaction() as conn:
        drafts = build_drafts(conn, limit=args.limit)

    _DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DRAFT_PATH.write_text(
        json.dumps(drafts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _LOG.info("기록: %s (%s건)", _DRAFT_PATH, len(drafts))
    print(
        f"초안 {len(drafts)}건 기록 → {_DRAFT_PATH}\n"
        "다음: 사람이 질의 문장·정답 자산 id 를 검수·보정해 "
        "tests/fixtures/search/golden_ko.json 으로 확정하세요(질의 20~50건 권장). "
        "자동 라벨은 noisy → 초안일 뿐, 하니스는 확정본(golden_ko.json)만 사용합니다."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
