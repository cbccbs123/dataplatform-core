#!/usr/bin/env python3
"""한국어 골든셋 초안 생성 — 파일명 주제(prefix) 기반 정답군으로 질의·정답 초안을 만든다 (017 G4).

017 A/B(KoSimCSE vs BGE-M3) 하니스(`tests/test_embedding_ab_kpi.py`)는 **사람이 검수·확정한**
한국어 골든셋(`tests/fixtures/search/golden_ko.json`)으로 두 채널 retrieval 품질을 비교한다.

이 데이터는 파일명이 ``<주제>_<YouTube ID(11자)>_<제목>.<ext>`` 패턴으로 자연 군집한다
(예: ``무선_충전기_7iTajgt8pec_…``, ``라면_끓이기_GMjx9GrF1nY_…``). **같은 주제 = 관련 자산**
이므로, 주제 prefix 로 자산을 묶어 정답군으로 쓰면 "자기 자산 1건"보다 훨씬 의미 있는 골든셋이
된다(검수 부담도 작다 — 주제 그룹이 자동 정답). 이 스크립트는 주제별 그룹을 **결정적**으로 뽑아
``tests/fixtures/search/golden_ko.draft.json`` 에 기록한다.

규칙
  - **주제 prefix 추출**: basename 에서 ``<주제>_<11자 ID>_`` 의 ID 직전까지를 주제로 한다
    (``_topic_from_filename``). YouTube ID 는 ``[A-Za-z0-9_-]`` 11자(밑줄/하이픈 포함 가능)라
    한국어 주제 토큰과 구조적으로 구분된다. 패턴이 아니면 제외(예: ``manifest.json``).
  - **질의**: 주제명을 자연어로(밑줄→공백, 예: ``무선_충전기`` → "무선 충전기").
  - **정답**: 그 주제의 ``channel='st_bge'`` 백필된 자산 전부(평가 풀과 정합 — st_bge 없는 자산
    제외). 두 채널 공존 자산만 하니스가 평가하므로 st_bge 보유를 그룹 모집단으로 삼는다.
  - **--min-group N**(기본 2): 자산 수 N 미만 주제 제외(단일 자산 주제 노이즈↓).

⚠️ **초안은 사람 최종 확인이 필요하다 — 하니스는 확정본(golden_ko.json)만 사용한다.**
  사람이 이 초안을 열어 질의 문장·주제 선택을 검수·보정(주제 1~2개 제거/병합 등)해
  ``golden_ko.json`` 으로 확정한다(질의 20~50건 권장, plan R-3). 하니스는 ``golden_ko.json``
  이 **존재할 때만** 돈다(초안 파일은 입력으로 쓰지 않는다).

결정성(헌법 3조): 주제·asset_id 를 정렬해 같은 DB 상태면 2회 실행 동일 초안.
읽기 전용(SELECT 만, 쓰기·스키마 0). 학습 0(LLM·모델 미사용 — 파일명 규칙 추출만).

실행
    conda activate AuroraFS
    python scripts/build_golden_ko_draft.py --env dev --min-group 2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# scripts/ 직접 실행(python scripts/build_golden_ko_draft.py) 시 'src' 패키지를 찾도록
# 저장소 루트를 sys.path 에 추가한다(PYTHONPATH=. 없이도 동작). src 임포트보다 먼저 와야 한다.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from psycopg.rows import dict_row  # noqa: E402 — sys.path 부트스트랩 뒤에 와야 함

_LOG = logging.getLogger("meta_extract.build_golden_ko_draft")

# 초안 출력 경로(사람이 검수해 golden_ko.json 으로 확정). 초안 파일 자체는 커밋 대상 아님.
_DRAFT_PATH = _REPO_ROOT / "tests" / "fixtures" / "search" / "golden_ko.draft.json"

# 파일명 주제 prefix 규칙: <주제(밑줄 포함)>_<YouTube ID 11자>_<제목>.
# YouTube ID = [A-Za-z0-9_-] 11자(밑줄/하이픈 포함 가능). 주제는 비탐욕(.+?)으로 ID 직전까지.
# 한국어 주제 토큰은 ASCII 가 아니므로, ID 직전 11자 ASCII 토큰이 유일하게 매칭된다.
_TOPIC_ID_RE = re.compile(r"^(?P<topic>.+?)_(?P<id>[A-Za-z0-9_-]{11})_")


def _configure_logging() -> None:
    if _LOG.handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


def _topic_from_filename(name: str) -> str | None:
    """파일 basename 에서 주제 prefix(밑줄 보존)를 추출한다(패턴 아니면 None).

    ``무선_충전기_7iTajgt8pec_…`` → ``무선_충전기``. ID 가 밑줄을 포함해도(``5ncp-_GXBsU``)
    11자 ASCII 토큰으로 한 번에 매칭되므로 주제(``스마트폰``)가 정확히 잘린다.
    """
    m = _TOPIC_ID_RE.match(name)
    if not m:
        return None
    topic = m.group("topic").strip()
    return topic or None


def _query_from_topic(topic: str) -> str:
    """주제명을 자연어 질의로(밑줄→공백). ``무선_충전기`` → "무선 충전기"."""
    return topic.replace("_", " ").strip()


def _fetch_assets(conn: Any) -> list[dict[str, Any]]:
    """``channel='st_bge'`` 백필된 registered 자산의 asset_id·fs_path 를 결정적 순서로 조회.

    골든셋 정답은 st_bge 보유 자산으로 한정한다(하니스 평가 풀 = 두 채널 공존). ``ORDER BY
    a.asset_id`` 로 고정 순서(2회 실행 동일). 백필을 먼저 돌린 뒤 이 스크립트를 실행한다.
    """
    sql = (
        "SELECT DISTINCT a.asset_id, a.fs_path "
        "FROM asset a "
        "JOIN asset_embedding e ON e.asset_id = a.asset_id AND e.channel = 'st_bge' "
        "WHERE a.status = 'registered' "
        "ORDER BY a.asset_id"
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql)
        return list(cur.fetchall())


def build_drafts(conn: Any, *, min_group: int = 2) -> list[dict[str, Any]]:
    """주제별 그룹으로 골든셋 초안 ``[{"query", "relevant_asset_ids"}]`` 을 만든다(결정적).

    파일명에서 주제 prefix 를 추출해 자산을 묶고, 자산 수 ``min_group`` 미만 주제는 제외한다.
    질의=주제명(밑줄→공백), 정답=주제 자산 전부(중복 제거·asset_id 정렬). 주제명 정렬로 결정적.
    """
    rows = _fetch_assets(conn)
    groups: dict[str, set[str]] = {}
    for row in rows:
        topic = _topic_from_filename(Path(row["fs_path"]).name)
        if topic is None:
            continue
        groups.setdefault(topic, set()).add(str(row["asset_id"]))

    drafts: list[dict[str, Any]] = []
    skipped = 0
    for topic in sorted(groups):
        ids = sorted(groups[topic])
        if len(ids) < min_group:
            skipped += 1
            continue
        drafts.append({"query": _query_from_topic(topic), "relevant_asset_ids": ids})
    _LOG.info(
        "초안 생성: 주제 %s개 (조회 %s건, min-group<%s 제외 %s개)",
        len(drafts), len(rows), min_group, skipped,
    )
    return drafts


# ── 부트스트랩(backfill_bge_embeddings 와 동일 순서) ─────────────────────────────
# 1) load_dotenv(.env.{env}, override=False) → 2) init_settings(env) → 3) PostgresUtil() + `with` →
# 4) build_drafts → 5) golden_ko.draft.json 기록. 읽기 전용(SELECT 만).
def main() -> int:
    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    parser = argparse.ArgumentParser(
        description="한국어 골든셋 초안 생성 (017 A/B 하니스용; 주제 기반·사람 검수 전 초안)"
    )
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument(
        "--min-group", type=int, default=2, dest="min_group",
        help="정답군 최소 자산 수(미만 주제 제외, 기본 2)",
    )
    args = parser.parse_args()

    _configure_logging()
    dotenv_path = _REPO_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    db = PostgresUtil()
    with db, db.transaction() as conn:
        drafts = build_drafts(conn, min_group=args.min_group)

    _DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DRAFT_PATH.write_text(
        json.dumps(drafts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _LOG.info("기록: %s (%s주제)", _DRAFT_PATH, len(drafts))
    print(
        f"초안 {len(drafts)}주제 기록 → {_DRAFT_PATH}\n"
        "다음: 사람이 질의 문장·주제 선택을 검수·보정해 "
        "tests/fixtures/search/golden_ko.json 으로 확정하세요(질의 20~50건 권장). "
        "초안은 자동 군집일 뿐, 하니스는 확정본(golden_ko.json)만 사용합니다."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
