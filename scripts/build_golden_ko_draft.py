#!/usr/bin/env python3
"""한국어 골든셋 초안 생성 — 파일명 주제(prefix) 기반 정답군으로 질의·정답 초안을 만든다 (017 G4).

017 A/B(KoSimCSE vs BGE-M3) 하니스(`tests/test_embedding_ab_kpi.py`)는 **사람이 검수·확정한**
한국어 골든셋(`tests/fixtures/search/golden_ko.json`)으로 두 채널 retrieval 품질을 비교한다.

이 데이터는 파일명이 주제별로 자연 군집한다. **같은 주제 = 관련 자산**이므로, 주제로 자산을 묶어
정답군으로 쓰면 "자기 자산 1건"보다 훨씬 의미 있는 골든셋이 된다(검수 부담도 작다 — 주제 그룹이
자동 정답). 이 스크립트는 주제별 그룹을 **결정적**으로 뽑아 ``--out`` 경로(기본
``tests/fixtures/search/golden_ko.draft.json``)에 기록한다.

코퍼스는 **3종 명명 규약**이 혼재한다(2026-06 대량 증분 이후) — 추출기가 셋 다 처리한다:
  - 신규 출처-prefix: ``youtube_<주제>_<영상ID>.<ext>`` / ``wikipedia_<주제>_<문서ID>.<ext>``
    (예: ``youtube_사막_3bTA2c2n2QI.jpg``, ``wikipedia_고려청자_1031019.txt``). 주제 = 2번째
    밑줄 필드(밑줄 없는 단일 토큰, 공백 포함 가능: ``기후 변화``·``우주 탐사``).
  - 구형: ``<주제>_<YouTube ID(11자)>_<제목>.<ext>``(예: ``무선_충전기_7iTajgt8pec_…``).
    주제 = ID 직전 prefix(밑줄 포함: ``등산_입문``).

규칙
  - **주제 추출**(``_extract_topic``): 위 3종 규약에서 (그룹키, 표시주제)를 뽑는다. 구형 ID 는
    ``[A-Za-z0-9_-]`` 11자라 한국어 주제 토큰과 구조적으로 구분된다. 패턴이 아니면 제외(예:
    ``manifest.json``).
  - **출처 교차 병합**: 그룹키 = 주제의 첫 밑줄 토큰. 구형 ``등산_입문``·신규 ``youtube_등산``·
    ``wikipedia_등산`` 이 모두 키 ``등산`` 으로 병합돼 한 질의의 정답이 출처·모달리티를 가로지른다
    (질의 "등산" 의 진짜 정답 = 출처 불문 모든 등산 자산). 신규 주제는 밑줄이 없어 키=주제.
  - **질의**: 그룹 내 가장 서술적인(긴) 원문 주제를 자연어로(밑줄→공백, ``무선_충전기`` → "무선 충전기").
  - **정답**: 그 주제의 registered 자산 전부(2026-07-23: 구 st_bge 채널 요건 제거·현행 단일 채널
    st_api. 도메인 제외 전면 제거로 medical 포함).
  - **오라벨 제외**(``_EXCLUDE_FILES``): 수기 검증된 도메인-무관 위키 파일(파일명 토픽 ≠ 본문 내용,
    예: ``영어 회화`` 파일이 오르세 미술관 기사)은 정답군에서 뺀다. 정답이 1건만 남아 ``min-group``
    미만이 되면 그 주제 질의 자체가 빠진다(오염된 소그룹 질의 제거).
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

from src.config.filename_util import display_file_name  # noqa: E402 — 065 {asset_id}__ 프리픽스 제거

_LOG = logging.getLogger("meta_extract.build_golden_ko_draft")

# 초안 출력 경로(사람이 검수해 golden_ko.json 으로 확정). 초안 파일 자체는 커밋 대상 아님.
_DRAFT_PATH = _REPO_ROOT / "tests" / "fixtures" / "search" / "golden_ko.draft.json"

# 신규 출처-prefix 명명: youtube_/wikipedia_ 다음 필드가 주제(다음 _ 까지). 주제엔 밑줄이 없어
# 공백 포함 단일 토큰으로 잡힌다(``기후 변화``·``우주 탐사``).
_SRC_PREFIXES = ("youtube", "wikipedia")
# 구형 주제 prefix 규칙: <주제(밑줄 포함)>_<YouTube ID 11자>_<제목>.
# YouTube ID = [A-Za-z0-9_-] 11자(밑줄/하이픈 포함 가능). 주제는 비탐욕(.+?)으로 ID 직전까지.
# 한국어 주제 토큰은 ASCII 가 아니므로, ID 직전 11자 ASCII 토큰이 유일하게 매칭된다.
_TOPIC_ID_RE = re.compile(r"^(?P<topic>.+?)_(?P<id>[A-Za-z0-9_-]{11})_")
# 명시 주제 접미(2026-07-23 추가): ``<원본명>_(<주제>).<ext>``(수집 시 부여·미디어/레거시 다수).
# 예: ``골프_(골프).txt``·``…응급처치법_(응급처치).mp3``. youtube_/wikipedia_ 프리픽스보다 신뢰도 높아 1순위.
_TOPIC_SUFFIX_RE = re.compile(r"_\(([^)]+)\)\.[^.]+$")

# 수기 검증된 **도메인-무관 오라벨** 파일(파일명 토픽 ≠ 본문 내용) — 골든 정답에서 제외(2026-06-17).
# 위키 본문 표제어가 토픽과 무관한 사례(예: `영어 회화` 파일이 오르세 미술관 기사). 이런 자산이
# 정답군에 섞이면 recall 을 거짓으로 낮춘다(검색은 올바르게 미반환). 소스 파일을 재라벨/삭제하면
# 이 목록에서 빼면 된다. 인물·하위주제·동의어 기사(농구→서장훈, 초밥→스시 등)는 토픽 관련이라 제외 안 함.
_EXCLUDE_FILES = frozenset({
    "wikipedia_영어 회화_32276.txt",   # 오르세 미술관(박물관)
    "wikipedia_바둑_307905.txt",       # 배우 유오성
    "wikipedia_야구_744753.txt",       # 배우 박서준
    "wikipedia_자수_8613.txt",         # 오리온자리(별자리)
    "wikipedia_기타_110277.txt",       # 정액(생식) — 동음이의 '기타=etc'
    "wikipedia_도예_10011.txt",        # 이란 이슬람 공화국
    "wikipedia_드론_4148426.txt",      # 2026 이란 전쟁
    "wikipedia_요가_587393.txt",       # 레노버 그룹(노트북 'Yoga')
    "wikipedia_종이접기_241748.txt",   # 키노피오(마리오 캐릭터)
    "wikipedia_사막_5736.txt",         # 동아시아(지역)
    "wikipedia_첼로_13319.txt",        # 비올라(다른 악기)
    "wikipedia_김밥_3769997.txt",      # 전주 조직폭력배 범죄사건
    # 경계(주제 아닌 회사/영화/행정구역 기사 — 본문 표제어가 토픽과 다름):
    "wikipedia_전기차_3731649.txt",    # (주)서진시스템(부품사)
    "wikipedia_와인_953218.txt",       # LVMH(명품 그룹)
    "wikipedia_로봇_117998.txt",       # 영화 《A.I.》
    "wikipedia_수채화_1078581.txt",    # 영화 《비 오는 날 수채화》
    "wikipedia_캠핑_34998.txt",        # 성주군(행정구역)
})


def _configure_logging() -> None:
    if _LOG.handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


def _extract_topic(name: str) -> tuple[str, str] | None:
    """파일 basename 에서 ``(그룹키, 표시주제)`` 를 추출한다(3종 규약 대응, 패턴 아니면 None).

    - 신규 ``youtube_<주제>_<id>.<ext>`` / ``wikipedia_<주제>_<id>.<ext>``: 주제 = 2번째 밑줄 필드
      (``youtube_사막_3bTA2c2n2QI.jpg`` → ``사막``). 신규 주제는 밑줄이 없어 그룹키 = 주제 그대로.
    - 구형 ``무선_충전기_7iTajgt8pec_…`` → 표시주제 ``무선_충전기``. ID 가 밑줄을 포함해도
      (``5ncp-_GXBsU``) 11자 ASCII 토큰으로 한 번에 매칭돼 주제가 정확히 잘린다.

    그룹키 = 표시주제의 첫 밑줄 토큰 → 구형 ``등산_입문`` 과 신규 ``등산`` 이 키 ``등산`` 으로 병합
    (출처 교차). 신규 주제(밑줄 없음)는 키 = 주제.
    """
    # 1순위: 명시 주제 접미 ``_(<주제>).<ext>``(수집 시 부여). 프리픽스/구형보다 신뢰도 높음.
    ms = _TOPIC_SUFFIX_RE.search(name)
    if ms and ms.group(1).strip():
        topic = ms.group(1).strip()
        return (topic, topic)
    parts = name.split("_")
    if len(parts) >= 3 and parts[0] in _SRC_PREFIXES:
        topic = parts[1].strip()
        return (topic, topic) if topic else None
    m = _TOPIC_ID_RE.match(name)
    if not m:
        return None
    topic = m.group("topic").strip()
    if not topic:
        return None
    return (topic.split("_", 1)[0], topic)


def _query_from_topic(topic: str) -> str:
    """주제명을 자연어 질의로(밑줄→공백). ``무선_충전기`` → "무선 충전기"."""
    return topic.replace("_", " ").strip()


def _fetch_assets(conn: Any) -> list[dict[str, Any]]:
    """registered 자산의 asset_id·fs_path 를 결정적 순서로 조회(도메인 무관·medical 포함).

    2026-07-23: 구 A/B 하니스의 ``channel='st_bge'`` 요건 제거 — 현행 코퍼스는 단일 활성 채널
    (st_api)만 백필돼 st_bge 조인이 0행이었다. 검색 대상 전체(registered)를 정답 모집단으로 삼는다
    (도메인 제외 전면 제거로 medical 포함). ``ORDER BY a.asset_id`` 로 고정 순서(2회 실행 동일).
    """
    sql = (
        "SELECT a.asset_id, a.fs_path "
        "FROM asset a "
        "WHERE a.status = 'registered' "
        "ORDER BY a.asset_id"
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql)
        return list(cur.fetchall())


def build_drafts(conn: Any, *, min_group: int = 2) -> list[dict[str, Any]]:
    """주제별 그룹으로 골든셋 초안 ``[{"query", "relevant_asset_ids"}]`` 을 만든다(결정적).

    파일명에서 (그룹키, 표시주제)를 추출해 그룹키로 자산을 묶고(출처 교차 병합), 자산 수
    ``min_group`` 미만 그룹은 제외한다. 질의 = 그룹 내 가장 서술적인(긴) 표시주제(밑줄→공백),
    정답 = 그룹 자산 전부(중복 제거·asset_id 정렬). 그룹키 정렬로 결정적.
    """
    rows = _fetch_assets(conn)
    groups: dict[str, set[str]] = {}
    labels: dict[str, set[str]] = {}  # 그룹키 → 본 표시주제들(질의 문구 선정용)
    excluded = 0
    for row in rows:
        # 2026-07-23: registered_dest 가 붙인 ``{asset_id}__`` 프리픽스를 벗겨 원본 파일명으로 추출
        # (프리픽스가 붙으면 split('_')[0] 이 UUID 라 _extract_topic 이 전부 실패했다).
        name = display_file_name(row["fs_path"])
        if name in _EXCLUDE_FILES:  # 수기 검증 오라벨 — 정답군에서 제외
            excluded += 1
            continue
        extracted = _extract_topic(name)
        if extracted is None:
            continue
        key, label = extracted
        groups.setdefault(key, set()).add(str(row["asset_id"]))
        labels.setdefault(key, set()).add(label)

    drafts: list[dict[str, Any]] = []
    skipped = 0
    for key in sorted(groups):
        ids = sorted(groups[key])
        if len(ids) < min_group:
            skipped += 1
            continue
        # 질의 문구 = 가장 서술적인(긴) 표시주제(동률은 정렬로 결정적). 구형 데이터가 있으면
        # ``등산 입문`` 처럼 기존 골든 문구를 보존하고, 신규-only 그룹은 단일 주제(``초콜릿``)가 된다.
        label = sorted(labels[key], key=lambda s: (-len(s), s))[0]
        drafts.append({"query": _query_from_topic(label), "relevant_asset_ids": ids})
    _LOG.info(
        "초안 생성: 주제 %s개 (조회 %s건, 오라벨 제외 %s건, min-group<%s 제외 %s개)",
        len(drafts), len(rows), excluded, min_group, skipped,
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
    parser.add_argument(
        "--out", type=Path, default=_DRAFT_PATH,
        help="출력 경로(기본 golden_ko.draft.json). 확정본 직접 생성 시 golden_ko.json 지정.",
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

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(drafts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _LOG.info("기록: %s (%s주제)", out_path, len(drafts))
    print(
        f"골든셋 {len(drafts)}주제 기록 → {out_path}\n"
        "초안(--out 미지정)은 자동 군집일 뿐 — 사람이 질의 문장·주제 선택을 검수·보정해 "
        "tests/fixtures/search/golden_ko.json 으로 확정하세요. 하니스는 확정본만 사용합니다."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
