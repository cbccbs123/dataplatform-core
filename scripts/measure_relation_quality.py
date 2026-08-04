"""관계 품질 측정 러너 (spec 031 — T007/T008/T009).

서브커맨드:
  curate   부트스트랩 후보(path_signal 쌍 + 고confidence graph_edge)를 surface 해 **검토 초안** 골든을 만든다.
           사람이 이 초안을 편집·검증(잘못된 쌍 제거·kind 확정·고립 추가)해야 골든이 된다(ADR 결정1 — silver 자동채택 금지).
  snapshot 골든 소스마다 후보(union) + LLM 제안 1회를 **동결 스냅샷**(JSON)으로 저장. graph_edge **미기록**(측정 전용·SC-004).
  measure  골든+스냅샷 → 후보recall·관계 P/R·kind·고립·임계스윕 리포트(LLM 0·결정적).

측정 전용 — 어떤 서브커맨드도 graph_edge/relation_kind 에 쓰지 않는다(읽기 + LLM 호출만). 실 DB/LLM 필요.

실행:
  python -m scripts.measure_relation_quality --env dev curate   --out tests/golden/relations/relation_golden.draft.json
  python -m scripts.measure_relation_quality --env dev snapshot --golden <golden.json> --out <snapshot.json>
  python -m scripts.measure_relation_quality --env dev measure  --golden <golden.json> --snapshot <snapshot.json>
  # shadow A/B(079) — 골든 대신 active 엣지 보유 자산 표본으로 두 변형을 각각 동결한다.
  python -m scripts.measure_relation_quality --env dev snapshot \
      --sample-active 200 --seed 20260728 --prompt-variant no-circular-hint --out <snapshot_B.json>
"""
from __future__ import annotations

import json
import posixpath
import random
from collections import Counter
from collections.abc import Callable
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.database.postgres_util import PostgresUtil
from src.relations.prompt import RELATION_CONFIDENCE_GUIDE_KO
from src.relations.quality.golden import Golden, parse_golden, resolve_asset_keys
from src.relations.quality.metrics import isolated_candidates
from src.relations.quality.report import build_report
from src.relations.quality.snapshot import (
    ProposedEdge,
    Snapshot,
    SourceSnapshot,
    dump_snapshot,
    load_snapshot,
)

LlmFn = Callable[[str], dict[str, Any]]

# 🔴 스냅샷 생성은 **반드시 순차**다(1). 병렬로 올리지 마라 — 집계 잡음이 10배로 커진다.
#
# ⚠️ **LLM 생성은 순차로도 완전 재현되지 않는다.** 같은 시드·같은 프롬프트로 20자산을 2회 돌린
# 실측(2026-07-29):
#   · 순차 · 개입안 변형   → 제안 0/20 상이 · 집계 차이 0.0pp
#   · 순차 · **운영 프롬프트** → 제안 **2/20 상이** · 집계 차이 0.3pp   ← 순차도 흔들린다
#   · 동시(6)              → 제안 1/20 상이 · 집계 차이 **3.2pp**      ← 잡음이 10배
# 원인은 LLM 서버의 연속 배칭(continuous batching)이다 — 같은 프롬프트가 다른 배치 구성에 실리면
# 배치 행렬곱의 부동소수점 결합 순서가 달라져 로짓이 미세하게 흔들리고, 긴 생성에서 그 편차가
# 누적돼 토큰 선택이 갈린다. **temperature=0 으로 막을 수 없다**(샘플링이 아니라 로짓이 다르다).
#
# 그래서 순차를 유지하는 근거는 "순차면 동일하다"가 **아니라** "순차가 집계 잡음을 10배 작게
# 유지한다"다. 재현성 기준도 "2회 실행 100% 동일"에서 **"후보 집합 100% 동일 + 집계 잡음이
# 개입 효과의 1/10 이하"** 로 교체됐다 — 근거·전체 표는 `docs/decisions/2026-07-29-llm-determinism-layers.md`
# (앞선 "순차는 결정적, 병렬이 원인" 보고는 표본 1회로 단정한 오보였고 같은 ADR 에서 정정했다).
#
# ⚠️ 판정(`scripts/judge_relations.py`)은 동시 6 이어도 관측상 안정적이다(39건 × 4회 전부 동일).
#    출력이 `verdict`·`why` 두 필드로 짧아 미세 편차가 토큰 선택을 뒤집지 못한다 — 즉 이 제약은
#    **긴 구조적 출력을 생성하는 호출에만** 적용된다.
#
# 대가: 자산당 ~13초라 1,000자산에 약 3.5시간이 걸린다. 그래도 잡음에 파묻힌 측정보다는 낫다.
# 값을 바꾸면 `tests/test_measure_relation_quality.py` 의 봉인 테스트가 실패한다(의도적 장치).
_SNAPSHOT_CONCURRENCY = 1

# 점수 축 v3 — **"선택한 종류의 정의 안에서, 이 관계가 얼마나 뚜렷한가"** (종류 조건부 척도).
# ✅ **2026-07-31 채택** — 운영 상수 `src/relations/prompt.py:RELATION_CONFIDENCE_GUIDE_KO` 로
# 승격(단일 정본). 아래 변형들은 그 상수를 참조한다 — 채택 후 "baseline" 이 곧 이 문구다.
# 측정 요약(같은 200자산·후보 100% 동일·대조군=채택 전 프롬프트):
#   (a) 정의만        : dup 값별 strong 41→73→89% 단조 ✅ · 전체 정확도 79.6→83.2%
#   (b) 정의+keywords : dup 46→72→84% 단조 ✅ · 전체 83.5% · 요약 80자 이하 쌍 76.3→83.4%
#   형식 판정은 둘 다 미달(② same_domain 중간 역전 — 1σ 노이즈 범위 · ③′ 쌍 소실 5.7/6.3%
#   — 92% 이상이 약한 same_domain 꼬리) → 실질 근거로 (b) 채택(사용자 결정).
# v2("confidence-reading")의 실측이 설계 근거다:
#   · same_domain 안에서 완벽 단조(0.3→2.4% / 0.5→10.1% / 0.7→30.6% / 0.9→63.0% strong) — 작동
#   · 🔴 duplicate_near 는 97% 가 0.9 로 붕괴 — v2 의 축("같은 구체적 대상인가")이 dup 을
#     **고른 이유 그 자체**라 선택과 동어반복이 됐다(사용자가 사전에 예측한 결합).
# v3 는 종류마다 척도를 달리해 선택과 분리한다: dup 은 "겹침의 정도"(같은 측면인가),
# same_domain 은 "분야 공유의 좁기", 명시적 3종은 "근거의 명시성".
_CONFIDENCE_GUIDE_PERKIND_KO = RELATION_CONFIDENCE_GUIDE_KO

# shadow A/B 변형 — **운영 프롬프트는 바꾸지 않는다.** 여기 테이블만 갈아끼워 비교하고,
# 통과한 변형만 나중에 운영 상수로 옮긴다(spec 폐기 기준 4항).
PROMPT_VARIANTS: dict[str, dict] = {
    # 대조군 — 현행 운영 프롬프트 그대로. 2026-07-29 채택으로 옛 "no-circular-hint" 변형이
    # 운영 기본값이 됐으므로 baseline 이 곧 그 문구다.
    "baseline": {},
    # 081 `same_domain` 폐기 검증 — **카탈로그에서 종류를 빼면** 무슨 일이 생기나(X팔).
    # ⚠️ 이 변형은 힌트를 바꾸는 것이 아니라 **선택지 자체를 제거**한다. 그래서 프롬프트가 3곳 바뀐다:
    #   ① same_domain 선택지 사라짐(의도) ② 🔴 anti-dup 구분 문구도 함께 사라짐(부수 효과 —
    #      `_build_relation_kind_guide` 가 duplicate_near ∧ same_domain 일 때만 붙인다)
    #   ③ JSON 예시 종류가 same_domain → duplicate_near 로 대체
    # ②가 079 에서 duplicate_near 과대적용을 막은 핵심 장치라, ①과 ② 가 **같은 방향으로 겹친다**
    # (둘 다 duplicate_near 를 부풀린다). v2(힌트만 넓힘)보다 강한 개입이므로 결과를 예단하지 말 것.
    # 실패하면 Y팔(제거 + anti_dup 강제 주입)로 ②의 영향을 분리한다.
    "drop-same-domain": {
        "catalog_exclude_kinds": ("same_domain",),
    },
    # Y팔 — X 미달 원인이 ①(선택지 제거)인지 ②(anti-dup 문구 동반 소실)인지 **분리**한다.
    # X 는 둘을 함께 바꿨고 dup 이 24.2%→72.2% 로 폭증했다(기준 ①·② 미달).
    # Y 는 선택지만 빼고 **억제 문구를 강제로 살린다** — 통과하면 "폐기는 하되 문구를 살려야 한다"는
    # 뜻이고, `_build_relation_kind_guide` 의 조건(same_domain 이 있을 때만 붙임)을 고치는 근거가 된다.
    #
    # ⚠️ 문구를 그대로 쓸 수 없다 — 현행 문구 끝이 "…분야만 같으면 ``same_domain`` 이다" 로
    # **없어진 종류를 가리킨다.** 존재하지 않는 코드를 지시하면 LLM 이 그 이름으로 출력할 위험이 있고
    # (그러면 영속화 단계에서 미등록 kind 로 버려진다) 지시 자체가 모순이다. 그래서 뒷부분을
    # **"관계를 만들지 않는다"** 로 바꿔, 갈 곳 없는 쌍을 duplicate_near 로 밀지 말고 **기권**하도록
    # 명시한다. 앞부분(대상이 다르면 duplicate_near 가 아니다)은 079 문구를 그대로 유지한다.
    "drop-same-domain-keep-antidup": {
        "catalog_exclude_kinds": ("same_domain",),
        "anti_dup_override": (
            "\n\n**구분:** 주제·세부주제가 같아도 **다루는 대상이 다르면** "
            "``duplicate_near`` 가 아니다. 대상이 다르고 분야만 같을 뿐이면 "
            "**어떤 관계도 만들지 말고 그 후보를 건너뛴다.**"
        ),
    },
    # ── 2026-07-31 · `confidence` 에 **기준을 정의**한다 ────────────────────────────────
    # 왜: 현행 프롬프트에는 confidence 지시가 **한 줄도 없다**(출력 예시의 ``0.75`` 가 유일한
    # 단서). 그 결과 값이 자기보고 감각치가 됐다 — 실측(라벨 2,092건):
    #   · 서로 다른 값 15개뿐 · 0.85/0.80/0.90/0.95 에 82% 집중 · 최저 0.6(3건)
    #   · 이름표 정확도와의 분리력 **−9.9pp**(높은 점수가 오히려 덜 정확)
    #   · 대안으로 시험한 임베딩 유사도도 −6.7pp — **계산값으로 바꿔도 안 된다**
    # 그래서 임계·노출·큐 정책을 이 값 위에 세울 수 없다(그 위에 쌓았던 설계는 전부 잠정 철회).
    #
    # 설계 원리: 점수를 **"근거 정보가 얼마나 충분했나"** 로 정의한다 — 관계의 강약이 아니다.
    #   ⚠️ 강약을 점수로 옮기면 ``relation_type_code`` 의 복사본이 되어 **새 정보가 0**이다
    #      (1차 초안이 그 함정에 빠졌다). 그래서 "약한 관계라도 근거가 확실하면 높은 값" 을
    #      문구에 못 박고, ``same_domain`` + ``0.9`` 예시를 함께 준다.
    #   LLM 이 실제로 보는 재료가 양쪽 요약(소스 1200자·후보 500자)·파일명·매체타입·주제뿐이므로,
    #   구간 조건도 그 재료의 상태로만 서술한다(사람이 요약을 열어 **검증 가능**해야 한다).
    #
    # 값을 4개로 이산화하는 이유: LLM 이 이미 4~6개 값만 쓰고 있어 잃는 정밀도가 없고, 대신
    # 각 값의 뜻이 정해져 사후 검증과 정책 수립이 가능해진다.
    #
    # ⚠️ `same_series` 문구 통일(발견 5)은 **이 변형에 섞지 않는다** — 어제 X팔이 두 변경을 함께
    # 넣어 원인 분리에 실패했다(dup 24.2%→72.2%). 성질이 다른 변경은 따로 잰다.
    # ── 2026-08-03 · 명시적 3종(references·derived_from·same_series) 정의 문구 통일 ──────────
    # 왜: 전량 재생성 실측에서 이 3종의 이름표 정확도가 25~33% 로 무너졌다(합계 16건).
    #   references 9건 33.3% · derived_from 4건 25.0% · same_series 3건 33.3%
    #   실례: 창덕궁↔덕수궁을 "연작"(조선 궁궐 라인업) · 첼로↔바이올린을 "연작"(바이올린족 라인업)
    #        · 스위스 고르너그라트↔울릉도를 "인용".
    # 이 3종은 **점수 하한을 면제**받는 특혜(`approval_policy.SIMILARITY_KINDS` 분기)까지 갖는데,
    # 근거가 "저신뢰여도 정보량이 있다"였다. 실측이 그 전제를 부정했으므로 정의를 좁힌다.
    #
    # 원인 진단: 종류 정의가 프롬프트 **세 곳**에 흩어져 있고 서로 어긋난다.
    #   ① 엣지케이스 안내(`prompt.py` 모듈 docstring) — "같은 stem + 순번/버전" (엄격·정확)
    #   ② 선택 힌트(`RELATION_KIND_HINTS_KO`) — "브랜드 **라인업** 등 연속·묶음" (느슨)
    #   ③ DB `relation_kind.description` — "같은 시리즈·연작·**라인업** 연결" (느슨)
    # "라인업"이 LLM 에게 *"같은 범주에 속하는 것들"* 로 읽혀 궁궐·현악기·기후지도를 묶었다.
    # → ②③ 을 ① 에 맞춰 **파일명·명시 근거 요건**으로 좁힌다(①은 이미 옳아 손대지 않는다).
    #
    # ⚠️ 점수 기준(v3)과 **섞지 않는다** — 점수 기준은 "고른 뒤 얼마나 확실한가"이고 이건
    #   "무엇을 고르는가"다. 축이 달라 함께 바꾸면 원인 분리가 안 된다(07-30 X팔의 실패).
    # 2026-08-03 · 위 문구 통일 + **소스 파일명 공급**. 1차 측정(무작위 200자산)에서 명시적 3종이
    # 4→1건으로 줄었으나 **표본이 작아 판정 불가**였고, 그 원인으로 소스 파일명 부재를 발견했다:
    # 후보는 `filename` 을 받는데 소스는 못 받아 "같은 어간 + 순번" 비교가 원리상 불가능했다.
    # 재료를 공급하는 것이라 문구 변경과 같은 변경의 일부로 본다(별 팔로 쪼개지 않는다).
    # ⚠️ 이 팔은 **표적 표본**(경로 신호 후보를 가진 자산)으로 재야 한다 — 무작위 200자산은
    # 명시적 3종 기대값이 2.3건뿐이라(전량 1,398자산에 16건) 애초에 판정력이 없는 설계였다.
    "explicit-kinds-tightened-srcname": {
        "include_source_filename": True,
        # 문구는 아래 "explicit-kinds-tightened" 와 동일하게 유지한다 —
        # 사후에 `_inherit_tightened()` 가 채운다(딕셔너리 리터럴에서 서로 참조할 수 없어서).
    },
    "explicit-kinds-tightened": {
        "kind_hints_override": {
            "same_series": (
                "**파일명이 같은 어간 + 순번/버전**일 때만 "
                "(``강의_1부``/``강의_2부`` · ``manual_v1``/``manual_v2``). "
                "같은 범주·분야에 속한다는 이유로 고르지 말 것 — 그것은 ``same_domain`` 이다"
            ),
            "references": (
                "한쪽이 다른쪽을 **명시적으로 가리킬 때만** — 제목·파일명·본문에 상대의 이름이 "
                "실제로 등장해야 한다. 주제가 겹친다는 이유로 고르지 말 것"
            ),
            "derived_from": (
                "한쪽이 다른쪽에서 **생성됐음이 드러날 때만** — 원문→요약/번역/전사, "
                "``report``→``report_summary`` 처럼 파생 관계가 파일명이나 내용에 나타나야 한다. "
                "같은 대상을 다룬다는 이유로 고르지 말 것 — 그것은 ``duplicate_near`` 다"
            ),
        },
        "catalog_description_override": {
            "same_series": "파일명이 같은 어간 + 순번/버전인 연작(범주 묶음이 아니다)",
            "references": "한쪽이 다른쪽을 명시적으로 가리키는 인용·링크·제목 참조",
            "derived_from": "한쪽이 다른쪽에서 생성된 파생(원문→요약·번역·전사)",
        },
    },
    # ✅ v3 (a)팔 — **측정 후 채택(2026-07-31)**. 이제 운영 기본과 동일하므로 no-op 변형이다.
    # 채택 전(무기준) 프롬프트를 재현하려면 {"confidence_guide_override": ""} 를 쓴다.
    "confidence-perkind": {
        "confidence_guide_override": _CONFIDENCE_GUIDE_PERKIND_KO,
    },
    # ✅ v3 (b)팔 — (a) + **keywords 보강** · **측정 후 채택(2026-07-31)** — 운영은 후보 경로
    # (`asset_candidates`·`path_signal`)가 keywords 를 직접 싣게 배선돼 이 키도 이제 no-op 다.
    # keywords 채택 근거: 요약 80자 이하 쌍 정확도 76.3→83.4%(+7.1pp). about 은 keywords
    # 부분집합이라 제외, labels 는 저점수 일반명사(노이즈)라 제외 — 실물 확인.
    "confidence-perkind-kw": {
        "confidence_guide_override": _CONFIDENCE_GUIDE_PERKIND_KO,
        "include_keywords": True,
    },
    # ── 2026-07-31 · 점수 축 v2 — **"두 요약을 읽고 판단한 내용 연결의 정도"** ──────────────
    # 🟡 **측정 완료(2026-07-31) — 부분 성공·미달.** 값 분포는 처음으로 퍼졌고(29/26/28/17%)
    # same_domain 안에서 완벽 단조(strong 비율 2.4→10.1→30.6→63.0%)를 얻었으나,
    # duplicate_near 가 97% 0.9 로 붕괴 — 이 축("같은 구체적 대상인가")이 dup 선택 기준과
    # 동어반복이기 때문(기준 ② 미달). dup −21.9% 는 손상이 아니라 자기 교정이었다(쌍 소실 0 ·
    # 약한 dup 53건을 same_domain 으로 강등 · dup 정확도 73.0→80.4%). v3(종류 조건부 척도)가 후속.
    # v1("confidence-defined" · 근거 정보 충분성)은 자기평가라 실패했다 — 값이 0.9(93%)/0.7 로
    # 붕괴하고, 값별 정확도도 역방향(0.7 이 83.3% > 0.9 의 79.8%). 기준 ①(단일값 <70%) 미달.
    # v2 는 자기 상태가 아니라 **쌍의 내용**을 판단시킨다: 읽어야만 답할 수 있는 질문
    # ("두 요약이 같은 구체적 대상을 다루나")이라 코드로 대체 불가·자기평가 아님.
    # 판정 루브릭(RUBRIC_KO_V1)과 같은 축이므로 기존 라벨 2,092건으로 **추가 판정 없이** 검증된다.
    "confidence-reading": {
        "confidence_guide_override": (
            "\n- ``confidence``: 자기 확신이 아니다. **두 요약을 실제로 읽고, 내용이 얼마나 "
            "맞물리는지**를 판단해 적는다. 먼저 ``reason`` 에 연결 근거를 요약 속 내용으로 쓰고, "
            "그 근거에 맞는 값을 고른다 — 근거가 먼저, 점수가 나중이다. "
            "아래 네 값 중 하나만 쓴다(중간값 금지).\n"
            "  - ``0.9`` 두 요약이 **같은 구체적 대상**(같은 장소·인물·사건·작품·제품·요리)을 "
            "중심으로 다룬다. 표기가 달라도 읽으면 같은 대상임을 알 수 있으면 포함한다. "
            "예: \"그랜드캐니언 스카이워크 전망대\" ↔ \"콜로라도강 협곡의 관광 명소\".\n"
            "  - ``0.7`` 중심 대상은 다르지만, **한쪽 내용의 상당 부분이 다른 쪽에서도 실제로 "
            "다뤄진다**. 같은 사건의 배경, 인물과 그 업적, 원본과 해설. "
            "예: 세종대왕의 애민 정책 오디오 ↔ 한글 창제 과정 문서. "
            "(반례: 같은 \"조선 시대\"라는 배경만 공유하면 0.7 이 아니라 0.3 이다.)\n"
            "  - ``0.5`` 내용 겹침은 없지만 **같은 활동·같은 목적**을 다룬다. "
            "예: 김치 담그는 법 영상 ↔ 배추 절이는 법 문서 (둘 다 김장이라는 활동).\n"
            "  - ``0.3`` 읽어 보면 **분야 표딱지 외에 공통점이 없다**. "
            "예: 김치 영상 ↔ 인삼 영상 (발효식품이라는 범주만 공유).\n"
            "  ⚠️ 관계 종류와 **별개로** 판단한다 — ``same_domain`` 을 골랐어도 읽어 보니 같은 "
            "대상이면 0.9 다. 종류에 점수를 맞추지 말 것.\n"
            "  ⚠️ 근거는 **요약에 실제로 있는 내용**이어야 한다. 요약에 없는 배경지식으로 점수를 "
            "올리지 말 것."
        ),
    },
    # 🔴 **검증 후 미달(2026-07-31)** — 점수 축 v1: "근거 정보가 얼마나 충분했나"(자기평가).
    # 값 준수 100%·종류 독립은 달성했으나 0.9 에 93.4% 붕괴 + 값별 정확도 역방향으로 게이트 불능.
    # 교훈: **기준을 정확히 줘도 자기보고는 못 쓴다** — LLM 은 자기 판단의 정보 부족을 인정하지
    # 않는다(하위 2단계 사용 0건). 재시도하지 말 것. v2("confidence-reading")가 후속.
    "confidence-defined": {
        "confidence_guide_override": (
            "\n- ``confidence``: **관계 종류를 무엇으로 골랐는지와 무관하게**, 그 판단을 내릴 "
            "근거 정보가 얼마나 충분했는지를 나타낸다.\n"
            "  ⚠️ 관계가 강한지 약한지는 ``relation_type_code`` 가 이미 표현한다 — 여기서 다시 "
            "표현하지 말 것. **약한 관계라도 근거가 확실하면 높은 값을 쓴다.**\n"
            "  아래 네 값 중 하나를 **그대로** 적는다(중간값·다른 값 금지).\n"
            "  - ``0.9`` 양쪽 요약에 판단에 필요한 내용이 다 있었다. 추측하지 않았다.\n"
            "  - ``0.7`` 한쪽 요약이 짧거나 일반적이어서, 판단의 일부를 주제·파일명으로 메웠다.\n"
            "  - ``0.5`` 요약이 없거나 무내용(로고·검은 화면·자동 생성 문구 등)이어서 "
            "파일명·주제·매체타입만으로 추정했다.\n"
            "  - ``0.3`` 후보 정보가 거의 없어 판단 자체가 불확실하다.\n"
            "  - 예: 양쪽 요약이 구체적이고 서로 **다른 대상**이면 → ``same_domain`` + ``0.9`` "
            "(약한 관계지만 근거는 확실하다). 후보가 로고 이미지라 요약이 한 줄이면 → ``0.5``."
        ),
    },
    # 🔴 **검증 후 폐기됨(2026-07-30)** — 재시도하지 말 것. 보고서 §9.1 에 전체 근거.
    # 착안: 채택본의 "거의 같은 형식" 조건이 DB 정의보다 좁아 매체만 다른 같은-대상 쌍을
    # same_domain 으로 밀어냄(골든 실증). 형식 조건을 제거하면 그 오분류가 실제로 고쳐진다
    # (골든 kind_acc 61.2→79.7% · 모집단에서 same_domain 오배치 strong 59→14건).
    # 폐기 이유: **자동승인 정밀도가 82.1→66.5% 로 무너진다**(`specs/081-relation-approval-exposure/spec.md` 가 쓰는 결정 지표) —
    # duplicate_near 가 331쌍으로 넓어지며 weak 157건을 함께 삼킨다. 명시적 관계도 27→3쌍 붕괴.
    # 보존 이유: "형식 조건 제거"는 직관적으로 옳아 보여 재시도 유혹이 큰 안이다. 측정 결과를
    # 코드에 남겨 같은 길을 두 번 가지 않게 한다.
    "dup-same-subject-v2-discarded": {
        "kind_hints_override": {
            "duplicate_near": "**같은 구체적 대상**을 다룰 때 — 매체·형식이 달라도 대상이 같으면 해당",
        },
    },
    # 채택 이전의 운영 문구(순환 지시 포함) — §4 shadow A/B 를 현재 코드에서 재현하거나
    # 회귀 비교할 때만 쓴다. 순환 지시: 후보는 정의상 전부 임베딩 유사도로 온 것이라
    # "임베딩 유사도로 가져온 후보처럼" 힌트는 "전 후보에 duplicate_near"가 된다.
    "circular-hint-legacy": {
        "kind_hints_override": {
            "duplicate_near": "임베딩 유사도로 가져온 후보처럼 **내용·장면·주제 근접**할 때",
            "same_domain": "같은 주제·분야·도메인으로 묶일 때(예: 둘 다 게임, 둘 다 교통)",
        },
        "anti_dup_override": (
            "\n\n**구분:** 단순히 주제가 같으면 ``same_domain`` , "
            "유사도·근접 후보라면 ``duplicate_near`` 를 우선 고려한다."
        ),
    },
}

# srcname 팔은 문구 통일분을 **그대로** 물려받아야 한다 — 문구가 갈라지면 "소스 파일명 공급"의
# 효과가 아니라 "문구 차이"를 재게 된다. 딕셔너리 리터럴 안에서는 서로 참조할 수 없어 여기서 합친다.
PROMPT_VARIANTS["explicit-kinds-tightened-srcname"] = {
    **PROMPT_VARIANTS["explicit-kinds-tightened"],
    **PROMPT_VARIANTS["explicit-kinds-tightened-srcname"],
}


# ── 읽기 헬퍼(graph 무기록) ──────────────────────────────────────────────────
def _source_summary_modality(conn: Connection[Any], asset_id: str) -> tuple[str, str]:
    """소스 자산의 (요약, modality). 프롬프트 구성용 — 읽기 전용."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT a.modality, COALESCE(m.ext_meta->>'summary', '') AS summary "
            "FROM asset a LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id "
            "WHERE a.asset_id = %s LIMIT 1",
            (asset_id,),
        )
        r = cur.fetchone()
    if not r:
        return "", ""
    return str(r["summary"] or ""), str(r["modality"] or "")


def sample_active_sources(db: PostgresUtil, *, n: int, seed: int) -> list[str]:
    """active 엣지를 가진 자산에서 시드 고정으로 ``n`` 건을 뽑는다(읽기 전용).

    골든이 아니라 표본으로 스냅샷을 뜨는 이유: A/B 는 구·신 **상대 비교**라 정답셋이 필요 없다.
    골든 재스냅샷은 머지의 선행조건이지 실험의 선행조건이 아니다(ADR 결정 7).

    Args:
        db: DB 핸들.
        n: 뽑을 자산 수.
        seed: 난수 시드.

    Returns:
        정렬된 자산 id 목록. 풀이 ``n`` 보다 작으면 전수.

    Raises:
        ValueError: active 엣지를 가진 자산이 하나도 없을 때. 조용히 빈 스냅샷을 만들면
            A/B 가 "차이 없음"으로 보이는데 실은 아무것도 재지 않은 것이다.
    """
    # node_kind='asset' 가드는 레포 관례(graph_query·review·asset_topic_query 동일) —
    # entity 노드는 asset_id 가 NULL 이라 빼지 않으면 None 이 소스 id 로 섞인다.
    sql = """
        SELECT DISTINCT nd.asset_id::text AS asset_id
        FROM graph_edge ge
        JOIN node nd ON nd.node_id IN (ge.src_node, ge.dst_node) AND nd.node_kind = 'asset'
        WHERE ge.status = 'active'
        ORDER BY 1
    """  # 대칭 엣지는 행이 하나라 양끝을 IN 으로 함께 본다(한쪽만 보면 절반이 빠진다).
    with db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql)
        ids = [r["asset_id"] for r in cur.fetchall()]
    if not ids:
        raise ValueError(
            "active 엣지를 가진 자산이 없다 — 표본을 뽑을 수 없다. "
            "관계 생성이 한 번도 돌지 않았거나 잘못된 DB 를 보고 있는지 확인하라.")
    if len(ids) < n:
        print(f"⚠️ 표본 풀 {len(ids)}건 < 요청 {n}건 — 전수를 쓴다. "
              f"비율만 보고하지 말고 실제 n 을 함께 적어라.", flush=True)
    # 정렬된 모집단 + 고정 시드 = 같은 시드면 항상 같은 표본(SC-002 재현성).
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def _read_candidates_prompt(
    conn: Connection[Any], sid: str, cfg: Any, config: dict,
    *, prompt_variant: dict | None = None,
) -> tuple[list, str]:
    """소스 sid 의 후보(union)와 LLM 프롬프트를 만든다(읽기 전용). LLM 호출은 호출자가 트랜잭션 밖에서.

    Args:
        sid: 소스 자산 id — 이 자산을 기준으로 후보를 모은다.
        prompt_variant: shadow A/B 프롬프트 변형 정의(``PROMPT_VARIANTS`` 의 한 항목).
            ``build_relation_proposal_prompt`` 의 override 인자로 **그대로 풀어 넣는다**.
            ``None`` 이나 빈 dict(=``baseline``)이면 운영과 바이트 동일한 프롬프트다.

    Returns:
        ``(후보 목록, LLM 프롬프트)``.
    """
    from src.relations.asset_candidates import find_embedding_candidates
    from src.relations.asset_entry import union_candidates
    from src.relations.path_signal import find_path_signal_candidates
    from src.relations.prompt import build_relation_proposal_prompt
    from src.relations.relation_type_catalog import fetch_active_relation_kinds

    emb = find_embedding_candidates(
        conn, source_asset_id=sid, top_k=config["top_k"],
        embedding_kind=config["embedding_kind"], min_sim=config["min_sim"],
    )
    path = find_path_signal_candidates(conn, source_asset_id=sid, limit=cfg.relations.path_top_k)
    cands = union_candidates(emb, path)
    summary, modality = _source_summary_modality(conn, sid)
    kinds = fetch_active_relation_kinds(conn)
    variant = dict(prompt_variant or {})
    # `catalog_exclude_kinds` 는 override 인자가 아니라 **카탈로그 자체를 줄이는** 측정 전용 키다.
    # 종류를 빼면 프롬프트가 3곳 바뀐다(선택지·anti-dup 구분 문구·JSON 예시) — 힌트 교체와 성격이
    # 다르므로 build_relation_proposal_prompt 로 그대로 넘기지 않고 여기서 카탈로그를 걸러 낸다.
    drop = {str(k).strip().lower() for k in variant.pop("catalog_exclude_kinds", ())}
    if drop:
        kinds = [k for k in kinds if str(k.get("type_code", "")).lower() not in drop]
    # `catalog_description_override` — DB `relation_kind.description` 을 **측정용으로만** 갈아끼운다.
    # 종류 정의 문구는 프롬프트 세 곳(엣지케이스 안내·선택 힌트·카탈로그 description)에 흩어져 있고
    # 그중 description 은 DB 에서 온다. 문구 통일 효과를 재려면 세 곳을 함께 바꿔야 하므로
    # 이 키가 필요하다(DB 를 건드리지 않고 프롬프트 입력만 교체).
    desc = variant.pop("catalog_description_override", None) or {}
    if desc:
        kinds = [{**k, "description": desc.get(str(k.get("type_code", "")), k.get("description"))}
                 for k in kinds]
    # `include_keywords` 도 측정 전용 키 — 소스·후보의 keywords 를 DB 에서 읽어 프롬프트에 싣는다.
    # 운영 후보 경로는 keywords 를 만들지 않으므로 이 키가 없으면 프롬프트는 기존과 바이트 동일.
    # `include_source_filename` — 소스 파일명을 프롬프트에 싣는다(측정 전용).
    # 후보는 `filename` 을 받는데 소스는 못 받아 **양쪽 파일명 비교가 불가능**했다 —
    # same_series·references·derived_from 정의가 그 비교를 전제하므로 재료 공급이 필요하다.
    if variant.pop("include_source_filename", False):
        with conn.cursor() as cur:
            cur.execute("SELECT fs_path FROM asset WHERE asset_id = %s", (sid,))
            row = cur.fetchone()
        variant["source_filename"] = posixpath.basename(str(row[0] or "")) if row else None
    if variant.pop("include_keywords", False):
        ids = [str(c["id"]) for c in cands] + [str(sid)]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT asset_id::text, COALESCE(ext_meta->>'keywords', '')"
                " FROM asset_metadata WHERE asset_id = ANY(%s::uuid[])",
                (ids,),
            )
            kw_map = dict(cur.fetchall())
        for c in cands:
            c["keywords"] = kw_map.get(str(c["id"]), "")
        variant["source_keywords"] = kw_map.get(str(sid)) or None
    prompt = build_relation_proposal_prompt(
        source_summary=summary, source_media_type=modality,
        candidates=cands, relation_kinds_catalog=kinds,
        **variant)
    return cands, prompt


def build_snapshot(
    db: PostgresUtil, golden: Golden | None = None, *, config: dict,
    llm_fn: LlmFn | None = None, source_ids: list[str] | None = None,
    prompt_variant: dict | None = None,
) -> tuple[Snapshot, dict[str, str], list[str]]:
    """소스마다 후보 union + 제안(llm_fn 또는 실 LLM)을 모아 (Snapshot, key_to_id, missing) 반환.

    ⚠️ **아무것도 저장하지 않는다** — 측정 전용이라 엣지·관계 어휘를 기록하지 않는다.
    ⚠️ LLM 호출은 **트랜잭션 밖**에서 한다 — 느린 호출이 커넥션을 붙잡으면 다른 작업이 밀린다.

    Args:
        db: DB 핸들.
        golden: 정답 묶음. 주면 **여기서 소스를 도출한다**(쌍 양끝 + 고립). ``None`` 이면
            ``source_ids`` 로 소스를 받는다.
        config: 후보 조회 설정(임계·상한 등).
        llm_fn: 제안 함수. **바꿔 끼울 수 있게 열어 뒀다** — 실제 LLM 없이 고정 응답으로
            측정 배선을 검증한다. ``None``(기본)이면 실 LLM(``propose_edges_json``).
        source_ids: 소스 자산 id 목록. **골든 없이** 표본으로 스냅샷을 뜰 때 쓴다
            (구·신 프롬프트 A/B 는 상대 비교라 정답셋이 필요 없다). ``None`` 이면 ``golden`` 경로.
        prompt_variant: 프롬프트 변형 정의. ``None``(기본)이면 운영과 동일한 프롬프트를 쓴다.

    Returns:
        ``(스냅샷, 골든 키→자산 id 매핑, 해소 못 한 키 목록)``.
        ``source_ids`` 경로에서는 골든이 없으므로 매핑·미해소가 빈 값이다.

    Raises:
        ValueError: ``golden`` 과 ``source_ids`` 를 둘 다 주거나 둘 다 안 줬을 때.
    """
    from src.relations.asset_entry import target_emb_score_map
    from src.relations.llm_propose import parse_and_normalize_edges, propose_edges_json

    # 소스가 어디서 오는지 모호하면 "무엇을 측정했는가"가 흐려진다 — DB 를 건드리기 전에 막는다.
    if (golden is None) == (source_ids is None):
        raise ValueError("golden 과 source_ids 중 **정확히 하나**를 준다")

    fn: LlmFn = llm_fn if llm_fn is not None else propose_edges_json
    # 소스별 제안 실패를 모은다 — 조용히 넘기면 "제안이 없는 자산"과 구분되지 않는다.
    _failures: list[tuple[str, str]] = []
    if golden is not None:
        with db.transaction() as conn:
            mapping, missing = resolve_asset_keys(conn, golden)
        source_keys = {k for p in golden.pairs for k in (p.a, p.b)} | set(golden.isolated)
        sids = sorted({mapping[k] for k in source_keys if k in mapping})
    else:
        # 표본 경로 — 정답셋이 없으니 해소할 키도 없다(매핑·미해소는 빈 값).
        mapping, missing, sids = {}, [], sorted(source_ids or [])

    def _one(sid: str) -> tuple[str, SourceSnapshot]:
        """소스 하나를 동결한다 — 다른 소스와 완전히 독립이라 병렬 실행이 안전하다."""
        with db.transaction() as conn:  # 짧은 읽기 트랜잭션 — 후보·프롬프트만
            cands, prompt = _read_candidates_prompt(
                conn, sid, _settings(), config, prompt_variant=prompt_variant)
        # 033 FR-006: 후보의 {id: emb_score} 맵을 동결해 제안 엣지에 부착(2D 자동승인 스윕/AND 게이트가 참조).
        # path-only 후보·후보 맵 밖 타깃(LLM 환각)은 0.0 sentinel — target_emb_score_map 이 union 후보 그대로 보존.
        emb_map = target_emb_score_map(cands)
        # ★ LLM(또는 주입) — 트랜잭션 밖.
        # ⚠️ **한 소스의 실패가 전체 실행을 죽이지 않게 흡수한다.** 450자산 순차 실행이 90분인데,
        #    LLM 응답 하나가 파싱 불가하면(빈 응답·스키마 불능) 그 시간이 통째로 날아간다.
        #    실제로 겪었다 — `RelationProposalParseError` 한 건으로 A팔이 소멸했고, 부분 결과도
        #    남지 않아 재시작밖에 없었다. 판정 러너(`judge_relations.judge_one`)는 처음부터
        #    건별로 흡수했는데 여기만 빠져 있었다.
        #    실패한 소스는 **제안 0건으로 기록**한다 — 스냅샷에서 사라지면 "후보는 있었는데 제안이
        #    없었다"와 "실행이 실패했다"를 구분할 수 없다. `_failures` 로 별도 집계해 보고한다.
        try:
            raw = fn(prompt)
            edges = parse_and_normalize_edges(raw)
        except Exception as exc:  # noqa: BLE001 — 한 건 실패가 측정 전체를 멈추지 않게
            print(f"⚠️ 제안 실패(소스 {sid}): {type(exc).__name__} — 제안 0건으로 기록하고 계속한다.",
                  flush=True)
            _failures.append((sid, type(exc).__name__))
            edges = []
        return sid, SourceSnapshot(
            # 033 FR-004: 후보를 (id, emb_score) 로 동결 → N1 min_sim 스윕이 후보 단계 recall 을
            # 점수 임계로 재측정(전체 후보 기준 — proposed 부분집합 아님). path-only=0.0 그대로.
            candidates=tuple((str(c["id"]), float(c["emb_score"])) for c in cands),
            proposed=tuple(
                ProposedEdge(
                    target=str(e["target_media_item_id"]),
                    kind=str(e.get("relation_type_code") or ""),
                    confidence=float(e.get("confidence") or 0.0),
                    topic_ko=str(e.get("topic_ko") or ""),
                    emb_score=emb_map.get(str(e["target_media_item_id"]), 0.0),
                )
                for e in edges
            ),
        )

    # 소스를 **하나씩 순차로** 처리한다(위 `_SNAPSHOT_CONCURRENCY` 주석의 근거).
    # 병렬 분기를 남겨두지 않는 이유: 상한이 1 이라 그 분기는 도달 불가능한 죽은 코드였고,
    # 죽은 병렬 경로는 "올려도 되나 보다"는 오해를 부른다. 되살릴 땐 위 ADR 을 먼저 읽어라.
    sources = dict(_one(sid) for sid in sids)
    if _failures:
        rate = len(_failures) / max(1, len(sids))
        kinds = Counter(k for _, k in _failures)
        print(f"⚠️ 제안 실패 {len(_failures)}/{len(sids)}건 ({100 * rate:.1f}%) — {dict(kinds)}",
              flush=True)
        # 실패가 많으면 그 스냅샷으로 A/B 를 하면 안 된다 — "개입 효과"와 "실패 분포 차이"가
        # 섞여 구분되지 않는다. 임계는 판정 러너의 error 게이트(5%)와 같이 둔다.
        if rate > 0.05:
            print(f"🔴 실패율 {100 * rate:.1f}% > 5% — 이 스냅샷은 A/B 비교에 쓰지 말 것"
                  f"(개입 효과와 실패 분포가 섞인다).", flush=True)
    # config 에 실패 요약을 남긴다 — 스냅샷 파일만 보고도 신뢰도를 판단할 수 있어야 한다.
    config_out = {**config, "propose_failures": len(_failures), "sources_total": len(sids)}
    return Snapshot(config=config_out, sources=sources), mapping, missing


def _settings() -> Any:
    """설정을 초기화해 돌려준다(측정 스크립트는 운영과 같은 설정을 써야 한다)."""
    from src.config.settings import get_current_settings
    return get_current_settings()


# ── curate: 부트스트랩 후보 surface → 검토 초안 골든 ──────────────────────────
def _bootstrap_candidate_pairs(conn: Connection[Any], *, edge_conf_min: float) -> list[dict]:
    """검토용 후보 쌍: ① 동일 폴더·stem path_signal 쌍 ② confidence≥임계 graph_edge 쌍. (asset_id·fs_path·_source·_suggest_kind)"""
    from src.relations.path_signal import find_path_signal_candidates

    seen: set[frozenset] = set()
    out: list[dict] = []
    with conn.cursor(row_factory=dict_row) as cur:
        # ① 고confidence graph_edge → kind 제안과 함께(엣지 자체가 kind 보유).
        cur.execute(
            "SELECT na.asset_id AS a, nb.asset_id AS b, rk.kind_code AS kind, ge.confidence AS conf "
            "FROM graph_edge ge "
            "JOIN node na ON na.node_id = ge.src_node AND na.node_kind = 'asset' "
            "JOIN node nb ON nb.node_id = ge.dst_node AND nb.node_kind = 'asset' "
            "JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id "
            "WHERE ge.confidence >= %s ORDER BY ge.confidence DESC, ge.edge_id ASC",
            (edge_conf_min,),
        )
        for r in cur.fetchall():
            key = frozenset((str(r["a"]), str(r["b"])))
            if key in seen:
                continue
            seen.add(key)
            out.append({"a": str(r["a"]), "b": str(r["b"]), "_source": "edge",
                        "_suggest_kind": str(r["kind"]), "_conf": float(r["conf"] or 0.0)})
        # ② path_signal 쌍(registered 자산 순회·동일폴더+stem). kind 는 사람이 확정(제안 미상).
        cur.execute("SELECT asset_id FROM asset WHERE status = 'registered' ORDER BY asset_id")
        reg_ids = [str(r["asset_id"]) for r in cur.fetchall()]
    for sid in reg_ids:
        for c in find_path_signal_candidates(conn, source_asset_id=sid, limit=10):
            key = frozenset((sid, str(c["id"])))
            if key in seen:
                continue
            seen.add(key)
            out.append({"a": sid, "b": str(c["id"]), "_source": "path_signal", "_suggest_kind": ""})
    return out


def _asset_fs_path(conn: Connection[Any], ids: set[str]) -> dict[str, str]:
    """자산 id 를 경로로 바꿀 매핑을 **한 번에** 조회한다(리포트에 이름을 붙이려고).

    Args:
        conn: DB 연결.
        ids: 조회할 자산 id 집합. **비어 있으면 DB 를 건드리지 않는다**.

    Returns:
        ``{asset_id: 경로}``.
    """
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT asset_id, fs_path FROM asset WHERE asset_id = ANY(%s)", (list(ids),))
        return {str(a): str(p) for a, p in cur.fetchall()}


def _registered_asset_ids(conn: Connection[Any]) -> list[str]:
    """registered 자산 id 전체(정렬) — 고립 후보 모집단."""
    with conn.cursor() as cur:
        cur.execute("SELECT asset_id FROM asset WHERE status = 'registered' ORDER BY asset_id")
        return [str(r[0]) for r in cur.fetchall()]


def cmd_curate(db: PostgresUtil, out_path: str, *, edge_conf_min: float) -> dict:
    """검토 초안 골든을 만든다 — 후보 쌍을 fs_path 키로, `_review:true`·제안 kind 와 함께 출력.

    ★ 이 산출물은 **골든이 아니다** — 사람이 편집(잘못된 쌍 제거·kind 확정·`_review` 제거·고립 검증)해야 골든이 된다.
    """
    with db.transaction() as conn:
        raw_pairs = _bootstrap_candidate_pairs(conn, edge_conf_min=edge_conf_min)
        # 부트스트랩 쌍(고conf graph_edge + path_signal)에 등장한 자산 = 관계/경로 후보 보유.
        ids = {p["a"] for p in raw_pairs} | {p["b"] for p in raw_pairs}
        # `specs/051-relation-golden-coverage/spec.md` C2: registered 중 그 집합에 없는 자산 = 관계 0 ∧ path 0 = 고립 후보(관계 단계·FR-101).
        #   035 isolation 의미(평가완료·엣지 0)와 일치. min_sim 이 낮아 임베딩 후보는 거의 모두 존재하므로
        #   "임베딩 후보 0" 대신 "관계/경로 후보 0"으로 고립을 정의한다(임베딩 전수 스캔 불요·결정적).
        reg_ids = _registered_asset_ids(conn)
        iso_ids = isolated_candidates(set(reg_ids), ids)
        id2path = _asset_fs_path(conn, ids | set(iso_ids))
    draft_pairs = []
    for p in raw_pairs:
        a, b = id2path.get(p["a"]), id2path.get(p["b"])
        if not a or not b or a == b:
            continue
        draft_pairs.append({"a": a, "b": b, "kind": p.get("_suggest_kind") or "REVIEW",
                            "note": f"{p['_source']}", "_review": True})
    draft = {"version": 1, "key_type": "fs_path", "pairs": draft_pairs,
             "isolated": sorted(id2path[i] for i in iso_ids if i in id2path),
             "_NOTE": "검토 초안 — 사람이 잘못된 쌍 제거·kind 확정·_review/_NOTE 제거·고립 검증 후에야 골든."}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(draft, f, ensure_ascii=False, indent=2)
    return {"draft_pairs": len(draft_pairs), "isolated": len(draft["isolated"]), "out": out_path}


# ── snapshot / measure ───────────────────────────────────────────────────────
def _make_config(args: Any, cfg: Any) -> dict:
    """측정 조건을 dict 로 굳힌다 — 스냅샷에 함께 저장돼 "어떤 조건의 수치인지" 남는다.

    ``prompt_variant`` 를 함께 굳히는 이유: shadow A/B 는 같은 표본을 두 번 떠서 비교하는데,
    파일명(``snap_A``/``snap_B``)으로만 구분하면 **보고서에서 두 팔을 뒤바꿔 적는 사고**를
    막을 방법이 없다. 스냅샷 자체가 자기가 어느 팔인지 알고 있어야 한다.
    측정 로직은 이 키를 읽지 않으므로 수치에는 영향이 없다.

    Args:
        args: 파싱된 CLI 인자. ``top_k``·``embedding_kind``·``prompt_variant`` 를 읽는다.
        cfg: 활성 설정. ``args`` 가 비운 값의 기본값 출처다.

    Returns:
        스냅샷 ``config`` 에 그대로 실릴 dict.
    """
    return {"top_k": args.top_k or cfg.relations.top_k, "min_sim": cfg.relations.min_sim,
            "embedding_kind": args.embedding_kind,
            # 골든 경로(변형 개념이 없는 호출)에서도 "baseline" 이 박힌다 — 사후에
            # "이 스냅샷은 운영 프롬프트였다"를 단언할 수 있으니 그편이 낫다.
            "prompt_variant": getattr(args, "prompt_variant", None) or "baseline"}


def assert_same_candidates(a: Snapshot, b: Snapshot) -> None:
    """두 스냅샷의 후보 집합이 같은지 단언한다(A/B 오염 검출).

    프롬프트만 바꾼 A/B 에서 후보가 달라졌다면 후보 단계가 함께 흔들렸다는 뜻이고, 그러면
    "프롬프트 때문에 좋아졌다"고 말할 수 없다. **실험을 계속하기 전에 멈춘다.**

    Args:
        a: 대조군 스냅샷.
        b: 실험군 스냅샷.

    Raises:
        AssertionError: 소스 집합이나 어느 소스의 후보 목록이 다를 때.
    """
    if set(a.sources) != set(b.sources):
        raise AssertionError(
            f"소스 집합이 다르다: A만 {sorted(set(a.sources) - set(b.sources))[:5]} / "
            f"B만 {sorted(set(b.sources) - set(a.sources))[:5]}")
    for sid in sorted(a.sources):
        ca, cb = a.sources[sid].candidates, b.sources[sid].candidates
        if ca != cb:
            raise AssertionError(f"후보가 다르다(source={sid}): A={ca[:3]} B={cb[:3]}")


def cmd_snapshot(
    db: PostgresUtil, golden: Golden | None = None, *, config: dict, out_path: str,
    source_ids: list[str] | None = None, prompt_variant: dict | None = None,
) -> dict:
    """골든(또는 표본) 소스마다 후보·LLM 제안을 받아 **파일로 동결**한다.

    이후 임계를 바꿔 가며 재측정할 때 LLM 을 다시 부르지 않기 위한 단계다.

    Args:
        db: DB 핸들.
        golden: 정답 묶음. 주면 여기서 소스를 도출한다. ``None`` 이면 ``source_ids`` 경로.
        config: 후보 조회 설정(임계·상한 등) — 스냅샷에 함께 저장된다.
        out_path: 동결 JSON 을 쓸 경로.
        source_ids: 골든 없이 쓸 소스 자산 id 목록(shadow A/B 표본). ``None`` 이면 ``golden`` 경로.
            ``golden`` 과 **정확히 하나만** 준다(둘 다/둘 다 아님이면 ``build_snapshot`` 이 거부).
        prompt_variant: 프롬프트 변형 정의(``PROMPT_VARIANTS`` 의 한 항목).
            ``None``·빈 dict(=baseline)이면 운영과 바이트 동일한 프롬프트다.

    Returns:
        동결 결과 요약 dict(파일 경로·소스 수·미해소 키 등).
    """
    snap, mapping, missing = build_snapshot(
        db, golden, config=config, source_ids=source_ids, prompt_variant=prompt_variant)
    payload = {"snapshot": dump_snapshot(snap), "key_to_id": mapping, "missing_keys": missing}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"sources": len(snap.sources), "missing_keys": len(missing), "out": out_path}


# 033 FR-004·005: measure 가 출력하는 임계 스윕 격자.
#   N1(min_sim): 후보 코사인 유사도 하한 후보 — recall/통과 후보 수.
#   #3(auto_approve 2D): LLM conf × 후보 emb_score 격자 — 자동승인 precision/승인 수.
_MIN_SIM_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
_AA_CONF_GRID = [0.8, 0.85, 0.9, 0.95]
_AA_EMB_GRID = [0.0, 0.3, 0.4, 0.5, 0.6]


def _resolve_golden_pairs(golden: Golden, key_to_id: dict[str, str]) -> list[tuple[str, str]]:
    """골든 쌍을 자산 id 공간으로 옮긴다.

    Args:
        golden: 정답 묶음(사람이 읽는 키로 적혀 있다).
        key_to_id: 키 → 자산 id 매핑.

    Returns:
        자산 id 쌍 목록. **양쪽이 모두 해소된 쌍만** 담는다 — 한쪽만 아는 쌍을 넣으면
        재현율 분모가 부풀어 지표가 실제보다 나빠 보인다.
    """
    pairs: list[tuple[str, str]] = []
    for p in golden.pairs:
        a, b = key_to_id.get(p.a), key_to_id.get(p.b)
        if a is not None and b is not None:
            pairs.append((a, b))
    return pairs


def cmd_measure(golden: Golden, snapshot_path: str, *, confidence_min: float = 0.0) -> dict:
    """골든+스냅샷 → 리포트. LLM 0·DB 0(스냅샷에 key_to_id 포함 — 결정적·SC-002).

    ``confidence_min``: 제안 엣지 accepted 판정 임계. **프로덕션 자동승인(RELATION_AUTO_APPROVE_MIN=0.9)
    으로 측정해야 precision/recall/isolation 이 실제 동작을 반영**한다(`specs/051-relation-golden-coverage/spec.md` — 0.0 이면 저신뢰 제안까지
    accepted 로 세어 isolation_accuracy 가 항상 0). 비회귀 게이트는 baseline 의 confidence_min 을 재사용한다.

    033 FR-004·005: 동결 스냅샷 위에서 min_sim 스윕(N1)·2D 자동승인 스윕(#3) 표를 더해 출력한다.
    **읽기 전용** — graph_edge/relation_kind 미기록(measure 의 측정 전용 성질 보존·SC-004).
    - N1 스윕 후보 신호 = 동결된 SourceSnapshot.candidates 의 (id, emb_score)(전체 후보 — FR-004).
    - #3 스윕 신호 = 제안 엣지의 emb_score(자동승인 대상은 제안 엣지이므로).
    ※ 스윕 하한 탐색범위는 스냅샷 생성 시 후보 조회 min_sim(현 0.2) 이상 — 그 아래를 보려면 더 낮은
      floor 로 스냅샷 재생성. N1 감사 목표는 "0.2 과느슨 → 상향"이라 상향 스윕으로 충분.
    """
    with open(snapshot_path, encoding="utf-8") as f:
        payload = json.load(f)
    snap = load_snapshot(payload["snapshot"])
    key_to_id = {str(k): str(v) for k, v in payload.get("key_to_id", {}).items()}
    report = build_report(golden, snap, key_to_id, confidence_min=confidence_min)

    # 033 스윕 — 동결 스냅샷(asset_id 공간) 위 결정적 재측정. 골든은 key_to_id 로 정합.
    from src.relations.quality.metrics import auto_approve_sweep, min_sim_sweep

    gpairs = _resolve_golden_pairs(golden, key_to_id)
    proposed = {sid: list(ss.proposed) for sid, ss in snap.sources.items()}
    # N1: 동결된 후보 (id, emb_score) 전체를 신호로 — 후보 단계 recall 을 점수 임계로 재측정(FR-004).
    cand_by_src = {sid: list(ss.candidates) for sid, ss in snap.sources.items()}
    report["min_sim_sweep"] = min_sim_sweep(gpairs, cand_by_src, thresholds=_MIN_SIM_GRID)
    report["auto_approve_sweep"] = auto_approve_sweep(
        gpairs, proposed, conf_thresholds=_AA_CONF_GRID, emb_thresholds=_AA_EMB_GRID)
    return report


def _dump_report(report: dict, path: str) -> None:
    """측정 리포트를 비교 기준 파일로 동결한다.

    Args:
        report: 측정 결과.
        path: 저장 경로.

    **키를 정렬해 쓴다** — 그래야 내용이 같을 때 파일 diff 가 비고, 무엇이 실제로 달라졌는지
    바로 보인다.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)


def _load_golden(path: str) -> Golden:
    """골든 파일을 읽어 검증까지 마친 객체로 돌려준다(형식 오류면 예외)."""
    with open(path, encoding="utf-8") as f:
        return parse_golden(json.load(f))


def main() -> int:
    """관계 품질 측정 CLI 진입점 — 하위 명령(스냅샷 동결·측정·큐레이션)을 분기한다.

    Returns:
        0=성공, 그 외=실패(셸 종료 코드).
    """
    import argparse
    from pathlib import Path

    from dotenv import load_dotenv

    from src.config.settings import init_settings

    p = argparse.ArgumentParser(description="관계 품질 측정 러너 (spec 031)")
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("curate", help="부트스트랩 후보 → 검토 초안 골든")
    pc.add_argument("--out", required=True)
    pc.add_argument("--edge-conf-min", type=float, default=0.8)

    ps = sub.add_parser("snapshot", help="골든(또는 active 표본) 소스 LLM 제안 동결")
    ps.add_argument("--golden", default=None, help="골든 파일 경로(--sample-active 와 택일)")
    ps.add_argument("--out", required=True)
    ps.add_argument("--top-k", dest="top_k", type=int, default=None)
    ps.add_argument("--embedding-kind", dest="embedding_kind", choices=["st", "clip", "both"], default="both")
    ps.add_argument("--sample-active", dest="sample_active", type=int, default=None,
                    help="골든 대신 active 엣지 보유 자산 N건을 시드 고정 표본으로 쓴다(A/B용)")
    ps.add_argument("--seed", type=int, default=None, help="--sample-active 의 표본 시드")
    ps.add_argument("--prompt-variant", dest="prompt_variant",
                    choices=sorted(PROMPT_VARIANTS), default="baseline",
                    help="프롬프트 변형(baseline=운영과 동일). shadow A/B 전용")

    pm = sub.add_parser("measure", help="골든+스냅샷 → 리포트(LLM 0)")
    pm.add_argument("--golden", required=True)
    pm.add_argument("--snapshot", required=True)
    pm.add_argument("--out", default=None, help="리포트를 baseline_report.json 로 동결(선택)")
    pm.add_argument("--confidence-min", dest="confidence_min", type=float, default=None,
                    help="accepted 판정 임계(미지정 시 RELATION_AUTO_APPROVE_MIN — 프로덕션 자동승인)")

    args = p.parse_args()
    if args.cmd == "snapshot":
        # 소스가 골든인지 표본인지 모호하면 "무엇을 측정했는가"가 흐려진다 — DB 를 열기 전에 막는다.
        if (args.golden is None) == (args.sample_active is None):
            ps.error("--golden 과 --sample-active 중 정확히 하나를 지정한다")
        # 시드가 없으면 매 실행 표본이 달라져 A/B 두 팔의 소스가 어긋난다(재현성 SC-002).
        if args.sample_active is not None and args.seed is None:
            ps.error("--sample-active 를 쓸 때는 --seed 를 함께 준다(같은 시드 = 같은 표본)")

    dotenv_path = Path(__file__).resolve().parents[1] / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    if args.cmd == "measure":  # DB/LLM 불요(스냅샷 기반)
        # confidence_min 미지정이면 프로덕션 자동승인 임계(RELATION_AUTO_APPROVE_MIN)로 측정.
        cmin = args.confidence_min if args.confidence_min is not None else _settings().relations.auto_approve_min
        report = cmd_measure(_load_golden(args.golden), args.snapshot, confidence_min=cmin)
        if args.out:
            _dump_report(report, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    db = PostgresUtil()
    with db:
        if args.cmd == "curate":
            out = cmd_curate(db, args.out, edge_conf_min=args.edge_conf_min)
        else:  # snapshot — 소스는 골든 또는 active 표본 중 하나(위에서 상호배타 검증됨)
            cfg = _settings()
            sids = (sample_active_sources(db, n=args.sample_active, seed=args.seed)
                    if args.sample_active is not None else None)
            out = cmd_snapshot(
                db, _load_golden(args.golden) if args.golden else None,
                config=_make_config(args, cfg), out_path=args.out, source_ids=sids,
                prompt_variant=PROMPT_VARIANTS[args.prompt_variant])
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
