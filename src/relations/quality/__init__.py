"""관계 품질 측정 도구 모음.

순수 측정 로직(LLM/DB 0·결정적·단위테스트)을 모은 패키지.
- `golden`: 골든 관계셋 모델 + `parse_golden`(순수 검증).
- `snapshot`: LLM 제안 동결 스냅샷 모델 + JSON round-trip.
- `metrics`: 후보 recall·관계 P/R·임계 스윕·커버리지 곡선·구간 추정(순수 메트릭).
- `report`: 골든+스냅샷+키매핑 → 리포트 조립(순수).
- `rubric`: 맹검 판정 기준(버전 매긴 상수).
- `verdicts`: 맹검 판정 라벨 모델 + JSON round-trip.

DB/LLM 글루(키 해소·러너)는 이 패키지 밖(러너/사람 몫)이다.
"""
from src.relations.quality.golden import Golden, GoldenPair, parse_golden
from src.relations.quality.metrics import (
    candidate_recall,
    cohen_kappa,
    coverage_curve,
    relation_metrics,
    threshold_sweep,
    wilson_interval,
)
from src.relations.quality.report import build_report
from src.relations.quality.rubric import RUBRIC_KO_V1, RUBRIC_VERSION
from src.relations.quality.snapshot import (
    ProposedEdge,
    Snapshot,
    SourceSnapshot,
    dump_snapshot,
    load_snapshot,
)
from src.relations.quality.verdicts import (
    VALID_VERDICTS,
    Verdict,
    VerdictSet,
    dump_verdicts,
    error_rate,
    load_verdicts,
    verdict_counts,
)

# 판정 산출물 저장 위치(레포 루트 기준).
# ⚠️ 이 경로도 **버전관리 제외**다(사용자 결정 2026-08-05 · 라벨이 1.6MB 로 커져 PR 을 덮었다).
#    그래도 ``tests/golden/`` 과 경로를 갈라 두는 이유는 남는다 — 판정(측정 기록)과 골든(사람이
#    검증한 정답)을 물리적으로 분리해 **"판정을 골든으로 자동 승격하지 않는다"는 규율을 강제**한다.
#    로컬 파일이 유일본이므로 지우면 소급 재채점이 불가능하다(verdicts.py 모듈 docstring 참조).
VERDICTS_DIR = "tests/fixtures/relations/verdicts"

__all__ = [
    "Golden", "GoldenPair", "parse_golden",
    "ProposedEdge", "Snapshot", "SourceSnapshot", "dump_snapshot", "load_snapshot",
    "candidate_recall", "relation_metrics", "threshold_sweep",
    "coverage_curve", "wilson_interval", "cohen_kappa",
    "build_report",
    "RUBRIC_VERSION", "RUBRIC_KO_V1",
    "VALID_VERDICTS", "Verdict", "VerdictSet", "dump_verdicts", "load_verdicts",
    "verdict_counts", "error_rate", "VERDICTS_DIR",
]
