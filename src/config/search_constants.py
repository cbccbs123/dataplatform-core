"""OpenSearch 검색 융합·게이트·컷의 기본값 단일 출처(027 F1 — 순수 상수·의존 0).

027 독립 비평이 발견한 즉시 결함 F1: 임계·가중치 기본값이 **3곳에 복제·구식화**돼 있었다
(settings resolver·검색 모듈 중복 상수·calibrate 스크립트 — 코드 0.15/0.43 vs 운영 0.17/0.50,
새 환경에 설정 없이 배포하면 구식 코드 기본값이 적용되는 지뢰). 이 모듈을 **유일 출처**로 두고
``src.config.settings`` resolver·``src.search.opensearch_search``·calibrate 가 공유한다(FR-006).

순수 상수만 둔다(import 0·IO 0) — settings 미초기화 환경(순수 단위)에서도 안전하게 참조된다.
임계는 **코사인 스케일**(클라이언트 융합 전환으로 정규화 스케일 임계 4종이 폐기됨, 027). 실측
확정치(EPS·FLOOR·RESULT_FLOOR)는 G4 실OS 재보정 후 **여기 1곳만** 갱신한다.
"""

from __future__ import annotations

# ── 버킷 게이트(robust baseline) 기본값 ──────────────────────────────────────
# OS 검색은 기본 게이트 on(프로덕션 경로) — settings 미설정 환경에서도 무관 결과 표출을 차단한다.
OS_CUTOFF_ENABLED_DEFAULT: bool = True
# 상대 신호 임계: keep = (top − baseline) ≥ EPS. baseline = '하위 절반 평균'(robust — 밀집 토픽
# '충전' 오컷 구조 해결). **G4 실OS 재보정 확정치**(있음 Δmin 0.224 vs 없음 Δmax 0.189 분포 기준).
OS_CUTOFF_EPS_DEFAULT: float = 0.18
# 절대 backstop 임계: keep 의 두 번째 조건 top ≥ FLOOR(코퍼스 전체가 평탄할 때의 느슨한 하한).
OS_CUTOFF_FLOOR_DEFAULT: float = 0.50

# ── per-result 컷(단일 코사인 스케일) 기본값 ────────────────────────────────
# 행 유지 = BM25 매칭(어휘 증거) OR 원시 코사인 ≥ RESULT_FLOOR. 024 의 모달리티별 정규화 스케일 임계
# 4종을 대체하는 전역 1개(코사인 스케일). **G4 재보정 확정치**(없음 top 최대 0.538 위·차단 96% 운영점).
OS_RESULT_FLOOR_DEFAULT: float = 0.55

# ── 융합·서브검색 기본값 ────────────────────────────────────────────────────
# 융합 가중치 (w_bm25, w_knn) — 서브검색 순서 [bm25, knn] 과 동일 의미. min-max 정규화 후 가중평균.
OS_FUSION_WEIGHTS_DEFAULT: tuple[float, float] = (0.5, 0.5)
# BM25 multi_match operator 기본값 — **'and'(전 토큰 매칭·운영 검증값)**. 025 는 회귀0 원칙으로
# 'or' 기본+.env 옵트인이었으나, 027 F1 원칙(코드 기본=운영 보정값 — 새 환경 지뢰 제거)과 어휘
# 구제 규칙의 전제(BM25 매칭=전 토큰 증거)에 따라 기본을 운영점으로 통일했다(리뷰 후속).
OS_BM25_OPERATOR_DEFAULT: str = "and"
# 게이트 표본 하한 — 모달리티당 kNN 을 max(요청 k, 이 값)으로 뽑아 robust baseline(하위 절반 평균)이
# 충분한 표본에서 안정되게 한다(요청 k 가 작아도 background 추정이 흔들리지 않게).
OS_KNN_SAMPLE_K: int = 50

__all__ = [
    "OS_BM25_OPERATOR_DEFAULT",
    "OS_CUTOFF_ENABLED_DEFAULT",
    "OS_CUTOFF_EPS_DEFAULT",
    "OS_CUTOFF_FLOOR_DEFAULT",
    "OS_FUSION_WEIGHTS_DEFAULT",
    "OS_KNN_SAMPLE_K",
    "OS_RESULT_FLOOR_DEFAULT",
]
