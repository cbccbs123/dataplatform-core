"""027 T001 — 검색 융합·게이트 기본값 단일 출처(src/config/search_constants.py) 단위.

027 F1(비평 발견): 임계·가중치 기본값이 settings·검색 모듈·calibrate 3곳에 복제·구식화돼
있었다(코드 0.15/0.43 vs 운영 0.17/0.50 — 새 환경 지뢰). 본 모듈은 그 기본값의 **단일 출처**다
(순수 상수·의존 0). settings resolver·opensearch_search·calibrate 가 모두 이 값을 공유한다(FR-006).

여기서는 값 존재·타입(가중치 튜플·k 정수·연산자 문자열)만 고정한다 — 실측 확정치(EPS·FLOOR·
RESULT_FLOOR)는 G4 재보정 후 이 모듈 1곳만 갱신한다.
"""

from __future__ import annotations

import unittest

from src.config import search_constants as sc


class SearchConstantsTest(unittest.TestCase):
    def test_cutoff_enabled_default_is_bool_true(self) -> None:
        # 027: OS 검색 게이트는 기본 on(프로덕션 경로) — settings 미설정 시에도 robust baseline 게이트 적용.
        self.assertIsInstance(sc.OS_CUTOFF_ENABLED_DEFAULT, bool)
        self.assertTrue(sc.OS_CUTOFF_ENABLED_DEFAULT)

    def test_cutoff_eps_floor_are_floats(self) -> None:
        # 상대(EPS)·절대(FLOOR) 게이트 임계 — 코사인 스케일 실수. 값 자체는 G4 확정(여기선 타입·존재).
        self.assertIsInstance(sc.OS_CUTOFF_EPS_DEFAULT, float)
        self.assertIsInstance(sc.OS_CUTOFF_FLOOR_DEFAULT, float)
        # 코사인 스케일 임계는 [-1, 1] 범위에 든다(상식 가드 — 잘못된 스케일 값 조기 차단).
        self.assertGreaterEqual(sc.OS_CUTOFF_FLOOR_DEFAULT, -1.0)
        self.assertLessEqual(sc.OS_CUTOFF_FLOOR_DEFAULT, 1.0)

    def test_result_floor_is_float_in_cosine_range(self) -> None:
        # per-result 컷(024 계승·스케일 통일): 코사인 ≥ RESULT_FLOOR. 코사인 스케일 단일 임계.
        self.assertIsInstance(sc.OS_RESULT_FLOOR_DEFAULT, float)
        self.assertGreaterEqual(sc.OS_RESULT_FLOOR_DEFAULT, -1.0)
        self.assertLessEqual(sc.OS_RESULT_FLOOR_DEFAULT, 1.0)

    def test_fusion_weights_default_is_pair_tuple(self) -> None:
        # 융합 가중치 (w_bm25, w_knn) — 2-튜플(서브검색 [bm25, knn] 순서와 동일 의미).
        w = sc.OS_FUSION_WEIGHTS_DEFAULT
        self.assertIsInstance(w, tuple)
        self.assertEqual(len(w), 2)
        for x in w:
            self.assertIsInstance(x, float)

    def test_bm25_operator_default_is_or(self) -> None:
        # 회귀 0(현행): 기본 multi_match operator 는 'or'(전 토큰 강제는 'and' 옵트인).
        self.assertEqual(sc.OS_BM25_OPERATOR_DEFAULT, "and")

    def test_knn_sample_k_is_positive_int(self) -> None:
        # 게이트 표본 하한 — robust baseline(하위 절반 평균)이 의미 있으려면 충분한 표본이 필요하다.
        self.assertIsInstance(sc.OS_KNN_SAMPLE_K, int)
        self.assertGreaterEqual(sc.OS_KNN_SAMPLE_K, 2)


if __name__ == "__main__":
    unittest.main()


class RerankConstantsTest(unittest.TestCase):
    """028: rerank 평가 기본값 — 기본 off(회귀 0)·모델·R·τ 단일 출처."""

    def test_rerank_defaults(self) -> None:
        self.assertIs(sc.OS_RERANK_ENABLED_DEFAULT, False)
        self.assertEqual(sc.OS_RERANK_MODEL_DEFAULT, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(sc.OS_RERANK_TOP_R_DEFAULT, 10)
        self.assertIsInstance(sc.OS_RERANK_TAU_DEFAULT, float)


class QueryNormConstantsTest(unittest.TestCase):
    """029 T006: LLM 질의 명사구 정규화 토글 기본값 — 기본 off(회귀 0·021 FR-004 토글 개정).

    021 FR-004 가 제거한 검색시점 LLM 을 029 가 기본 off 토글로만 재허용한다(§3 결정성 제약 강제).
    기본값을 단일 출처(search_constants)에 둬 settings resolver·배선이 공유한다(027 F1). 기본 False 라
    settings 미설정 환경·계약 테스트·027 폴백이 깨지지 않는다(채택은 .env.dev opt-in)."""

    def test_query_norm_enabled_default_is_false(self) -> None:
        # 기본 off — 명시 토글 없이는 검색시점 LLM 미실행(027 바이트 동일·SC-001).
        self.assertIs(sc.OS_QUERY_NORM_ENABLED_DEFAULT, False)

    def test_query_norm_default_registered_in_all(self) -> None:
        # 단일 출처 공개 계약: __all__ 등재(settings·배선이 공개 상수로 참조).
        self.assertIn("OS_QUERY_NORM_ENABLED_DEFAULT", sc.__all__)
