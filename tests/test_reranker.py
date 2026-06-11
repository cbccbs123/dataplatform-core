"""028 — reranker 모듈 단위(모델 무로드 — lazy 경계·빈 입력만 순수 검증).

실제 CrossEncoder 채점은 평가 스크립트(scripts/measure_rerank_ab.py·실모델)가 담당하고,
여기서는 모듈 import 가 무거운 의존을 당기지 않는 것과 빈 입력 단락만 봉인한다.
검색 경로 통합(τ 필터·재정렬)은 test_opensearch_search 의 가짜 rerank_fn 단위가 검증.
"""

from __future__ import annotations

import sys
import unittest


class RerankerModuleTest(unittest.TestCase):
    def test_import_is_light(self) -> None:
        # 모듈 import 만으로 sentence_transformers/torch 를 당기지 않는다(lazy — 검색 경로 순수성).
        for m in list(sys.modules):
            if "reranker" in m:
                del sys.modules[m]
        before = "sentence_transformers" in sys.modules
        import src.search.reranker  # noqa: F401

        self.assertEqual("sentence_transformers" in sys.modules, before)

    def test_empty_texts_short_circuit(self) -> None:
        # 빈 입력은 모델 로드 없이 [] — score_pairs 가 lazy 경계를 지킨다.
        from src.search.reranker import score_pairs

        self.assertEqual(score_pairs("질의", [], model_name="아무모델"), [])


if __name__ == "__main__":
    unittest.main()
