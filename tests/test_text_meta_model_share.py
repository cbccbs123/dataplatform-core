"""069 T003(P1-7) — text_meta 가 text_embedder 의 모델 캐시를 공유(이중 로드 해소). 실모델 0."""

from __future__ import annotations

import unittest


class TestEmbeddingModelCacheShared(unittest.TestCase):
    def test_same_function_object(self) -> None:
        # 핵심: 두 모듈의 get_embedding_model 이 **같은 함수 객체**(같은 lru_cache) — 별도 캐시로
        # 같은 체크포인트를 두 번 로드하던 P1-7 이 구조적으로 불가능해졌음을 증명.
        from src.embedders.text_embedder import get_embedding_model as embedder_fn
        from src.extractors.text_meta_extractor import get_embedding_model as meta_fn

        self.assertIs(meta_fn, embedder_fn)

    def test_count_tokens_uses_shared_cache(self) -> None:
        # count_tokens 경로가 공유 캐시를 실제로 경유한다(모델 mock — 네트워크·실로드 0).
        from unittest.mock import MagicMock, patch

        import src.extractors.text_meta_extractor as tme

        fake = MagicMock()
        fake.tokenizer.encode.return_value = [1, 2, 3]
        with patch.object(tme, "get_embedding_model", return_value=fake) as mk:
            n = tme.count_tokens("안녕하세요", model_name="m")
        self.assertEqual(n, 3)
        mk.assert_called_once_with("m")


if __name__ == "__main__":
    unittest.main()
