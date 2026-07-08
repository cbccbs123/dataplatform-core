"""062 G1 — API 텍스트 임베딩 seam 단위 테스트 [FR-101·105·SC-01/02/03/05].

네트워크 실호출 없이 ``requests.post`` 를 mock 한다. ``embed_texts_api`` 는 로컬 ``embed_texts`` 대칭 —
raw(정규화) 벡터를 반환하고 패딩은 하지 않는다(다운스트림 pad 단일 적용).
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock, patch

from src.embedders.text_embedder_api import embed_texts_api

_BASE = "http://192.168.109.254:32721/v1"


def _resp(data: list[dict]) -> MagicMock:
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"data": data}
    return m


class TestEmbedTextsApi(unittest.TestCase):
    def test_empty_input_no_network(self) -> None:
        with patch("requests.post") as post:
            self.assertEqual(embed_texts_api([], base_url=_BASE, model="bge-m3"), [])
            post.assert_not_called()

    @patch("requests.post")
    def test_body_endpoint_and_raw_no_padding(self, post) -> None:
        post.return_value = _resp([{"embedding": [3.0, 4.0], "index": 0}])
        out = embed_texts_api(["안녕"], base_url=_BASE, model="bge-m3", normalize_embeddings=False)
        # 엔드포인트·바디 계약
        args, kwargs = post.call_args
        self.assertEqual(args[0], _BASE + "/embeddings")
        self.assertEqual(kwargs["json"], {"model": "bge-m3", "input": ["안녕"]})
        # raw 반환(패딩 안 함) — 입력 임베딩 차원 그대로(1536 아님)
        self.assertEqual(out, [[3.0, 4.0]])

    @patch("requests.post")
    def test_l2_normalize(self, post) -> None:
        post.return_value = _resp([{"embedding": [3.0, 4.0], "index": 0}])
        out = embed_texts_api(["x"], base_url=_BASE, model="bge-m3", normalize_embeddings=True)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in out[0])), 1.0, places=6)
        self.assertAlmostEqual(out[0][0], 0.6, places=6)
        self.assertAlmostEqual(out[0][1], 0.8, places=6)

    @patch("requests.post")
    def test_restores_order_by_index(self, post) -> None:
        # 서버가 순서를 뒤집어 줘도 index 로 입력 순서 복원
        post.return_value = _resp([
            {"embedding": [2.0], "index": 1},
            {"embedding": [1.0], "index": 0},
        ])
        out = embed_texts_api(["a", "b"], base_url=_BASE, model="bge-m3", normalize_embeddings=False)
        self.assertEqual(out, [[1.0], [2.0]])

    @patch("requests.post")
    def test_no_auth_header_by_default(self, post) -> None:
        post.return_value = _resp([{"embedding": [1.0], "index": 0}])
        embed_texts_api(["x"], base_url=_BASE, model="bge-m3")
        self.assertNotIn("Authorization", post.call_args.kwargs["headers"])

    @patch("requests.post")
    def test_bearer_header_when_key(self, post) -> None:
        post.return_value = _resp([{"embedding": [1.0], "index": 0}])
        embed_texts_api(["x"], base_url=_BASE, model="bge-m3", api_key="secret")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer secret")

    @patch("requests.post")
    def test_batch_split(self, post) -> None:
        # N=5, batch_size=2 → ceil(5/2)=3 요청·순서·개수 보존
        post.side_effect = [
            _resp([{"embedding": [0.0], "index": 0}, {"embedding": [1.0], "index": 1}]),
            _resp([{"embedding": [2.0], "index": 0}, {"embedding": [3.0], "index": 1}]),
            _resp([{"embedding": [4.0], "index": 0}]),
        ]
        out = embed_texts_api(
            ["a", "b", "c", "d", "e"], base_url=_BASE, model="bge-m3",
            batch_size=2, normalize_embeddings=False,
        )
        self.assertEqual(post.call_count, 3)
        self.assertEqual(out, [[0.0], [1.0], [2.0], [3.0], [4.0]])

    @patch("requests.post")
    def test_count_mismatch_raises(self, post) -> None:
        # 입력 2개인데 응답 1개 → 명확한 오류(재시도로 삼키지 않음)
        post.return_value = _resp([{"embedding": [1.0], "index": 0}])
        with self.assertRaises(ValueError):
            embed_texts_api(["a", "b"], base_url=_BASE, model="bge-m3")

    @patch("requests.post")
    def test_empty_embedding_raises(self, post) -> None:
        # 빈 embedding → 조용한 0벡터 오염 방지(FR-105·SC-05). 정상 "빈 텍스트"와 구분해 즉시 오류.
        post.return_value = _resp([{"embedding": [], "index": 0}])
        with self.assertRaises(ValueError):
            embed_texts_api(["a"], base_url=_BASE, model="bge-m3")


if __name__ == "__main__":
    unittest.main()
