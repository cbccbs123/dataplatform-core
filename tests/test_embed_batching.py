"""069 T320(D7·P2-32) — 임베딩 청크 배치화. 실모델·네트워크 0.

측정(scratchpad·120 실코퍼스 청크): 운영 API bge-m3 개별요청 vs 배치요청 bit-identical(차이 0),
로컬 bge max|Δ|4.3e-7·cos이탈 8.8e-12(검색 무영향) → 배치화 안전·재백필 불필요.

이 단위 테스트는 **구조**(청크별 N회 호출 → 전 청크 1회 배치 호출)와 매핑 보존을 봉인한다.
수치 동일성은 실모델/실API 가 필요해 측정으로 검증(단위 테스트 범위 밖).
"""
from __future__ import annotations

import unittest
from unittest import mock


class TestEmbeddingTextChunksBatched(unittest.TestCase):
    """파일 기반 embedding_text_chunks — 전 청크를 1회 배치 임베딩."""

    def _run(self, *, chunks: list[str], channel=None, settings=None):
        import src.embedders.text_embedder as te

        calls: list[list[str]] = []

        def _fake_embed_many(texts, **kw):
            calls.append(list(texts))
            # 결정적 가짜 벡터: 텍스트 길이 기반(매핑 검증용)
            return [[float(len(t)), 1.0] for t in texts]

        def _fake_iter(path, *, file_kind, encoding, chunk_size, overlap_size, max_input_chars):
            yield from chunks

        with mock.patch.object(te, "_embed_many", side_effect=_fake_embed_many), \
             mock.patch.object(te, "iter_document_chunks", _fake_iter), \
             mock.patch.object(te, "pad_embedding_to_storage_dim", side_effect=lambda v: v), \
             mock.patch.object(te, "get_embedding_model",
                               return_value=mock.MagicMock(max_seq_length=None)), \
             mock.patch.object(te.Path, "is_file", return_value=True):
            out = te.embedding_text_chunks(
                "/tmp/x.txt", file_kind="txt", chunk_size=512, channel=channel, settings=settings,
            )
        return out, calls

    def test_single_batch_call_for_all_chunks(self) -> None:
        # 핵심(D7): 청크가 여러 개여도 _embed_many 는 **1회**·전 청크를 한 리스트로.
        out, calls = self._run(chunks=["가나다", "라마", "바사아자"])
        self.assertEqual(len(calls), 1, f"배치 1회여야: {len(calls)}회")
        self.assertEqual(calls[0], ["가나다", "라마", "바사아자"])

    def test_chunk_index_and_vector_mapping(self) -> None:
        # chunk_index 순번(0,1,2)·벡터가 청크 순서대로 매핑(가짜벡터=길이).
        out, _ = self._run(chunks=["가나다", "라마", "바사아자"])
        self.assertEqual([r["chunk_index"] for r in out], [0, 1, 2])
        self.assertEqual([r["embedding_vector"][0] for r in out], [3.0, 2.0, 4.0])

    def test_empty_chunk_replaced_with_space(self) -> None:
        # 공백/빈 청크는 " " 로 치환돼 임베딩(기존 동작 보존) — 길이 1.
        out, calls = self._run(chunks=["   ", "정상"])
        self.assertEqual(calls[0][0], " ")
        self.assertEqual(out[0]["embedding_vector"][0], 1.0)

    def test_empty_document_single_zero_vector(self) -> None:
        # 청크 0개 → 인덱스0 zero-vector 1개, 배치 호출 없음(빈 임베딩 방지).
        out, calls = self._run(chunks=[])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["chunk_index"], 0)
        self.assertEqual(calls, [])

    def test_channel_none_uses_local_embed_texts(self) -> None:
        # channel 미지정 → 로컬 embed_texts 경유(embed_texts_for 아님).
        import src.embedders.text_embedder as te

        def _fake_iter(path, **kw):
            yield "청크"

        with mock.patch.object(te, "embed_texts", return_value=[[9.0]]) as m_local, \
             mock.patch.object(te, "embed_texts_for") as m_api, \
             mock.patch.object(te, "iter_document_chunks", _fake_iter), \
             mock.patch.object(te, "pad_embedding_to_storage_dim", side_effect=lambda v: v), \
             mock.patch.object(te, "get_embedding_model",
                               return_value=mock.MagicMock(max_seq_length=None)), \
             mock.patch.object(te.Path, "is_file", return_value=True):
            te.embedding_text_chunks("/tmp/x.txt", file_kind="txt", chunk_size=512, channel=None)
        m_local.assert_called_once()
        m_api.assert_not_called()

    def test_vector_count_mismatch_raises(self) -> None:
        # zip(strict=True) 방어선: _embed_many 가 청크수보다 적은 벡터를 돌려주면(응답 손실 등) 즉시
        # ValueError — silent 하게 앞쪽 청크만 매핑되고 뒤 청크가 사라지는 오류를 차단(069 리뷰 🟡).
        import src.embedders.text_embedder as te

        def _fake_iter(path, **kw):
            yield "청크1"
            yield "청크2"

        with mock.patch.object(te, "_embed_many", side_effect=lambda texts, **kw: [[0.1]]), \
             mock.patch.object(te, "iter_document_chunks", _fake_iter), \
             mock.patch.object(te, "pad_embedding_to_storage_dim", side_effect=lambda v: v), \
             mock.patch.object(te, "get_embedding_model",
                               return_value=mock.MagicMock(max_seq_length=None)), \
             mock.patch.object(te.Path, "is_file", return_value=True):
            with self.assertRaises(ValueError):
                te.embedding_text_chunks("/tmp/x.txt", file_kind="txt", chunk_size=512)

    def test_channel_set_uses_embed_texts_for(self) -> None:
        # channel 지정 → embed_texts_for(백엔드 라우팅) 경유.
        import src.embedders.text_embedder as te

        def _fake_iter(path, **kw):
            yield "청크"

        with mock.patch.object(te, "embed_texts_for", return_value=[[9.0]]) as m_api, \
             mock.patch.object(te, "embed_texts") as m_local, \
             mock.patch.object(te, "iter_document_chunks", _fake_iter), \
             mock.patch.object(te, "pad_embedding_to_storage_dim", side_effect=lambda v: v), \
             mock.patch.object(te, "get_embedding_model",
                               return_value=mock.MagicMock(max_seq_length=None)), \
             mock.patch.object(te.Path, "is_file", return_value=True):
            te.embedding_text_chunks(
                "/tmp/x.txt", file_kind="txt", chunk_size=512, channel="st_api")
        m_api.assert_called_once()
        m_local.assert_not_called()


class TestEmbeddingPlainTextChunksBatched(unittest.TestCase):
    """문자열 기반 embedding_plain_text_chunks(STT) — 전 청크 1회 배치."""

    def _run(self, *, chunks: list[str]):
        import src.embedders.text_embedder as te

        calls: list[list[str]] = []

        def _fake_embed_many(texts, **kw):
            calls.append(list(texts))
            return [[float(len(t))] for t in texts]

        def _fake_iter(text, *, chunk_size, overlap_size, max_input_chars):
            yield from chunks

        with mock.patch.object(te, "_embed_many", side_effect=_fake_embed_many), \
             mock.patch.object(te, "iter_plain_text_chunks", _fake_iter), \
             mock.patch.object(te, "pad_embedding_to_storage_dim", side_effect=lambda v: v):
            out = te.embedding_plain_text_chunks("전문", chunk_size=512)
        return out, calls

    def test_single_batch_call(self) -> None:
        out, calls = self._run(chunks=["가나", "다라마"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ["가나", "다라마"])
        self.assertEqual([r["chunk_index"] for r in out], [0, 1])

    def test_empty_single_zero_vector(self) -> None:
        out, calls = self._run(chunks=[])
        self.assertEqual(len(out), 1)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
