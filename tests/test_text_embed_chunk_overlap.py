"""069 T311(D8·P2-33) — 임베딩 청크 overlap 설정 신설·배선 + 과대 chunk_size 경고. 실모델 0.

문제(원 리뷰 P2-33): ``embedding_text_chunks`` 의 청크 overlap 이 하드코딩 0 이라 설정으로 조정할 수
없고, ``chunk_size`` 가 모델 max_seq_length 를 크게 넘으면 인코딩 시 조용히 잘려(재현 불가) 임베딩
품질이 떨어질 수 있는데 가드가 없었다.

처방: ``text_embedding_chunk_overlap`` 설정(env ``TEXT_EMBED_CHUNK_OVERLAP``·기본 0) 신설·배선
(기본 0 = 하드코딩과 동일·동작 불변) + ``chunk_size > max_seq_length*2`` 경고(로그만·잘림 관측).
"""

from __future__ import annotations

import logging
import unittest
from unittest import mock


class TestOverlapThreading(unittest.TestCase):
    """settings 의 overlap 이 iter_document_chunks 까지 전달되는지(기본 0=불변)."""

    def _run_with(self, *, settings, overlap_capture: list) -> None:
        import src.embedders.text_embedder as te

        def _fake_iter_document_chunks(path, *, file_kind, encoding, chunk_size,
                                       overlap_size, max_input_chars):
            overlap_capture.append(overlap_size)
            yield "청크1"

        with mock.patch.object(te, "iter_document_chunks", _fake_iter_document_chunks), \
             mock.patch.object(te, "_embed_one", return_value=[0.1, 0.2]), \
             mock.patch.object(te.Path, "is_file", return_value=True):
            te.embedding_text_chunks(
                "/tmp/x.txt", file_kind="txt", chunk_size=512, settings=settings,
            )

    def test_default_overlap_zero_when_no_settings(self) -> None:
        # settings 미전달 → overlap 0(하드코딩과 동일·동작 불변).
        cap: list = []
        self._run_with(settings=None, overlap_capture=cap)
        self.assertEqual(cap, [0])

    def test_overlap_from_settings_field(self) -> None:
        # settings.text_embedding_chunk_overlap=64 → iter_document_chunks 로 64 전달.
        cap: list = []
        fake_settings = mock.MagicMock(text_embedding_chunk_overlap=64)
        self._run_with(settings=fake_settings, overlap_capture=cap)
        self.assertEqual(cap, [64])

    def test_missing_field_falls_back_to_zero(self) -> None:
        # settings 에 필드가 없어도(구 객체) getattr 폴백 0(안전).
        cap: list = []

        class _NoField:
            pass

        self._run_with(settings=_NoField(), overlap_capture=cap)
        self.assertEqual(cap, [0])


class TestChunkSizeWarning(unittest.TestCase):
    """chunk_size 가 모델 max_seq_length*2 초과 시 경고(로그만·동작 불변)."""

    def _run(self, *, chunk_size: int, max_seq: int):
        import src.embedders.text_embedder as te

        fake_model = mock.MagicMock()
        fake_model.max_seq_length = max_seq
        with mock.patch.object(te, "get_embedding_model", return_value=fake_model), \
             mock.patch.object(te, "iter_document_chunks",
                               side_effect=lambda *a, **k: iter(["청크"])), \
             mock.patch.object(te, "_embed_one", return_value=[0.1]), \
             mock.patch.object(te.Path, "is_file", return_value=True):
            with self.assertLogs("src.embedders.text_embedder", level="WARNING") as cm:
                te.embedding_text_chunks(
                    "/tmp/x.txt", file_kind="txt", chunk_size=chunk_size, channel=None,
                )
                logging.getLogger("src.embedders.text_embedder").warning("_sentinel_")
        return cm.output

    def test_warns_when_chunk_size_exceeds_double_max_seq(self) -> None:
        # chunk_size=1200 > max_seq(512)*2=1024 → 경고.
        out = self._run(chunk_size=1200, max_seq=512)
        self.assertTrue(any("max_seq_length" in m or "잘림" in m for m in out),
                        f"과대 chunk_size 경고 기대: {out}")

    def test_no_warning_within_limit(self) -> None:
        # chunk_size=512 <= 512*2 → 경고 없음(sentinel 만).
        out = self._run(chunk_size=512, max_seq=512)
        real = [m for m in out if "_sentinel_" not in m]
        self.assertEqual(real, [], f"한도 내 경고 없어야: {real}")


class TestSettingDefault(unittest.TestCase):
    def test_env_unset_defaults_zero(self) -> None:
        # 069 D8: TEXT_EMBED_CHUNK_OVERLAP 미설정 → 0(동작 불변). resolver 직접 확인(전체 settings 빌드 회피).
        import os

        from src.config.settings import _env_int_default

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEXT_EMBED_CHUNK_OVERLAP", None)
            self.assertEqual(_env_int_default("TEXT_EMBED_CHUNK_OVERLAP", 0), 0)


if __name__ == "__main__":
    unittest.main()
