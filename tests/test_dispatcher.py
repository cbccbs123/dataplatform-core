"""F-3.4 디스패처(단순 분기) 단위 테스트.

실제 추출 로직(T1-6/T2-1)은 stub 이므로, 여기서는 modality→추출 함수 분기와
미지원 modality 처리, 데이터클래스 계약만 검증한다(DB·LLM 불필요).
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.dispatch import dispatcher
from src.dispatch.dispatcher import UnsupportedModalityError, dispatch_extract
from src.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind


def _ctx(modality: str, domain: str = "general") -> ExtractContext:
    return ExtractContext(file_path="/tmp/x", modality=modality, domain=domain)


def _rec(tag: str) -> AssetRecord:
    return AssetRecord(tags=[tag])


class TestTypes(unittest.TestCase):
    def test_assetrecord_defaults(self) -> None:
        r = AssetRecord()
        self.assertEqual(r.core_meta, {})
        self.assertEqual(r.ext_meta, {})
        self.assertEqual(r.tags, [])
        self.assertEqual(r.fts_plain, "")
        self.assertEqual(r.embeddings, [])

    def test_assetrecord_fields(self) -> None:
        e = EmbeddingItem(channel="st", vector=[0.1, 0.2], model_name="m")
        r = AssetRecord(core_meta={"a": 1}, ext_meta={"summary": "s"}, tags=["t"], fts_plain="x", embeddings=[e])
        self.assertEqual(r.core_meta["a"], 1)
        self.assertEqual(r.ext_meta["summary"], "s")
        self.assertEqual(r.embeddings[0].channel, "st")

    def test_embeddingitem_defaults(self) -> None:
        e = EmbeddingItem(channel="clip", vector=[0.0], model_name="legacy")
        self.assertEqual(e.chunk_index, 0)
        self.assertIsNone(e.model_version)


class TestRouting(unittest.TestCase):
    def test_text_kinds_route_to_extract_text(self) -> None:
        # 문서류(txt/pdf/json/word/excel/powerpoint) 전부 extract_text 로.
        for kind in ALLOWED_TEXT_META_FILE_KINDS:
            with mock.patch.object(dispatcher, "extract_text", return_value=_rec("text")) as m:
                out = dispatch_extract(_ctx(kind))
            self.assertEqual(out.tags, ["text"], f"{kind} 가 text 로 분기되지 않음")
            m.assert_called_once()

    def test_image_route(self) -> None:
        with mock.patch.object(dispatcher, "extract_image", return_value=_rec("image")) as m:
            out = dispatch_extract(_ctx(MediaKind.IMAGE.value))
        self.assertEqual(out.tags, ["image"])
        m.assert_called_once()

    def test_video_route(self) -> None:
        with mock.patch.object(dispatcher, "extract_video", return_value=_rec("video")) as m:
            out = dispatch_extract(_ctx(MediaKind.VIDEO.value))
        self.assertEqual(out.tags, ["video"])
        m.assert_called_once()

    def test_audio_route(self) -> None:
        with mock.patch.object(dispatcher, "extract_audio", return_value=_rec("audio")) as m:
            out = dispatch_extract(_ctx(MediaKind.AUDIO.value))
        self.assertEqual(out.tags, ["audio"])
        m.assert_called_once()

    def test_unsupported_modality_raises(self) -> None:
        for bad in (MediaKind.UNKNOWN.value, "bogus", ""):
            with self.assertRaises(UnsupportedModalityError):
                dispatch_extract(_ctx(bad))

    def test_domain_does_not_change_modality_routing(self) -> None:
        # domain 이 medical 이어도 6월 단계에서는 modality(txt)로 분기.
        with mock.patch.object(dispatcher, "extract_text", return_value=_rec("text")) as m:
            out = dispatch_extract(_ctx(MediaKind.TEXT.value, domain="medical"))
        self.assertEqual(out.tags, ["text"])
        m.assert_called_once()


if __name__ == "__main__":
    unittest.main()
