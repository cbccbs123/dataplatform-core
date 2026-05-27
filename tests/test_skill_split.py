"""v2 단계 A — skill 추출/임베딩 분해의 mock 기반 단위 테스트.

함수-로컬 import(`from src.x import y`)는 호출 시점에 소스 모듈 속성을 읽으므로
소스 모듈 속성을 patch 하면 가짜로 대체된다. 실제 모델/파일 없이 분해 구조를 검증한다.
"""
from __future__ import annotations

import unittest
from unittest import mock

from src.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext


def _ctx(modality="txt"):
    return ExtractContext(file_path="/d/a." + modality, modality=modality, settings=object())


class TestTextSplit(unittest.TestCase):
    def test_extract_meta_has_no_embeddings(self) -> None:
        from src.skills import text_skill
        with mock.patch.multiple("src.extractors.text_meta_extractor", extract_text_meta=mock.Mock(return_value={"size": 1})), \
             mock.patch.multiple("src.llm.text_summarizer", summarize_and_extract_keywords=mock.Mock(return_value={"summary": "s", "keywords": ["k"]})), \
             mock.patch.multiple("src.preprocess.media_item_search_text", build_media_item_fts_plain=mock.Mock(return_value="fts")):
            ctx = _ctx()
            ctx.settings = mock.Mock(encoding="utf-8", text_embedding_chunk_size=100,
                                     text_embedding_model="m", text_embedding_normalize=True)
            rec = text_skill._extract_text_meta(ctx)
        self.assertIsInstance(rec, AssetRecord)
        self.assertEqual(rec.embeddings, [])
        self.assertEqual(rec.ext_meta.get("summary"), "s")
        self.assertEqual(rec.fts_plain, "fts")

    def test_embed_returns_items_and_wrapper_composes(self) -> None:
        from src.skills import text_skill
        chunks = [{"embedding_vector": [0.1], "chunk_index": 0}, {"embedding_vector": [0.2], "chunk_index": 1}]
        ctx = _ctx()
        ctx.settings = mock.Mock(encoding="utf-8", text_embedding_chunk_size=100,
                                 text_embedding_model="m", text_embedding_normalize=True)
        with mock.patch.multiple("src.embedders.text_embedder", embedding_text_chunks=mock.Mock(return_value=chunks)):
            embs = text_skill._embed_text(ctx, AssetRecord())
        self.assertEqual([e.chunk_index for e in embs], [0, 1])
        self.assertTrue(all(isinstance(e, EmbeddingItem) and e.channel == "st" for e in embs))


class TestImageSplit(unittest.TestCase):
    def _cfg(self):
        return mock.Mock(labels_score_min=0.0, image_labels_meta_top_k=5,
                         text_embedding_model="m", text_embedding_normalize=True)

    def test_extract_stashes_clip_vec_and_embed_reuses(self) -> None:
        from src.skills import image_skill
        ctx = _ctx("image"); ctx.settings = self._cfg()
        zs = {"label_scores": {"고양이": 0.9}, "clip_image_embedding": [0.5] * 4}
        with mock.patch.multiple("src.extractors.image_meta_extractor", extract_image_meta=mock.Mock(return_value={"w": 10})), \
             mock.patch.multiple("src.llm.image_summarizer", summarize_image_caption_keywords_objects=mock.Mock(return_value={"summary": "s", "objects": ["고양이"]})), \
             mock.patch.multiple("src.embedders.image_embedder",
                                 zero_shot_tag_image_korean_clip=mock.Mock(return_value=zs),
                                 clip_zero_shot_ko_meta_items=mock.Mock(return_value=[{"label": "고양이", "score": 0.9}])), \
             mock.patch.multiple("src.preprocess.media_item_search_text", build_media_item_fts_plain=mock.Mock(return_value="fts")):
            rec = image_skill._extract_image_meta(ctx)
        self.assertEqual(rec.embeddings, [])
        self.assertEqual(ctx.scratch["clip_vec"], [0.5] * 4)

        with mock.patch.multiple("src.embedders.text_embedder",
                                 embed_texts=mock.Mock(return_value=[[0.1, 0.2]]),
                                 pad_embedding_to_storage_dim=mock.Mock(side_effect=lambda v: v)), \
             mock.patch.multiple("src.preprocess.vlm_text_for_embedding",
                                 build_image_vlm_text_for_embedding=mock.Mock(return_value="텍스트")):
            embs = image_skill._embed_image(ctx, rec)
        chans = sorted(e.channel for e in embs)
        self.assertEqual(chans, ["clip", "st"])
        clip = next(e for e in embs if e.channel == "clip")
        self.assertEqual(clip.vector, [0.5] * 4)


class TestAudioSplit(unittest.TestCase):
    def test_extract_stashes_stt_text_and_embed_reuses(self) -> None:
        from src.skills import audio_skill
        ctx = _ctx("audio")
        ctx.settings = mock.Mock(text_embedding_chunk_size=100, text_embedding_model="m", text_embedding_normalize=True)
        with mock.patch.multiple("src.preprocess.stt", transcribe_audio_local=mock.Mock(return_value={"text": "안녕"})), \
             mock.patch.multiple("src.extractors.audio_meta_extractor", extract_audio_meta=mock.Mock(return_value={"dur": 3})), \
             mock.patch.multiple("src.llm.text_summarizer", summarize_and_extract_keywords_from_audio=mock.Mock(return_value={"summary": "s"})), \
             mock.patch.multiple("src.preprocess.media_item_search_text", build_media_item_fts_plain=mock.Mock(return_value="fts")):
            rec = audio_skill._extract_audio_meta(ctx)
        self.assertEqual(rec.embeddings, [])
        self.assertEqual(ctx.scratch["stt_text"], "안녕")

        with mock.patch.multiple("src.embedders.text_embedder",
                                 embedding_plain_text_chunks=mock.Mock(return_value=[{"embedding_vector": [0.1], "chunk_index": 0}])):
            embs = audio_skill._embed_audio(ctx, rec)
        self.assertEqual([e.channel for e in embs], ["st"])
        self.assertEqual(embs[0].chunk_index, 0)


class TestVideoSplit(unittest.TestCase):
    def test_extract_stashes_keyframes_and_embed_pairs(self) -> None:
        from src.skills import video_skill
        ctx = _ctx("video")
        ctx.settings = mock.Mock(video_max_keyframes=2, labels_score_min=0.0,
                                 video_keyframe_labels_meta_top_k=5, text_embedding_model="m",
                                 text_embedding_normalize=True)
        frames = [{"scene_index": 0, "start_sec": 0, "end_sec": 1, "frame_sec": 0, "jpeg_bytes": b"x"}]
        clip_ve = {"keyframes": [{"clip_image_embedding": [0.7] * 4,
                                  "summary": {"summary": "s", "keywords": ["k"]},
                                  "labels": [{"label": "l", "score": 0.9}]}]}
        with mock.patch.multiple("src.preprocess.video_keyframes",
                                 extract_video_representative_frame_bytes=mock.Mock(return_value=frames),
                                 extract_video_basic_meta=mock.Mock(return_value={"dur": 5})), \
             mock.patch.multiple("src.llm.image_summarizer",
                                 summarize_image_caption_keywords_objects_from_jpeg_bytes=mock.Mock(return_value={"summary": "s", "objects": ["k"]})), \
             mock.patch.multiple("src.llm.video_summarizer", summarize_video_from_scene_results=mock.Mock(return_value={"summary": "v"})), \
             mock.patch.multiple("src.embedders.video_embedder", embed_video_keyframes_clip=mock.Mock(return_value=clip_ve)), \
             mock.patch.multiple("src.preprocess.media_item_search_text", build_media_item_fts_plain=mock.Mock(return_value="fts")):
            rec = video_skill._extract_video_meta(ctx)
        self.assertEqual(rec.embeddings, [])
        self.assertEqual(len(ctx.scratch["keyframes"]), 1)
        self.assertEqual(ctx.scratch["keyframes"][0]["clip_vec"], [0.7] * 4)

        with mock.patch.multiple("src.embedders.text_embedder",
                                 embed_texts=mock.Mock(return_value=[[0.1]]),
                                 pad_embedding_to_storage_dim=mock.Mock(side_effect=lambda v: v)), \
             mock.patch.multiple("src.preprocess.vlm_text_for_embedding",
                                 build_image_vlm_text_for_embedding=mock.Mock(return_value="텍스트")):
            embs = video_skill._embed_video(ctx, rec)
        self.assertEqual(sorted(e.channel for e in embs), ["clip", "st"])
        self.assertTrue(all(e.chunk_index == 0 for e in embs))


if __name__ == "__main__":
    unittest.main()
