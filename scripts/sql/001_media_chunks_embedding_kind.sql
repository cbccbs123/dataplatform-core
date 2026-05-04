-- 이미지: VLM 텍스트(SentenceTransformer) 청크와 CLIP 시각 청크를 구분한다.
-- 텍스트 문서 청크는 모두 'st'로 둔다.

ALTER TABLE media_chunks
    ADD COLUMN IF NOT EXISTS embedding_kind text NOT NULL DEFAULT 'st';

-- 기존 이미지 행은 chunk_index=0 하나뿐이며 CLIP 벡터였다.
UPDATE media_chunks mc
SET embedding_kind = 'clip'
FROM media_items mi
WHERE mc.media_item_id = mi.id
  AND mi.media_type = 'image'
  AND mc.chunk_index = 0;
