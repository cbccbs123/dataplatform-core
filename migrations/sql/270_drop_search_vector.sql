-- 270 — asset_metadata.search_vector 제거 (spec 037: OpenSearch 전용 정리 — PG FTS 폐기).
--
-- 배경: search_vector(PostgreSQL to_tsvector 기반 FTS)는 삭제된 PG 검색 경로(media_search)만 읽었다.
--   037 에서 검색을 OpenSearch 전용으로 전환하면서 BM25 풀텍스트는 OS 색인(ext_meta 기반)이 담당한다.
--   적재(asset_persist)도 더 이상 to_tsvector 를 채우지 않으므로 컬럼과 GIN 인덱스가 dead 가 됐다.
-- 변경: GIN 인덱스 드롭 + search_vector 컬럼 드롭(IF EXISTS 로 멱등).
--   downgrade 는 컬럼·인덱스 구조만 복원하며, 적재 텍스트(데이터)는 재적재가 필요하다(비복원).

DROP INDEX IF EXISTS idx_asset_metadata_search_vector;
ALTER TABLE asset_metadata DROP COLUMN IF EXISTS search_vector;
