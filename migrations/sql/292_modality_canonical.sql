-- 053 (v292): asset.modality file_kind → canonical(+unknown 격리표식).
-- 순서 불변식: 기존 CHECK가 'text'를 불허하므로 DROP → UPDATE → ADD 순서 필수.

-- 1) 기존 CHECK 제거(100_asset_core.sql 의 인라인 무명 CHECK → PG 기본명 asset_modality_check)
ALTER TABLE asset DROP CONSTRAINT IF EXISTS asset_modality_check;

-- 2) 텍스트 계열 file_kind → canonical 'text' (image/video/audio/unknown 무변경)
UPDATE asset SET modality = 'text'
 WHERE modality IN ('txt', 'pdf', 'json', 'word', 'excel', 'powerpoint');

-- 3) canonical CHECK 재정의(text/image/video/audio + unknown 격리표식)
ALTER TABLE asset ADD CONSTRAINT asset_modality_check
    CHECK (modality IN ('text', 'image', 'video', 'audio', 'unknown'));

-- 4) 컬럼 의미 갱신(파일종류가 아니라 canonical 모달리티)
COMMENT ON COLUMN asset.modality IS
    '모달리티(canonical): text/image/video/audio + unknown(격리표식). 세분류(file_kind)는 fs_path 확장자로 재도출.';

-- 5) 통계 갱신(카디널리티 급감 → planner 정합)
ANALYZE asset;
