-- =============================================================================
-- labels ext_meta json_schema 교정 — array of string → array of object (spec 039 시드 버그 수정).
--
-- 배경: v280 이 labels/objects/keyframes 의 items 를 보완하며 labels 를 {"type":"string"} 로 잘못 지정했다.
--   실제 추출기(image/video CLIP 제로샷)는 labels 를 **[{label, score}] 객체 배열**로 만든다(옛 백업 데이터
--   asset_metadata 도 동일). 039 ext_meta 값 검증(v291 ext_meta_field_registry 런타임 정본)이 활성화된
--   신규 적재에서 이 불일치로 image/video 자산이 ExtMetaValidationError 로 전부 실패한다.
-- 처방: labels 스키마를 실제 형식(object{label:string, score:number}·label 필수)으로 교정. objects(문자열
--   배열)·keyframes(객체 배열)·keywords(문자열)는 데이터와 일치하므로 무변경.
-- DDL 없음. 멱등 UPDATE(domain, meta_key). schema_registry(원 시드)·ext_meta_field_registry(런타임 정본) 둘 다.
-- 적용 순서: v297 이후. domain 은 general·medical 명시(v280/v290 스코프와 대칭·타 도메인 labels 오염 방지).
-- =============================================================================

-- 런타임 정본(041) — validate_ext_meta 가 읽는 테이블. general·medical 둘 다.
UPDATE ext_meta_field_registry
SET json_schema = '{"type":"array","items":{"type":"object","properties":{"label":{"type":"string"},"score":{"type":"number"}},"required":["label"]}}'::jsonb
WHERE domain IN ('general', 'medical') AND meta_key = 'labels';

-- 원 시드(039 v280) — 정합 유지.
UPDATE schema_registry
SET json_schema = '{"type":"array","items":{"type":"object","properties":{"label":{"type":"string"},"score":{"type":"number"}},"required":["label"]}}'::jsonb
WHERE domain IN ('general', 'medical') AND meta_key = 'labels';
