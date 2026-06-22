-- =============================================================================
-- F-4.13 — general ext_meta json_schema 값 검증용 완성 (spec 039).
-- v220 placeholder: labels/objects/keyframes 가 {"type":"array"} 만 있음 → items 보완.
-- DDL 없음. 멱등 UPDATE(domain, meta_key).
-- 적용 순서: v270 이후.
-- =============================================================================

UPDATE schema_registry
SET json_schema = '{"type":"array","items":{"type":"string"}}'::jsonb
WHERE domain = 'general' AND meta_key = 'labels';

UPDATE schema_registry
SET json_schema = '{"type":"array","items":{"type":"string"}}'::jsonb
WHERE domain = 'general' AND meta_key = 'objects';

UPDATE schema_registry
SET json_schema = '{"type":"array","items":{"type":"object"}}'::jsonb
WHERE domain = 'general' AND meta_key = 'keyframes';
