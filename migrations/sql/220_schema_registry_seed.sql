-- =============================================================================
-- F-4.13 — general 도메인 ext_meta 허용 키 시드 (src/skills/meta_split.py EXT_META_KEYS 7키).
-- 키-허용목록 검증(validate_ext_meta)의 기준. json_schema 는 placeholder(값 검증은 후속).
-- PK 는 정적 시드라 gen_random_uuid()(v170 관례). 멱등 ON CONFLICT(domain,meta_key).
-- 적용 순서: v210 이후.
-- =============================================================================

INSERT INTO schema_registry (schema_id, domain, meta_key, json_schema, description, status)
VALUES
    (gen_random_uuid(), 'general', 'summary',   '{"type":"string"}'::jsonb,                              '요약',        'active'),
    (gen_random_uuid(), 'general', 'keywords',  '{"type":"array","items":{"type":"string"}}'::jsonb,     '키워드',      'active'),
    (gen_random_uuid(), 'general', 'labels',    '{"type":"array"}'::jsonb,                                '라벨',        'active'),
    (gen_random_uuid(), 'general', 'objects',   '{"type":"array"}'::jsonb,                                '객체 후보',   'active'),
    (gen_random_uuid(), 'general', 'keyframes', '{"type":"array"}'::jsonb,                                '키프레임',    'active'),
    (gen_random_uuid(), 'general', 'stt',       '{"type":"string"}'::jsonb,                               'STT 전사',    'active'),
    (gen_random_uuid(), 'general', 'caption',   '{"type":"string"}'::jsonb,                               '캡션',        'active')
ON CONFLICT (domain, meta_key) DO NOTHING;
