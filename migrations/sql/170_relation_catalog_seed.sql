-- =============================================================================
-- F-3.5 — 관계 카탈로그 기본 시드 (5종 kind + '일반' 토픽 + active relation_type)
-- =============================================================================
-- 카탈로그 테이블(relation_kind/topic_parent/subtopic/type)은 v140 에서 생성됐으나 비어 있다.
-- 관계 제안(LLM)이 active relation_type 을 참조하므로 기본 카탈로그를 시드한다.
-- PK 는 정적 시드라 gen_random_uuid()(v4) 사용(런타임 자산은 앱 uuidv7). 멱등(ON CONFLICT).
-- =============================================================================

INSERT INTO relation_kind (relation_kind_id, kind_code, kind_name_ko, description, is_symmetric, status)
VALUES
    (gen_random_uuid(), 'same_domain', '동일 도메인', '주제·분야가 같은 연결', true, 'active'),
    (gen_random_uuid(), 'same_series', '동일 시리즈', '같은 시리즈·연작·라인업 연결', true, 'active'),
    (gen_random_uuid(), 'duplicate_near', '유사 근접', '내용/주제/장면의 근접 유사', true, 'active'),
    (gen_random_uuid(), 'references', '참조', '명시적 인용·링크·제목 참조', false, 'active'),
    (gen_random_uuid(), 'derived_from', '파생', '한 콘텐츠가 다른 콘텐츠에서 파생', false, 'active')
ON CONFLICT (kind_code) DO NOTHING;

INSERT INTO relation_topic_parent (topic_id, topic_ko, topic_en)
VALUES (gen_random_uuid(), '일반', 'general')
ON CONFLICT (topic_ko, topic_en) DO NOTHING;

INSERT INTO relation_subtopic (subtopic_id, topic_id, subtopic_ko, subtopic_en)
SELECT gen_random_uuid(), p.topic_id, '', ''
FROM relation_topic_parent p
WHERE p.topic_ko = '일반' AND p.topic_en = 'general'
ON CONFLICT (topic_id, subtopic_ko, subtopic_en) DO NOTHING;

INSERT INTO relation_type (relation_type_id, relation_kind_id, relation_subtopic_id, status)
SELECT gen_random_uuid(), k.relation_kind_id, s.subtopic_id, 'active'
FROM relation_kind k
CROSS JOIN relation_subtopic s
JOIN relation_topic_parent p ON p.topic_id = s.topic_id
WHERE p.topic_ko = '일반' AND p.topic_en = 'general'
  AND s.subtopic_ko = '' AND s.subtopic_en = ''
  AND k.kind_code IN ('same_domain', 'same_series', 'duplicate_near', 'references', 'derived_from')
  AND k.status = 'active'
ON CONFLICT (relation_kind_id, relation_subtopic_id) DO NOTHING;
