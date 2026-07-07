-- 058 v2 (v297): subtopic 부모 스코프 — topic_registry·topic_alias 에 parent_topic 추가.
-- 배경(ADR 2026-07-07 닫힌 분류체계 전환): topic 층은 닫힌 27+기타(parent_topic IS NULL),
--   subtopic 층은 열린 성장·부모 topic 스코프(parent_topic = 부모 topic_ko)로 2층 분리한다.
--   동의어 정리를 부모 안에서만 수행해 동음이의(교통>사고 ≠ 사회>사고)를 보존하고, 오병합의
--   폭발 반경을 부모 버킷 안에 가둔다(spec 058 v2 FR-102v2·C3).
--
-- 스코프 유니크(부분 유니크 인덱스 2개 — 층마다). PostgreSQL 은 컬럼 NULL 을 서로 다른 값으로
--   취급하므로 (parent_topic, *_ko) 단일 인덱스로는 topic 층(parent NULL)의 유일성을 못 지킨다.
--   → 층을 술어로 가른 **부분 유니크 인덱스 2개**로 분리한다:
--     registry: topic 층 = (topic_ko) WHERE parent_topic IS NULL /
--               subtopic 층 = (parent_topic, topic_ko) WHERE parent_topic IS NOT NULL.
--     alias   : topic 층 = (raw_ko)  WHERE parent_topic IS NULL /
--               subtopic 층 = (parent_topic, raw_ko)  WHERE parent_topic IS NOT NULL.
--   → 기존 단일 유니크(topic_registry.topic_ko UNIQUE·topic_alias.raw_ko PK)는 부모 스코프와
--     충돌하므로 드롭하고 위 부분 유니크 인덱스로 대체한다(downgrade 에서 원복·가역).
--
-- FK 결정(완화·정당화 — migration-reviewer 근거):
--   기존 alias.canonical_ko → registry.topic_ko 단순 FK 는 부모 스코프에서 유지 불가·부정확.
--   (1) FK 대상이던 registry.topic_ko UNIQUE 를 **부분 유니크 인덱스**로 대체하는데, PostgreSQL 은
--       부분 인덱스를 FK 참조 대상으로 삼을 수 없다(비-부분 유니크 제약/인덱스만 FK 대상).
--   (2) 대안인 복합 FK (parent_topic, canonical_ko)→(parent_topic, topic_ko) 는 MATCH SIMPLE 기본상
--       참조 컬럼 중 NULL 이 하나라도 있으면 검사를 스킵한다 → parent_topic 이 NULL 인 topic 층
--       alias 는 사실상 FK 미검사라 무의미하다.
--   → 따라서 FK 를 **드롭하고 앱 불변식으로 완화**한다: canonicalize seam 이 register_topic 등록
--     이후에만 alias 를 동결(_freeze_alias)하므로 canonical_ko 는 항상 같은 스코프의 정본을
--     가리킨다(src/relations/topic_canonicalize.py 코드 보증). downgrade 에서 단순 FK 를 원복(가역).
--
-- 멱등(ADD COLUMN IF NOT EXISTS·DROP CONSTRAINT IF EXISTS·CREATE UNIQUE INDEX IF NOT EXISTS) —
--   재적용·부트스트랩 안전. down 은 alembic v297 downgrade() 가 대칭 원복(가역).

-- ---------------------------------------------------------------------------
-- 1) parent_topic 컬럼 추가(멱등) — topic 층 NULL·subtopic 층 = 부모 topic_ko.
-- ---------------------------------------------------------------------------
ALTER TABLE topic_registry ADD COLUMN IF NOT EXISTS parent_topic TEXT;
ALTER TABLE topic_alias    ADD COLUMN IF NOT EXISTS parent_topic TEXT;

COMMENT ON COLUMN topic_registry.parent_topic IS
    '부모 topic_ko(subtopic 층). NULL 이면 topic 층(닫힌 27+기타). spec 058 v2 FR-102v2.';
COMMENT ON COLUMN topic_alias.parent_topic IS
    '부모 topic_ko(subtopic 층 스코프). NULL 이면 topic 층 매핑. spec 058 v2 FR-102v2.';

-- ---------------------------------------------------------------------------
-- 2) FK 완화: alias.canonical_ko FK 드롭(위 결정 (1)(2)). registry.topic_ko UNIQUE 드롭 의존도 해제.
-- ---------------------------------------------------------------------------
ALTER TABLE topic_alias DROP CONSTRAINT IF EXISTS topic_alias_canonical_ko_fkey;

-- ---------------------------------------------------------------------------
-- 3) registry: topic_ko 단일 UNIQUE 드롭 → 부모 스코프 부분 유니크 인덱스 2개.
-- ---------------------------------------------------------------------------
ALTER TABLE topic_registry DROP CONSTRAINT IF EXISTS topic_registry_topic_ko_key;
CREATE UNIQUE INDEX IF NOT EXISTS uq_topic_registry_topic_ko_root
    ON topic_registry (topic_ko) WHERE parent_topic IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_topic_registry_parent_sub
    ON topic_registry (parent_topic, topic_ko) WHERE parent_topic IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 4) alias: raw_ko PK 드롭 → 부모 스코프 부분 유니크 인덱스 2개(registry 와 동형).
-- ---------------------------------------------------------------------------
ALTER TABLE topic_alias DROP CONSTRAINT IF EXISTS topic_alias_pkey;
CREATE UNIQUE INDEX IF NOT EXISTS uq_topic_alias_raw_ko_root
    ON topic_alias (raw_ko) WHERE parent_topic IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_topic_alias_parent_raw
    ON topic_alias (parent_topic, raw_ko) WHERE parent_topic IS NOT NULL;
