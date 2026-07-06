-- 058 (v295): 관계 topic 정규화 — 성장하는 정본 레지스트리 + 해소 캐시 2테이블.
-- 배경: 관계 생성(run_relations)은 LLM 이 topic_ko/subtopic_ko 를 자유 기입해 동의어(요리/음식/식품)·
--   계층 불일치·모달리티 누수로 주제가 난립한다(spec 058 §개요·FR-1xx). 정본 어휘 집합과
--   해소 결과 캐시를 데이터로 두어 "정확일치→kNN→LLM 판정→캐시 동결"(C2) 결정성 파이프라인을 받친다.
-- topic_registry : 정본 topic 어휘의 진짜 목록(성장). 라벨 임베딩(1536D)을 kNN 후보 회수용으로 보관.
-- topic_alias    : raw_ko → canonical_ko 해소 캐시. LLM 판정 결과를 동결해 재실행 결정성(헌법 3조)을 보장.
-- 제약: 임베딩 1536D 고정·PG17+pgvector·UUIDv7 PK(앱 발급, src/database/ids.uuid7 — 런타임 자산 관례).
-- 멱등(IF NOT EXISTS) — 재적용·부트스트랩 안전. pgvector cosine 인덱스는 repo 관례(hnsw·vector_cosine_ops).

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- topic_registry: 정본 topic 어휘(성장하는 레지스트리). topic_ko 가 정본 라벨(유니크).
--   topic_id 는 앱에서 발급한 UUIDv7(런타임 등록 — asset 등 런타임 테이블 관례, DB default 없음).
--   embedding 은 라벨 문자열의 텍스트 임베딩(st_bge·1536D) — kNN 후보 회수용(등록 시 1회 저장).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS topic_registry (
    topic_id   UUID PRIMARY KEY,
    topic_ko   TEXT NOT NULL UNIQUE,
    topic_en   TEXT,
    embedding  VECTOR(1536),
    source     TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  topic_registry            IS '정본 topic 어휘 레지스트리(성장) — spec 058 FR-101.';
COMMENT ON COLUMN topic_registry.topic_id   IS 'PK — 앱 발급 UUIDv7(src/database/ids.uuid7).';
COMMENT ON COLUMN topic_registry.topic_ko   IS '정본 topic 라벨(유니크). alias.canonical_ko 가 참조.';
COMMENT ON COLUMN topic_registry.topic_en   IS '정본 topic 영문(topic_ko 1개당 1개 고정 — FR-204).';
COMMENT ON COLUMN topic_registry.embedding  IS '라벨 임베딩 1536D(st_bge) — kNN 후보 회수용.';
COMMENT ON COLUMN topic_registry.source     IS '등록 출처: seed/auto/batch.';

-- ---------------------------------------------------------------------------
-- topic_alias: 해소 캐시(raw_ko → canonical_ko). 정확일치 룩업의 저장소이자 LLM 판정 결과 동결처.
--   canonical_ko 는 반드시 정본(topic_registry.topic_ko)을 가리킨다(FK).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS topic_alias (
    raw_ko       TEXT PRIMARY KEY,
    canonical_ko TEXT NOT NULL REFERENCES topic_registry (topic_ko),
    decided_by   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  topic_alias              IS '관계 topic 해소 캐시(결정성 동결) — spec 058 FR-102.';
COMMENT ON COLUMN topic_alias.raw_ko       IS 'PK — 원본(자유 기입) topic 라벨.';
COMMENT ON COLUMN topic_alias.canonical_ko IS '해소된 정본(topic_registry.topic_ko) FK.';
COMMENT ON COLUMN topic_alias.decided_by   IS '해소 근거: exact/embed/llm/seed.';

-- ---- HNSW: 1536D 라벨 임베딩 코사인 유사도(kNN 후보 회수) ---------------------
-- IVFFlat 대신 HNSW — 빈 테이블에서도 학습 불필요(생성 순서 무관)·lists 튜닝 불필요.
-- asset_embedding(v130)·media_chunks 의 hnsw 인덱스와 일관. 기본 파라미터(m=16, ef_construction=64).
CREATE INDEX IF NOT EXISTS idx_topic_registry_embedding_hnsw
    ON topic_registry USING hnsw (embedding vector_cosine_ops);
