-- =============================================================================
-- 자산 자기주제(aboutness) 정본 테이블 — asset_topic (spec 065 · FR-101/102).
--
-- 배경: 지금까지 자산의 주제(topic/subtopic)는 자산이 스스로 갖지 않고, 검색·포탈이 쓰는 주제는
--   관계 이웃 엣지(graph_edge.topic)를 자산으로 투영해 만들었다(topic_query.project_asset_topics).
--   엣지 topic 은 관계 LLM 이 쌍(pairwise) 비교 중 붙이는 라벨이라 상대 자산 쪽으로 치우치고,
--   어떤 이웃 엣지가 활성 임계를 넘느냐(이웃 운)에 따라 자산 주제가 흔들려 오염됐다(농구 사례).
-- 처방(065): 주제를 **자산 자기 내용(summary/keywords/labels)에서 1회 확정한 정본**으로 분리 저장한다.
--   하이브리드 분류(임베딩 kNN → LLM 닫힌 확정 → 058 canonicalize)의 결과를 이 테이블에 upsert 한다.
--
-- FR-101: asset_id 를 PK 로 두어 자산당 primary 주제 1건(+ subtopic 1건). **행 부재 = "주제 미부여"**
--   (소비처는 topics 필드 생략 — 현행 "topics 없음" 자산과 동일 경로). 자산 삭제 시 ON DELETE CASCADE.
--   ※ 자연키 PK(asset_id 승계)는 헌법 6조 UUIDv7 PK 정책의 **의도된 예외**(1:1 per-asset 파생 테이블 —
--   250 relation_resolution 과 동일 패턴). 값 자체는 asset.asset_id(UUIDv7) 승계라 불변식 충족.
-- FR-102: topic_ko/topic_en 은 topic_registry topic 층(닫힌 28 정본)의 값만 허용한다 — **앱 레벨 검증**
--   (라벨 개명 유연성을 위해 FK 대신 적재 시 검증). subtopic 은 열림 + 058 canonicalize 결과.
-- FR-601: policy_version 을 행에 기록 — 프롬프트/후보수 변경 시 버전 증가로 재현성 추적.
-- 멱등: upsert(ON CONFLICT (asset_id) DO UPDATE)로 백필/재실행이 결정적(SC-05).
-- 적용 순서: v298 이후. 다른 테이블 무접촉(신규 테이블·인덱스만 추가).
-- =============================================================================

CREATE TABLE IF NOT EXISTS asset_topic (
    asset_id       UUID PRIMARY KEY REFERENCES asset (asset_id) ON DELETE CASCADE,
    topic_ko       TEXT NOT NULL,
    topic_en       TEXT,
    subtopic_ko    TEXT,
    subtopic_en    TEXT,
    confidence     DOUBLE PRECISION,
    decided_by     VARCHAR(20) NOT NULL DEFAULT 'hybrid',
    policy_version VARCHAR(40) NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ
);

-- 파생 조인·패싯용 — 같은 (topic_ko, subtopic_ko) 자산 묶음(find_same_topic_groups)·주제 패싯 조회 가속.
CREATE INDEX IF NOT EXISTS idx_asset_topic_pair ON asset_topic (topic_ko, subtopic_ko);

COMMENT ON TABLE asset_topic IS
    '자산 자기주제 정본(065·FR-101) — 자기 내용 하이브리드 분류 결과. 행 부재=미부여.';
COMMENT ON COLUMN asset_topic.topic_ko IS
    'topic_registry topic 층(닫힌 28) 정본 topic_ko — 앱 레벨 검증(FR-102).';
COMMENT ON COLUMN asset_topic.topic_en IS
    'topic_ko 의 영문 정본(registry 해소·NULL 허용 — registry topic_en 자체가 nullable).';
COMMENT ON COLUMN asset_topic.subtopic_ko IS
    '자산의 구체 주제어(열림) — 058 canonicalize_subtopic 정규화 결과. NULL=하위주제 없음.';
COMMENT ON COLUMN asset_topic.confidence IS
    '분류 LLM confidence(0~1·참고용) — 주제 채택 여부와 무관(닫힌집합 검증이 게이트).';
COMMENT ON COLUMN asset_topic.decided_by IS
    '판정 경로 어휘: hybrid(임베딩 kNN→LLM 확정·기본). 확장 여지: embed/manual.';
COMMENT ON COLUMN asset_topic.policy_version IS
    '분류 정책 버전(예 asset_topic.v1) — 프롬프트/후보수 변경 추적(FR-601).';
