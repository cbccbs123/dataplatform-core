-- =============================================================================
-- C+ 관계 보강 — graph_edge 가 relation_kind 를 직접 참조 + 주제는 엣지 topic jsonb.
-- relation_type / relation_subtopic / relation_topic_parent 3테이블 드롭(데카르트·inactive 제거).
-- 적용 순서: v220 이후. relation_kind 는 유지(통제 어휘).
-- =============================================================================

-- 1) 신규 컬럼: relation_kind 직접 참조 + topic jsonb 속성
ALTER TABLE graph_edge
    ADD COLUMN IF NOT EXISTS relation_kind_id UUID REFERENCES relation_kind (relation_kind_id) ON DELETE RESTRICT,
    ADD COLUMN IF NOT EXISTS topic JSONB NOT NULL DEFAULT '{}'::jsonb;

-- 2) 백필: 기존 relation_type → kind + topic jsonb 분해
UPDATE graph_edge ge
   SET relation_kind_id = rt.relation_kind_id
  FROM relation_type rt
 WHERE rt.relation_type_id = ge.relation_type_id;

UPDATE graph_edge ge
   SET topic = jsonb_build_object(
         'topic_ko',    COALESCE(tp.topic_ko, ''),
         'subtopic_ko', COALESCE(s.subtopic_ko, ''),
         'topic_en',    COALESCE(tp.topic_en, ''),
         'subtopic_en', COALESCE(s.subtopic_en, ''))
  FROM relation_type rt
  JOIN relation_subtopic s ON s.subtopic_id = rt.relation_subtopic_id
  JOIN relation_topic_parent tp ON tp.topic_id = s.topic_id
 WHERE rt.relation_type_id = ge.relation_type_id;

ALTER TABLE graph_edge ALTER COLUMN relation_kind_id SET NOT NULL;

-- 3) 엣지 식별자/인덱스 교체: relation_type_id → relation_kind_id
ALTER TABLE graph_edge DROP CONSTRAINT IF EXISTS uq_graph_edge;
ALTER TABLE graph_edge ADD  CONSTRAINT uq_graph_edge_kind UNIQUE (src_node, dst_node, relation_kind_id);
DROP INDEX IF EXISTS idx_graph_edge_type;
CREATE INDEX IF NOT EXISTS idx_graph_edge_kind ON graph_edge (relation_kind_id);

-- 4) 구 컬럼/카탈로그 드롭 (FK 순서: graph_edge.relation_type_id → relation_type → subtopic → topic_parent)
ALTER TABLE graph_edge DROP COLUMN IF EXISTS relation_type_id;
DROP TABLE IF EXISTS relation_type;
DROP TABLE IF EXISTS relation_subtopic;
DROP TABLE IF EXISTS relation_topic_parent;

-- 5) HITL 검토 상태: status 재정의(proposed/active/rejected, 'superseded' 삭제) + 감사 컬럼
--    기존 행은 'active'(과거 자동확정)라 새 CHECK 통과. 신규 insert 기본은 'proposed'.
ALTER TABLE graph_edge ADD COLUMN IF NOT EXISTS reviewed_by VARCHAR(255);
ALTER TABLE graph_edge ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;
ALTER TABLE graph_edge DROP CONSTRAINT IF EXISTS graph_edge_status_check;  -- 원 인라인 CHECK 자동명
ALTER TABLE graph_edge ALTER COLUMN status SET DEFAULT 'proposed';
ALTER TABLE graph_edge ADD CONSTRAINT graph_edge_status_check
    CHECK (status IN ('proposed', 'active', 'rejected'));

COMMENT ON COLUMN graph_edge.relation_kind_id IS 'FK→relation_kind(통제 어휘). 엣지 종류(성격).';
COMMENT ON COLUMN graph_edge.topic IS '주제 속성 jsonb: {topic_ko,subtopic_ko,topic_en,subtopic_en}. 비정규화.';
COMMENT ON COLUMN graph_edge.status IS 'HITL 검토 상태: proposed(LLM 제안·미검토) | active(승인/자동승인) | rejected(반려).';
COMMENT ON COLUMN graph_edge.reviewed_by IS '검토자 식별자(승인/반려 시).';
COMMENT ON COLUMN graph_edge.reviewed_at IS '검토 시각.';
