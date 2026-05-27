-- =============================================================================
-- v2 단계 C — 공용 그래프 계층 (node + graph_edge). asset_relation (가) 흡수.
-- 엣지 종류는 기존 relation_type 카탈로그를 재사용(graph_edge.relation_type_id).
-- 적용 순서: 170(카탈로그 시드) 이후. match_*·relation_* 카탈로그·asset_* 는 유지.
-- 레거시 asset_relation 드롭은 코드 컷오버 후 별도 마이그레이션에서.
-- =============================================================================

-- 통합 노드: 자산 노드 + 엔티티 노드(엔티티는 단계 D 의료 ER 에서 적재).
CREATE TABLE IF NOT EXISTS node (
    node_id     UUID PRIMARY KEY,
    node_kind   VARCHAR(20) NOT NULL CHECK (node_kind IN ('asset', 'entity')),
    asset_id    UUID REFERENCES asset (asset_id) ON DELETE CASCADE,   -- node_kind='asset'
    entity_type VARCHAR(40),                                          -- node_kind='entity'
    entity_uid  VARCHAR(255),
    canonical   JSONB NOT NULL DEFAULT '{}'::jsonb,
    status      VARCHAR(20) NOT NULL DEFAULT 'active',
    promoted_to UUID REFERENCES node (node_id) ON DELETE SET NULL,    -- 옵션 B case→patient
    promoted_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_node_kind CHECK (
        (node_kind = 'asset'  AND asset_id IS NOT NULL AND entity_type IS NULL) OR
        (node_kind = 'entity' AND entity_type IS NOT NULL)
    )
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_node_asset  ON node (asset_id) WHERE node_kind = 'asset';
CREATE UNIQUE INDEX IF NOT EXISTS uq_node_entity ON node (entity_type, entity_uid) WHERE node_kind = 'entity';

-- 통합 엣지: asset↔asset(일반)/asset→entity/entity→entity(의료 D). relation_type 카탈로그 참조.
CREATE TABLE IF NOT EXISTS graph_edge (
    edge_id          UUID PRIMARY KEY,
    src_node         UUID NOT NULL REFERENCES node (node_id) ON DELETE CASCADE,
    dst_node         UUID NOT NULL REFERENCES node (node_id) ON DELETE CASCADE,
    relation_type_id UUID NOT NULL REFERENCES relation_type (relation_type_id) ON DELETE RESTRICT,
    confidence       DOUBLE PRECISION,
    decision_id      UUID,   -- match_decision 느슨 참조(의료 ER, 일반=NULL; FK 미설정)
    reason           TEXT,
    status           VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'rejected')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ,
    CONSTRAINT chk_graph_edge_no_self CHECK (src_node <> dst_node),
    CONSTRAINT uq_graph_edge UNIQUE (src_node, dst_node, relation_type_id)
);
CREATE INDEX IF NOT EXISTS idx_graph_edge_src  ON graph_edge (src_node);
CREATE INDEX IF NOT EXISTS idx_graph_edge_dst  ON graph_edge (dst_node);
CREATE INDEX IF NOT EXISTS idx_graph_edge_type ON graph_edge (relation_type_id);

COMMENT ON TABLE node IS '통합 노드(자산 노드 + 엔티티 노드). 일반 관계·의료 KG 공용. asset_relation/entity 흡수.';
COMMENT ON COLUMN node.node_kind   IS 'asset(asset_id 채움) | entity(entity_type/uid 채움).';
COMMENT ON COLUMN node.asset_id    IS 'node_kind=asset 일 때 FK→asset(자산당 1노드).';
COMMENT ON COLUMN node.entity_type IS 'node_kind=entity 일 때 종류(patient/study/topic/case…, 단계 D).';
COMMENT ON COLUMN node.promoted_to IS '옵션 B case→patient 승격 대상 node_id.';
COMMENT ON TABLE graph_edge IS '통합 엣지(node↔node). relation_type 카탈로그 참조. asset_relation 흡수.';
COMMENT ON COLUMN graph_edge.relation_type_id IS 'FK→relation_type(공유 카탈로그). 엣지 종류.';
COMMENT ON COLUMN graph_edge.decision_id      IS 'match_decision 느슨 참조(의료 ER 증거, 일반=NULL).';
