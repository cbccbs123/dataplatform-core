-- v296: 스키마 자기문서화 — 실 테이블 미기입 컬럼 COMMENT 25개 보강 + 2026-07-01 임시 백업 테이블 6개 정리.
-- 배경: 실 스키마 14테이블은 테이블-레벨 COMMENT 는 전부 있으나 일부 컬럼 COMMENT 가 비어 있었다(감사 25컬럼/7테이블).
--   스키마를 자기설명적으로 만들기 위해 누락 컬럼에 COMMENT 를 채운다(데이터/제약 무변경 — 메타주석만).
-- 백업 정리: `*_bak_20260701` 6개는 2026-07-01 관계 재생성(relregen) 전 수동 백업으로 alembic 관리 밖·FK 참조 0.
--   스키마가 아니므로 DROP 으로 정리한다(IF EXISTS — 신규/부트스트랩 DB 에는 애초에 없어 no-op).
-- 가역성: COMMENT 보강은 downgrade 에서 IS NULL 로 원복(전부 직전 NULL 이었음). 백업 DROP 은 **비가역 정리**
--   (임시 백업 데이터는 복원 대상 아님·downgrade 는 재생성하지 않음). DDL 데이터/컬럼/제약 무변경.

-- ============================ 컬럼 COMMENT 보강 ============================

-- ext_meta_field_registry (OM 런타임 ext_meta 정본 — spec 041)
COMMENT ON COLUMN ext_meta_field_registry.domain      IS '도메인(general/medical 등) — (domain, meta_key) 복합으로 ext_meta 필드를 정의.';
COMMENT ON COLUMN ext_meta_field_registry.meta_key    IS 'ext_meta 필드 키(도메인 내 유일).';
COMMENT ON COLUMN ext_meta_field_registry.json_schema IS '해당 필드 값 검증용 JSON Schema(039 값 검증).';
COMMENT ON COLUMN ext_meta_field_registry.description IS '필드 설명(사람용 문서).';
COMMENT ON COLUMN ext_meta_field_registry.status      IS '필드 상태(active 등).';
COMMENT ON COLUMN ext_meta_field_registry.access_tier IS '필드 접근 등급(040 access_tier — public/authenticated/authorized/regulated·CHECK 4값).';
COMMENT ON COLUMN ext_meta_field_registry.created_at  IS '생성 시각.';

-- graph_edge (관계 엣지 — relation_kind 직접 참조 + topic jsonb)
COMMENT ON COLUMN graph_edge.edge_id    IS 'PK — 엣지 식별자(UUIDv7).';
COMMENT ON COLUMN graph_edge.src_node   IS '출발 노드(node.node_id) FK.';
COMMENT ON COLUMN graph_edge.dst_node   IS '도착 노드(node.node_id) FK. 대칭 엣지는 (src,dst) 정규화로 1행 수렴.';
COMMENT ON COLUMN graph_edge.confidence IS '관계 신뢰도 0~1(LLM 제안·자동승인 임계 비교).';
COMMENT ON COLUMN graph_edge.reason     IS '관계 근거 한 줄(LLM·비식별).';
COMMENT ON COLUMN graph_edge.created_at IS '생성 시각.';
COMMENT ON COLUMN graph_edge.updated_at IS '갱신 시각(ON CONFLICT 재제안 시 confidence/topic 갱신 — spec 032).';

-- node (통합 노드 — asset 노드 + entity 노드[의료 ER·단계 D])
COMMENT ON COLUMN node.node_id     IS 'PK — 노드 식별자(UUIDv7).';
COMMENT ON COLUMN node.entity_uid  IS '엔티티 노드의 외부 식별자(entity kind·의료 ER 등). asset 노드는 NULL.';
COMMENT ON COLUMN node.canonical   IS '엔티티 정규화 속성(jsonb·entity 노드).';
COMMENT ON COLUMN node.status      IS '노드 상태(active/proposed 등).';
COMMENT ON COLUMN node.promoted_at IS '엔티티 승격 시각(proposed→active).';
COMMENT ON COLUMN node.created_at  IS '생성 시각.';

-- relation_resolution (관계 재시도/미해소 큐 — spec 009·035)
COMMENT ON COLUMN relation_resolution.created_at IS '큐 등록 시각.';
COMMENT ON COLUMN relation_resolution.updated_at IS '상태 갱신 시각(pending/resolved/failed/isolated 전이).';

-- schema_registry (레거시 ext_meta 정본 — main 호환 병행)
COMMENT ON COLUMN schema_registry.access_tier IS '스키마 접근 등급(040 access_tier·레거시 병행).';

-- topic_registry / topic_alias (spec 058 — created_at 보강)
COMMENT ON COLUMN topic_registry.created_at IS '등록 시각.';
COMMENT ON COLUMN topic_alias.created_at    IS '해소 캐시 기록 시각.';

-- ============================ 임시 백업 테이블 정리 ============================
-- 2026-07-01 relregen 전 수동 백업(alembic 관리 밖·FK 참조 0). 스키마 아님 → 정리.
DROP TABLE IF EXISTS asset_lineage_both_bak_20260701;
DROP TABLE IF EXISTS asset_lineage_relations_bak_20260701;
DROP TABLE IF EXISTS graph_edge_bak_20260701;
DROP TABLE IF EXISTS graph_edge_bothonly_bak_20260701;
DROP TABLE IF EXISTS graph_edge_stfix_bak_20260701;
DROP TABLE IF EXISTS relation_resolution_bak_20260701;
