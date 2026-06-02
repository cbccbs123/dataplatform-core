-- =============================================================================
-- v240 — 자산 그룹화 철회: asset_group / asset.group_id / idx_asset_group_id 제거
-- =============================================================================
-- 경로 기반 자동 그룹화(구 spec 007)를 철회한다. 다운로드 "묶음"의 권위 정의는
-- 관계(graph_edge) 기반 ego-network(seed + 1-hop active 이웃)로 단일화된다
-- (spec 010 US3·FR-008). group_id 를 읽는 코드는 없다(소비처 0).
-- 의료 환자·검사 단위 묶음은 단계 D 엔티티 ER(node(node_kind='entity'), 011/014)이 담당.
-- 근거 ADR: docs/decisions/2026-06-02-relation-based-download-bundles.md
--
-- 드롭 순서(FK 의존 역순): 인덱스 → asset.group_id 컬럼(FK 동반) → asset_group 테이블.
-- =============================================================================

DROP INDEX IF EXISTS idx_asset_group_id;

-- asset.group_id 컬럼 제거 — REFERENCES asset_group(group_id) FK 도 함께 제거된다.
ALTER TABLE asset DROP COLUMN IF EXISTS group_id;

-- 더는 이 테이블을 참조하는 컬럼이 없으므로 안전하게 제거.
DROP TABLE IF EXISTS asset_group;
