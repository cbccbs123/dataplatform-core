-- =============================================================================
-- v2 단계 C 정리 — 일반 도메인(C)에서 쓰지 않는 테이블 드랍.
-- 레거시: asset_relation(→ graph_edge 흡수).
-- 의료/ER/옵션B(단계 D 착수 시 해당 단계 마이그레이션에서 재도입):
--   entity / entity_edge / asset_entity_link / match_candidate / match_decision /
--   match_evidence / er_policy / review_queue / unresolved_pool / content_cluster.
-- 유지: asset 코어 계층 + node/graph_edge + 관계 카탈로그(relation_*).
-- 적용 순서: v200 이후.
-- =============================================================================

DROP TABLE IF EXISTS asset_relation CASCADE;
DROP TABLE IF EXISTS asset_entity_link CASCADE;
DROP TABLE IF EXISTS entity_edge CASCADE;
DROP TABLE IF EXISTS entity CASCADE;
DROP TABLE IF EXISTS match_evidence CASCADE;
DROP TABLE IF EXISTS match_decision CASCADE;
DROP TABLE IF EXISTS match_candidate CASCADE;
DROP TABLE IF EXISTS content_cluster CASCADE;
DROP TABLE IF EXISTS unresolved_pool CASCADE;
DROP TABLE IF EXISTS review_queue CASCADE;
DROP TABLE IF EXISTS er_policy CASCADE;
