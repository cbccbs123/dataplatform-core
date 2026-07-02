-- 054 (v293): asset_lineage(activity, asset_id) 인덱스 — 스냅샷 버킷 relation_proposed EXISTS 가속.
-- query_assets(snapshot_bucket)·asset_stats(by_snapshot_bucket)의 상관 EXISTS
-- (l.asset_id = a.asset_id AND l.activity = 'relations.proposed.v1')가 인덱스 없이 seq scan → 상관 subplan
-- 비용추정 폭증(JIT 오작동 유발). (activity, asset_id) 복합 인덱스로 인덱스 프로브화(비용↓·JIT 회피).
-- activity 선두라 활동 접두 스캔에도 유용.
CREATE INDEX IF NOT EXISTS idx_asset_lineage_activity_asset
    ON asset_lineage (activity, asset_id);
