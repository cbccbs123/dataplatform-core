import unittest
from unittest.mock import MagicMock


class TestFetchActiveKinds(unittest.TestCase):
    def _conn_returning(self, rows):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchall.return_value = rows
        conn.cursor.return_value = cur
        return conn, cur

    def test_fetch_active_relation_kinds_filters_legacy_and_inactive(self):
        from src.relations.relation_type_catalog import fetch_active_relation_kinds
        rows = [{"type_code": "duplicate_near", "type_name": "유사 근접", "description": "..."}]
        conn, cur = self._conn_returning(rows)
        out = fetch_active_relation_kinds(conn)
        self.assertEqual(out[0]["type_code"], "duplicate_near")
        sql = cur.execute.call_args[0][0]
        self.assertIn("status = 'active'", sql)
        self.assertIn("relation_kind", sql)
        self.assertNotIn("relation_type", sql)  # 조합 테이블 미참조
