import unittest
from unittest.mock import MagicMock


class TestReview(unittest.TestCase):
    def _conn(self, rowcount=1):
        conn = MagicMock(); cur = MagicMock()
        cur.__enter__.return_value = cur; cur.rowcount = rowcount
        conn.cursor.return_value = cur
        return conn, cur

    def test_approve_sets_active_and_reviewer_with_guard(self):
        from src.relations.review import approve_edge
        conn, cur = self._conn()
        self.assertTrue(approve_edge(conn, edge_id="e1", reviewer="bc"))
        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]
        self.assertIn("reviewed_by", sql)
        self.assertIn("status = 'proposed'", sql)  # 이미 결정된 엣지 재결정 방지 가드
        self.assertEqual(params[0], "active")
        self.assertEqual(params[1], "bc")

    def test_reject_sets_rejected(self):
        from src.relations.review import reject_edge
        conn, cur = self._conn()
        self.assertTrue(reject_edge(conn, edge_id="e1", reviewer="bc"))
        self.assertEqual(cur.execute.call_args[0][1][0], "rejected")

    def test_promote_kind_only_inactive(self):
        from src.relations.review import promote_relation_kind
        conn, cur = self._conn()
        self.assertTrue(promote_relation_kind(conn, kind_code="gaming_hardware", reviewer="bc"))
        self.assertIn("status='inactive'", cur.execute.call_args[0][0].replace(" ", ""))
