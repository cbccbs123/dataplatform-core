import unittest
from unittest.mock import MagicMock


class TestCandidateThreshold(unittest.TestCase):
    def test_min_sim_param_added_to_having(self):
        from src.relations.asset_candidates import find_embedding_candidates
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchall.return_value = []
        conn.cursor.return_value = cur
        find_embedding_candidates(conn, source_asset_id="x", top_k=5, min_sim=0.3)
        sql = cur.execute.call_args[0][0]
        self.assertIn("HAVING", sql)
        params = cur.execute.call_args[0][1]
        self.assertIn(0.3, params)
