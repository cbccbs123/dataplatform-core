import unittest
from unittest.mock import MagicMock


class TestReview(unittest.TestCase):
    def _conn(self, rowcount=1):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.rowcount = rowcount
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


class TestListEdgesForReview(unittest.TestCase):
    """FR-101/102/103 — status별 식별보강 페이징 조회(엣지 행 그대로·C6·의료 제외)."""

    def _conn(self, *, total=1, rows=None):
        """dict_row 커서 대역 — COUNT 1회 + rows 1회 순서로 fetchone/fetchall 를 돌려준다."""
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchone.return_value = {"count": total}
        cur.fetchall.return_value = rows if rows is not None else []
        conn.cursor.return_value = cur
        return conn, cur

    def _sample_row(self):
        return {
            "edge_id": "e1",
            "kind_code": "same_domain",
            "confidence": 0.9,
            "reason": "유사",
            "topic": {"topic_ko": "게임"},
            "status": "proposed",
            "reviewed_by": None,
            "reviewed_at": None,
            "src_asset_id": "as1",
            "src_fs_path": "/data/문서A.txt",
            "src_modality": "text",
            "dst_asset_id": "as2",
            "dst_fs_path": "/data/영상B.mp4",
            "dst_modality": "video",
        }

    def test_review_statuses_constant(self):
        from src.relations.review import _REVIEW_STATUSES
        self.assertEqual(_REVIEW_STATUSES, ("proposed", "active", "rejected"))

    def test_sql_binds_status_join_order_medical(self):
        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=7, rows=[self._sample_row()])
        list_edges_for_review(conn, status="proposed", limit=50, offset=10)
        # 마지막 execute = 행 조회 SQL(그 앞은 COUNT). 두 SQL 을 합쳐 검사한다.
        sqls = " ".join(str(c.args[0]) for c in cur.execute.call_args_list)
        self.assertIn("JOIN node", sqls)
        self.assertIn("JOIN asset", sqls)
        self.assertIn("IS DISTINCT FROM 'medical'", sqls)  # 양끝 의료 제외
        # tiebreaker 포함 결정적 정렬
        rows_sql = str(cur.execute.call_args_list[-1].args[0])
        self.assertIn("ORDER BY", rows_sql)
        self.assertIn("confidence DESC NULLS LAST", rows_sql)
        self.assertIn("edge_id", rows_sql.split("ORDER BY", 1)[1])
        self.assertIn("LIMIT", rows_sql)
        self.assertIn("OFFSET", rows_sql)
        # status 는 %s 바인딩(f-string 인젝션 아님)
        self.assertIn("status = %s", rows_sql)
        rows_params = cur.execute.call_args_list[-1].args[1]
        self.assertEqual(rows_params[0], "proposed")
        self.assertEqual(rows_params[-2], 50)  # limit
        self.assertEqual(rows_params[-1], 10)  # offset

    def test_row_shape_with_src_dst_and_basename(self):
        from src.relations.review import list_edges_for_review
        conn, _cur = self._conn(total=1, rows=[self._sample_row()])
        result = list_edges_for_review(conn, status="proposed", limit=50, offset=0)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["limit"], 50)
        self.assertEqual(result["offset"], 0)
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertEqual(row["edge_id"], "e1")
        self.assertEqual(row["kind_code"], "same_domain")
        self.assertEqual(row["confidence"], 0.9)
        self.assertEqual(row["reason"], "유사")
        self.assertEqual(row["topic"], {"topic_ko": "게임"})
        self.assertEqual(row["status"], "proposed")
        self.assertIsNone(row["reviewed_by"])
        self.assertIsNone(row["reviewed_at"])
        # src/dst 각 {asset_id, file_name(basename), modality}
        self.assertEqual(row["src"], {"asset_id": "as1", "file_name": "문서A.txt", "modality": "text"})
        self.assertEqual(row["dst"], {"asset_id": "as2", "file_name": "영상B.mp4", "modality": "video"})

    def test_count_uses_same_where(self):
        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=42, rows=[])
        result = list_edges_for_review(conn, status="active", limit=10, offset=0)
        self.assertEqual(result["total"], 42)
        count_sql = str(cur.execute.call_args_list[0].args[0])
        self.assertIn("COUNT", count_sql.upper())
        self.assertIn("status = %s", count_sql)
        self.assertIn("IS DISTINCT FROM 'medical'", count_sql)
        self.assertEqual(cur.execute.call_args_list[0].args[1][0], "active")
