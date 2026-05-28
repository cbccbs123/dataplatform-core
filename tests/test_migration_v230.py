import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_DB_E2E") == "1", "실 DB 필요(RUN_DB_E2E=1)")
class TestMigrationV230(unittest.TestCase):
    def test_graph_edge_has_kind_and_topic_no_relation_type(self):
        from dotenv import load_dotenv
        load_dotenv(".env.dev", override=False)
        from src.config.settings import init_settings
        init_settings("dev")
        from src.database.postgres_util import PostgresUtil
        db = PostgresUtil()
        with db:
            with db.transaction() as conn:
                cols = {r[0] for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name='graph_edge'"
                ).fetchall()}
                self.assertIn("relation_kind_id", cols)
                self.assertIn("topic", cols)
                self.assertIn("reviewed_by", cols)
                self.assertIn("reviewed_at", cols)
                self.assertNotIn("relation_type_id", cols)
                # status CHECK 정의가 proposed 포함·superseded 제외인지 확인
                chk = conn.execute(
                    "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                    "WHERE conname = 'graph_edge_status_check'"
                ).fetchone()
                self.assertIsNotNone(chk)
                self.assertIn("proposed", chk[0])
                self.assertNotIn("superseded", chk[0])
                dropped = {r[0] for r in conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_name IN "
                    "('relation_type','relation_subtopic','relation_topic_parent')"
                ).fetchall()}
                self.assertEqual(dropped, set())
                self.assertIsNotNone(conn.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name='relation_kind'"
                ).fetchone())
