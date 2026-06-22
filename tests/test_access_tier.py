"""F-4.13 ext_meta access_tier 단위 테스트 (spec 040 Wave 1)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from src.registry.access_tier import (
    AUTHENTICATED,
    AUTHORIZED,
    PUBLIC,
    REGULATED,
    domain_floor,
    max_tier,
)
from src.registry.schema_registry import fetch_access_tiers


class DomainFloorTest(unittest.TestCase):
    def test_general_has_no_floor(self):
        self.assertIsNone(domain_floor("general"))

    def test_medical_floor_is_regulated(self):
        self.assertEqual(domain_floor("medical"), REGULATED)

    def test_review_floor_is_authorized(self):
        self.assertEqual(domain_floor("review"), AUTHORIZED)


class MaxTierTest(unittest.TestCase):
    def test_ordinal_max(self):
        self.assertEqual(max_tier(AUTHENTICATED, AUTHORIZED), AUTHORIZED)

    def test_regulated_beats_all(self):
        self.assertEqual(max_tier(PUBLIC, AUTHENTICATED, REGULATED, AUTHORIZED), REGULATED)

    def test_skips_none(self):
        self.assertEqual(max_tier(None, AUTHENTICATED, None), AUTHENTICATED)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            max_tier(None, None)


class FetchAccessTiersTest(unittest.TestCase):
    def test_returns_active_tier_map(self):
        rows = [
            {"meta_key": "summary", "access_tier": AUTHENTICATED},
            {"meta_key": "stt", "access_tier": AUTHORIZED},
        ]
        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = rows
        out = fetch_access_tiers(conn, "general")
        self.assertEqual(out, {"summary": AUTHENTICATED, "stt": AUTHORIZED})
        sql = cur.execute.call_args.args[0]
        self.assertIn("schema_registry", sql)
        self.assertIn("access_tier", sql)
        self.assertEqual(cur.execute.call_args.args[1], ("general",))


_RUN = os.getenv("RUN_DB_E2E") == "1"


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만(실 PostgreSQL·v290 head)")
class SchemaRegistryAccessTierDbE2eTest(unittest.TestCase):
    db = None  # type: ignore[assignment]

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv

        load_dotenv(".env.dev", override=False)
        from src.config.settings import init_settings

        init_settings("dev")
        from src.database.postgres_util import PostgresUtil

        try:
            cls.db = PostgresUtil()
            cls.db.__enter__()
            with cls.db.transaction() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name='schema_registry' AND column_name='access_tier'"
                )
                if cur.fetchone() is None:
                    raise unittest.SkipTest("access_tier 컬럼 없음(v290 미적용)")
        except unittest.SkipTest:
            raise
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"DB 미접속: {type(exc).__name__}: {exc}") from None

    @classmethod
    def tearDownClass(cls):
        if cls.db is not None:
            cls.db.__exit__(None, None, None)

    def test_general_stt_is_authorized(self):
        with self.db.transaction() as conn:
            tiers = fetch_access_tiers(conn, "general")
        self.assertEqual(tiers.get("stt"), AUTHORIZED)

    def test_medical_summary_is_regulated(self):
        with self.db.transaction() as conn:
            tiers = fetch_access_tiers(conn, "medical")
        self.assertEqual(tiers.get("summary"), REGULATED)
