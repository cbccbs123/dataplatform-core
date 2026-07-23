"""F-4.13 ext_meta 필드 레지스트리 단위·e2e 테스트 (spec 039·041)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from src.registry.ext_meta_field_registry import (
    ExtMetaValidationError,
    check_ext_meta_values,
    fetch_ext_key_schemas,
    validate_ext_meta,
)

_STRING = {"type": "string"}
_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}
_OBJECT_ARRAY = {"type": "array", "items": {"type": "object"}}


class CheckExtMetaValuesTest(unittest.TestCase):
    def test_valid_string_and_string_array(self):
        schemas = {"summary": _STRING, "keywords": _STRING_ARRAY}
        ext = {"summary": "요약", "keywords": ["a", "b"]}
        self.assertEqual(check_ext_meta_values(schemas, ext), [])

    def test_keyframes_object_array_valid(self):
        schemas = {"keyframes": _OBJECT_ARRAY}
        ext = {"keyframes": [{"scene_index": 0, "labels": ["장면"]}]}
        self.assertEqual(check_ext_meta_values(schemas, ext), [])

    def test_type_violations_sorted_by_key(self):
        schemas = {"keywords": _STRING_ARRAY, "summary": _STRING}
        ext = {"keywords": [1, 2], "summary": 42}
        violations = check_ext_meta_values(schemas, ext)
        self.assertEqual(violations, sorted(violations, key=lambda x: (x[0], x[1])))
        self.assertEqual({v[0] for v in violations}, {"keywords", "summary"})

    def test_array_element_type_violation(self):
        schemas = {"labels": _STRING_ARRAY}
        ext = {"labels": ["ok", 1]}
        violations = check_ext_meta_values(schemas, ext)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0][0], "labels")

    def test_key_not_in_ext_meta_skipped(self):
        schemas = {"stt": _STRING}
        self.assertEqual(check_ext_meta_values(schemas, {"summary": "x"}), [])

    def test_schema_missing_key_skipped(self):
        ext = {"summary": "x"}
        self.assertEqual(check_ext_meta_values({}, ext), [])

    def test_empty_or_none_ext_meta(self):
        schemas = {"summary": _STRING}
        self.assertEqual(check_ext_meta_values(schemas, {}), [])
        self.assertEqual(check_ext_meta_values(schemas, None), [])

    def test_non_validatable_schema_skipped(self):
        schemas = {"summary": {}, "keywords": {"items": {"type": "string"}}}
        ext = {"summary": 1, "keywords": [1]}
        self.assertEqual(check_ext_meta_values(schemas, ext), [])


class FetchExtKeySchemasTest(unittest.TestCase):
    def test_returns_validatable_schemas_only(self):
        rows = [
            {"meta_key": "summary", "json_schema": {"type": "string"}},
            {"meta_key": "bad", "json_schema": {}},
            {"meta_key": "labels", "json_schema": {"type": "array", "items": {"type": "string"}}},
        ]
        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = rows
        out = fetch_ext_key_schemas(conn, "general")
        self.assertEqual(
            out,
            {
                "summary": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "string"}},
            },
        )
        sql = cur.execute.call_args.args[0]
        self.assertIn("ext_meta_field_registry", sql)
        self.assertIn("json_schema", sql)
        self.assertEqual(cur.execute.call_args.args[1], ("general",))


class _FieldRegistryFakeCursor:
    def __init__(self, key_rows, schema_rows):
        self._key_rows = key_rows
        self._schema_rows = schema_rows
        self._last_sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._last_sql = sql

    def fetchall(self):
        if "json_schema" in self._last_sql:
            return self._schema_rows
        return self._key_rows


class _FieldRegistryFakeConn:
    def __init__(self, key_rows, schema_rows):
        self._cur = _FieldRegistryFakeCursor(key_rows, schema_rows)

    def cursor(self, *args, **kwargs):
        return self._cur


_GENERAL_KEYS = [
    {"meta_key": k}
    for k in ("summary", "keywords", "labels", "objects", "keyframes", "stt", "caption")
]
_GENERAL_SCHEMAS = [
    {"meta_key": "summary", "json_schema": {"type": "string"}},
    {"meta_key": "keywords", "json_schema": {"type": "array", "items": {"type": "string"}}},
]


class ValidateExtMetaTest(unittest.TestCase):
    def test_key_violation_message_unchanged(self):
        conn = _FieldRegistryFakeConn(_GENERAL_KEYS, _GENERAL_SCHEMAS)
        with self.assertRaises(ExtMetaValidationError) as cm:
            validate_ext_meta(conn, "general", {"unknown_key": "x"})
        self.assertIn("미등록 ext_meta 키", str(cm.exception))
        self.assertIn("unknown_key", str(cm.exception))

    def test_value_violation_raises(self):
        conn = _FieldRegistryFakeConn(_GENERAL_KEYS, _GENERAL_SCHEMAS)
        with self.assertRaises(ExtMetaValidationError) as cm:
            validate_ext_meta(conn, "general", {"keywords": [1, 2]})
        msg = str(cm.exception)
        self.assertIn("ext_meta 값 위반", msg)
        self.assertIn("keywords", msg)

    def test_key_violation_before_value(self):
        conn = _FieldRegistryFakeConn(_GENERAL_KEYS, _GENERAL_SCHEMAS)
        with self.assertRaises(ExtMetaValidationError) as cm:
            validate_ext_meta(conn, "general", {"bad_key": "x", "keywords": [1]})
        self.assertIn("미등록 ext_meta 키", str(cm.exception))

    def test_valid_ext_meta_passes(self):
        conn = _FieldRegistryFakeConn(_GENERAL_KEYS, _GENERAL_SCHEMAS)
        validate_ext_meta(conn, "general", {"summary": "요약", "keywords": ["a"]})

    def test_empty_allowed_keys_skips_all(self):
        conn = _FieldRegistryFakeConn([], [])
        validate_ext_meta(conn, "unknown_domain", {"anything": 1})


_RUN = os.getenv("RUN_DB_E2E") == "1"


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만(실 PostgreSQL·v291 head)")
class ExtMetaFieldRegistryDbE2eTest(unittest.TestCase):
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
                cur.execute("SELECT 1 FROM ext_meta_field_registry LIMIT 1")
                if cur.fetchone() is None:
                    raise unittest.SkipTest("ext_meta_field_registry 시드 없음(v291)")
        except unittest.SkipTest:
            raise
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"DB 미접속: {type(exc).__name__}: {exc}") from None

    @classmethod
    def tearDownClass(cls):
        if cls.db is not None:
            cls.db.__exit__(None, None, None)

    def test_labels_schema_has_object_items(self):
        # 298: labels 는 [{label, score}] **객체 배열**이 실제 형식(039 시드버그 string→object 교정).
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT json_schema FROM ext_meta_field_registry "
                "WHERE domain='general' AND meta_key='labels'"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0]["items"]["type"], "object")

    def test_validate_ext_meta_blocks_bad_keywords(self):
        with self.db.transaction() as conn:
            with self.assertRaises(ExtMetaValidationError) as cm:
                validate_ext_meta(conn, "general", {"keywords": [1, 2]})
            self.assertIn("ext_meta 값 위반", str(cm.exception))

    def test_validate_ext_meta_accepts_valid_general_meta(self):
        with self.db.transaction() as conn:
            validate_ext_meta(
                conn,
                "general",
                {"summary": "요약", "keywords": ["키워드"]},
            )

    def test_backfill_row_count_matches_general_plus_medical(self):
        with self.db.transaction() as conn:
            count = conn.execute("SELECT COUNT(*) FROM ext_meta_field_registry").fetchone()[0]
        self.assertGreaterEqual(count, 14)
