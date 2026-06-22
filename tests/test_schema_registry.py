"""F-4.13 ext_meta JSON Schema 값 검증 단위 테스트 (spec 039)."""
from __future__ import annotations

import unittest

from src.registry.schema_registry import check_ext_meta_values

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
