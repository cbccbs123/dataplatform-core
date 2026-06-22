"""ext_meta access_tier projection 단위 테스트 (spec 042)."""
from __future__ import annotations

import unittest

from src.registry.access_tier import (
    AUTHENTICATED,
    PUBLIC,
    project_ext_meta,
    principal_clearance,
    tier_allows,
)


class PrincipalClearanceTest(unittest.TestCase):
    def test_anonymous_is_public(self):
        self.assertEqual(principal_clearance(authenticated=False), PUBLIC)

    def test_jwt_is_authenticated(self):
        self.assertEqual(principal_clearance(authenticated=True), AUTHENTICATED)


class TierAllowsTest(unittest.TestCase):
    def test_authenticated_sees_authenticated_not_authorized(self):
        self.assertTrue(tier_allows(AUTHENTICATED, PUBLIC))
        self.assertTrue(tier_allows(AUTHENTICATED, AUTHENTICATED))
        self.assertFalse(tier_allows(AUTHENTICATED, "authorized"))


class ProjectExtMetaTest(unittest.TestCase):
    _TIERS = {"summary": "authenticated", "stt": "authorized", "keywords": "authenticated"}

    def test_public_hides_authenticated_and_authorized_keys(self):
        ext = {"summary": "요약", "stt": "전문", "keywords": ["a"]}
        out = project_ext_meta(ext, self._TIERS, domain="general", clearance=PUBLIC)
        self.assertEqual(out, {})

    def test_authenticated_keeps_summary_hides_stt(self):
        ext = {"summary": "요약", "stt": "전문", "keywords": ["a"]}
        out = project_ext_meta(ext, self._TIERS, domain="general", clearance=AUTHENTICATED)
        self.assertEqual(out, {"summary": "요약", "keywords": ["a"]})
        self.assertNotIn("stt", out)

    def test_medical_floor_blocks_authenticated_on_general_tier_field(self):
        ext = {"summary": "요약"}
        tiers = {"summary": "authenticated"}
        out = project_ext_meta(ext, tiers, domain="medical", clearance=AUTHENTICATED)
        self.assertEqual(out, {})

    def test_unlisted_key_passes_through(self):
        ext = {"custom": 1}
        out = project_ext_meta(ext, {}, domain="general", clearance=PUBLIC)
        self.assertEqual(out, {"custom": 1})
