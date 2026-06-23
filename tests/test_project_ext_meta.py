"""ext_meta access_tier projection 단위 테스트 (spec 042)."""
from __future__ import annotations

import unittest

from src.registry.access_tier import (
    AUTHENTICATED,
    AUTHORIZED,
    PUBLIC,
    principal_clearance,
    project_ext_meta,
    tier_allows,
)


class PrincipalClearanceTest(unittest.TestCase):
    def test_anonymous_is_public(self):
        self.assertEqual(principal_clearance(authenticated=False), PUBLIC)

    def test_jwt_is_authorized(self):
        self.assertEqual(principal_clearance(authenticated=True), AUTHORIZED)


class TierAllowsTest(unittest.TestCase):
    def test_authenticated_sees_authenticated_not_authorized(self):
        self.assertTrue(tier_allows(AUTHENTICATED, PUBLIC))
        self.assertTrue(tier_allows(AUTHENTICATED, AUTHENTICATED))
        self.assertFalse(tier_allows(AUTHENTICATED, "authorized"))

    def test_authorized_sees_authorized_tier(self):
        self.assertTrue(tier_allows(AUTHORIZED, "authorized"))


class ProjectExtMetaTest(unittest.TestCase):
    _TIERS = {"summary": "authenticated", "stt": "authorized", "keywords": "authenticated"}

    def test_public_hides_authenticated_and_authorized_keys(self):
        ext = {"summary": "요약", "stt": "전문", "keywords": ["a"]}
        out = project_ext_meta(ext, self._TIERS, domain="general", clearance=PUBLIC)
        self.assertEqual(out, {})

    def test_jwt_clearance_keeps_all_general_keys_including_stt(self):
        ext = {"summary": "요약", "stt": "전문", "keywords": ["a"]}
        out = project_ext_meta(
            ext,
            self._TIERS,
            domain="general",
            clearance=principal_clearance(authenticated=True),
        )
        self.assertEqual(out, ext)

    def test_jwt_clearance_still_blocked_on_medical_floor(self):
        ext = {"summary": "요약"}
        tiers = {"summary": "authenticated"}
        out = project_ext_meta(
            ext,
            tiers,
            domain="medical",
            clearance=principal_clearance(authenticated=True),
        )
        self.assertEqual(out, {})

    def test_unlisted_key_passes_through(self):
        ext = {"custom": 1}
        out = project_ext_meta(ext, {}, domain="general", clearance=PUBLIC)
        self.assertEqual(out, {"custom": 1})
