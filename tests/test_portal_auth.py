"""포탈 JWT 단위 테스트 (spec 042)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.portal.auth import authenticate_token, get_principal, issue_dev_token
from src.portal.auth.verifier import _reset_verifier_for_tests
from src.registry.access_tier import AUTHORIZED, PUBLIC


class PortalAuthTest(unittest.TestCase):
    def setUp(self):
        _reset_verifier_for_tests()
        self._env = mock.patch.dict(
            os.environ,
            {"PORTAL_JWT_SECRET": "test-secret", "PORTAL_AUTH_DISABLED": "0"},
            clear=False,
        )
        self._env.start()

    def tearDown(self):
        self._env.stop()
        _reset_verifier_for_tests()

    def test_issue_and_decode_roundtrip(self):
        token = issue_dev_token(user_id="alice")
        p = authenticate_token(token)
        self.assertEqual(p.user_id, "alice")
        self.assertEqual(p.clearance, AUTHORIZED)

    def test_get_principal_requires_token_when_auth_enabled(self):
        with self.assertRaises(HTTPException) as cm:
            get_principal(credentials=None)
        self.assertEqual(cm.exception.status_code, 401)

    def test_get_principal_auth_disabled_anonymous(self):
        with mock.patch.dict(os.environ, {"PORTAL_AUTH_DISABLED": "1"}):
            _reset_verifier_for_tests()
            p = get_principal(credentials=None)
        self.assertEqual(p.user_id, "anonymous")
        self.assertEqual(p.clearance, PUBLIC)

    def test_get_principal_auth_disabled_with_bearer(self):
        token = issue_dev_token(user_id="bob")
        with mock.patch.dict(os.environ, {"PORTAL_AUTH_DISABLED": "1"}):
            _reset_verifier_for_tests()
            p = get_principal(
                credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            )
        self.assertEqual(p.user_id, "bob")
        self.assertEqual(p.clearance, AUTHORIZED)
