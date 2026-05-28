import unittest
from unittest.mock import MagicMock, patch


class TestRegisterNewKinds(unittest.TestCase):
    def test_only_unknown_codes_registered_inactive(self):
        from src.relations.persist import register_new_relation_kinds
        edges = [
            {"relation_type_code": "duplicate_near", "reason": "r1"},   # active → skip
            {"relation_type_code": "gaming_hardware", "reason": "r2"},  # 신규 → 등록
        ]
        conn = MagicMock()
        with patch("src.relations.persist.ensure_relation_kind_for_llm_proposal") as ens:
            registered, skipped = register_new_relation_kinds(
                conn, edges=edges, active_kind_codes=frozenset({"duplicate_near"}))
        self.assertEqual((registered, skipped), (1, 1))
        ens.assert_called_once()
        self.assertEqual(ens.call_args.kwargs["status"], "inactive")
