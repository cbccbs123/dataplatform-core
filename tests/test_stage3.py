"""stage3_gemma.classify 동적 라벨 + complete 주입 테스트(네트워크 불필요)."""
from __future__ import annotations

import unittest

from src.classify.stage3_gemma import classify
from src.classify.types import DOMAIN_REVIEW


class TestStage3(unittest.TestCase):
    def test_label_in_allowed_returned(self) -> None:
        label, sig = classify("환자 진단", ["medical", "general"], complete=lambda p: '{"label": "medical"}')
        self.assertEqual(label, "medical")
        self.assertEqual(sig["stage3"], "llm")

    def test_dynamic_label_legal(self) -> None:
        label, _ = classify("계약 소송", ["medical", "legal", "general"], complete=lambda p: '{"label": "legal"}')
        self.assertEqual(label, "legal")

    def test_label_not_allowed_returns_review(self) -> None:
        label, sig = classify("x", ["medical", "general"], complete=lambda p: '{"label": "blah"}')
        self.assertEqual(label, DOMAIN_REVIEW)
        self.assertEqual(sig["stage3"], "llm_unclear")

    def test_complete_raises_returns_review(self) -> None:
        def boom(_p):
            raise RuntimeError("no endpoint")
        label, sig = classify("x", ["medical", "general"], complete=boom)
        self.assertEqual(label, DOMAIN_REVIEW)
        self.assertEqual(sig["stage3"], "error")

    def test_prompt_lists_allowed_labels(self) -> None:
        seen = {}
        def capture(prompt):
            seen["p"] = prompt
            return '{"label": "general"}'
        classify("본문", ["medical", "legal", "general"], complete=capture)
        self.assertIn("medical", seen["p"])
        self.assertIn("legal", seen["p"])


if __name__ == "__main__":
    unittest.main()
