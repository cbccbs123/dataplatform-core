"""src.relations 패키지 지연 export(__getattr__) 단위 테스트. DB·LLM 불필요.

__all__ 과 __getattr__ 분기가 어긋나면(삭제된 모듈 잔재 등) import 시점이 아니라
속성 접근 시점에야 터지므로, 전 항목을 실제로 resolve 해 잔재를 조기에 잡는다.
"""

from __future__ import annotations

import unittest

import src.relations as relations


class TestRelationsLazyExports(unittest.TestCase):
    def test_all_lazy_exports_resolve(self) -> None:
        # __all__ 의 모든 이름이 지연 import 로 실제 callable 에 닿는다
        for name in relations.__all__:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(relations, name)))

    def test_unknown_attribute_raises(self) -> None:
        with self.assertRaises(AttributeError):
            relations.does_not_exist  # noqa: B018 — 예외 발생 자체가 검증 대상


if __name__ == "__main__":
    unittest.main()
