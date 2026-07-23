"""E4 — PostgresUtil._build_conninfo 의 libpq conninfo 이스케이프 회귀 방지.

특수문자(공백·작은따옴표·백슬래시)가 든 비밀번호가 손조립 f-string 에서 깨지던 것을
psycopg `make_conninfo` 위임으로 고쳤다. `conninfo_to_dict` 로 왕복 파싱해 값이 보존되는지
(=이스케이프가 정확한지) 단언한다. 연결은 하지 않는다(풀은 지연 생성).
"""

from __future__ import annotations

import unittest

from psycopg.conninfo import conninfo_to_dict

from src.database.postgres_util import PostgresConfig, PostgresUtil


def _util(**cfg) -> PostgresUtil:
    return PostgresUtil(config=PostgresConfig(**cfg))


class TestBuildConninfoEscaping(unittest.TestCase):
    def test_special_char_password_roundtrips(self) -> None:
        # 작은따옴표 + 공백 + 백슬래시 — 손조립 f-string 이 깨지는 대표 케이스.
        pw = "p'a ss\\x"
        info = _util(host="h", port=5432, dbname="d", user="u", password=pw)._build_conninfo()
        parsed = conninfo_to_dict(info)
        self.assertEqual(parsed["password"], pw)  # 이스케이프/quoting 정확 → 값 보존
        self.assertEqual(parsed["host"], "h")
        self.assertEqual(parsed["user"], "u")
        self.assertEqual(parsed["dbname"], "d")

    def test_space_in_password_does_not_leak_into_next_key(self) -> None:
        # 공백이 있으면 손조립 f-string 은 'ss' 를 새 키워드로 오인해 파싱이 깨진다.
        info = _util(password="a b")._build_conninfo()
        self.assertEqual(conninfo_to_dict(info)["password"], "a b")

    def test_plain_password_still_ok(self) -> None:
        info = _util(password="simple")._build_conninfo()
        self.assertEqual(conninfo_to_dict(info)["password"], "simple")

    def test_statement_timeout_option_preserved(self) -> None:
        info = _util(statement_timeout_ms=5000)._build_conninfo()
        self.assertIn("statement_timeout=5000", conninfo_to_dict(info)["options"])

    def test_dsn_passthrough_unchanged(self) -> None:
        u = PostgresUtil(dsn="postgresql://x/y")
        self.assertEqual(u._build_conninfo(), "postgresql://x/y")


if __name__ == "__main__":
    unittest.main()
