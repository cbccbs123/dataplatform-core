"""081 SC-08 ② 전수 기계 불변식 검사의 순수부 — 검사 목록·읽기전용성·판정.

이 도구가 지키는 것: 전량 재실행 뒤 "게이트가 실제로 지켜졌나"를 **표본이 아니라 전 행**에서
기계적으로 확인한다. 내용(관계가 타당한가)은 골든·층화 표본이 보고, 여기서는 **규칙 위반 0**만 본다.
"""
from __future__ import annotations

import unittest

from scripts.verify_relation_invariants import CHECK_NAMES, build_checks, run_verify

_EXCLUDE = frozenset({"same_domain"})


class TestCheckList(unittest.TestCase):
    def test_선언된_검사를_모두_만든다(self):
        names = [c["name"] for c in build_checks(min_conf_similarity=0.75,
                                                 exclude_kinds=_EXCLUDE,
                                                 auto_approve_min=1.01)]
        self.assertEqual(names, list(CHECK_NAMES))

    def test_검사_이름이_중복되지_않는다(self):
        names = [c["name"] for c in build_checks(min_conf_similarity=0.75,
                                                 exclude_kinds=_EXCLUDE,
                                                 auto_approve_min=1.01)]
        self.assertEqual(len(names), len(set(names)))

    def test_모든_검사가_읽기_전용이다(self):
        # 검증 도구가 DB 를 바꾸면 "검증했더니 통과"가 자기충족이 된다.
        for c in build_checks(min_conf_similarity=0.75, exclude_kinds=_EXCLUDE,
                                 auto_approve_min=1.01):
            with self.subTest(c["name"]):
                head = c["sql"].strip().upper()
                self.assertTrue(head.startswith(("SELECT", "WITH")), head[:40])
                for w in ("INSERT", "UPDATE ", "DELETE", "DROP ", "ALTER ", "TRUNCATE"):
                    self.assertNotIn(w, c["sql"].upper(), f"{c['name']} 에 쓰기 키워드")

    def test_모든_검사가_카운트_하나를_돌려준다(self):
        # run_verify 가 행 모양을 가정하므로 계약을 고정한다.
        for c in build_checks(min_conf_similarity=0.75, exclude_kinds=_EXCLUDE,
                                 auto_approve_min=1.01):
            with self.subTest(c["name"]):
                self.assertIn("AS n", c["sql"])

    def test_노드_조인에_asset_가드가_있다(self):
        # entity 노드는 asset_id 가 NULL 이라 가드 없이 조인하면 None 이 섞인다(레포 관례).
        for c in build_checks(min_conf_similarity=0.75, exclude_kinds=_EXCLUDE,
                                 auto_approve_min=1.01):
            if "JOIN node" in c["sql"] or "node n" in c["sql"]:
                with self.subTest(c["name"]):
                    self.assertIn("node_kind", c["sql"])

    def test_임계와_제외목록이_SQL_에_반영된다(self):
        checks = {c["name"]: c for c in build_checks(min_conf_similarity=0.5,
                                                     exclude_kinds=frozenset({"references"}))}
        self.assertIn(0.5, checks["유사도_계열_저신뢰_잔존"]["params"])
        self.assertIn(["references"], checks["자동승인_제외_kind가_active"]["params"])

    def test_제외목록이_비면_그_검사는_건너뛴다(self):
        # 게이트를 끈 설정에서 "위반"을 보고하면 거짓 경보다.
        names = [c["name"] for c in build_checks(min_conf_similarity=0.0,
                                                 exclude_kinds=frozenset())]
        self.assertNotIn("자동승인_제외_kind가_active", names)
        self.assertNotIn("유사도_계열_저신뢰_잔존", names)

    def test_제외목록은_정렬돼_바인딩된다(self):
        checks = {c["name"]: c for c in build_checks(
            min_conf_similarity=0.75, exclude_kinds=frozenset({"same_domain", "duplicate_near"}))}
        self.assertIn(["duplicate_near", "same_domain"],
                      checks["자동승인_제외_kind가_active"]["params"])


class _FakeDb:
    """`run_verify` 가 쓰는 최소 인터페이스만 흉내(PostgresUtil.transaction → conn.cursor)."""

    def __init__(self, counts: dict[str, int]):
        self._counts = counts
        self.executed: list[str] = []

    def transaction(self):
        outer = self

        class _Cur:
            def execute(self, sql, params=None):
                outer.executed.append(sql)
                self._sql = sql

            def fetchone(self):
                # 검사 이름을 SQL 주석으로 심어 두고 그걸로 조작값을 고른다.
                for name, n in outer._counts.items():
                    if f"-- check:{name}" in self._sql:
                        return {"n": n}
                return {"n": 0}

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        class _Conn:
            def cursor(self, **_k):
                return _Cur()

        class _Ctx:
            def __enter__(self):
                return _Conn()

            def __exit__(self, *_a):
                return False

        return _Ctx()


class TestRunVerify(unittest.TestCase):
    _CHECKS_KW = {"min_conf_similarity": 0.75, "exclude_kinds": _EXCLUDE}

    def test_전부_0이면_통과다(self):
        rep = run_verify(_FakeDb({}), checks=build_checks(**self._CHECKS_KW))
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["violations"], 0)

    def test_위반이_있으면_실패다(self):
        rep = run_verify(_FakeDb({"자기참조_엣지": 3}), checks=build_checks(**self._CHECKS_KW))
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["violations"], 3)

    def test_위반_항목만_따로_보고한다(self):
        rep = run_verify(_FakeDb({"자기참조_엣지": 3, "대칭_중복행": 1}),
                         checks=build_checks(**self._CHECKS_KW))
        self.assertEqual({v["name"] for v in rep["failed"]}, {"자기참조_엣지", "대칭_중복행"})
        self.assertEqual(rep["violations"], 4)

    def test_모든_검사를_실행한다(self):
        # 하나라도 건너뛰면 "통과"가 거짓이 된다.
        db = _FakeDb({})
        checks = build_checks(**self._CHECKS_KW)
        run_verify(db, checks=checks)
        self.assertEqual(len(db.executed), len(checks))

    def test_결과가_검사_순서를_지킨다(self):
        checks = build_checks(**self._CHECKS_KW)
        rep = run_verify(_FakeDb({}), checks=checks)
        self.assertEqual([r["name"] for r in rep["results"]], [c["name"] for c in checks])


if __name__ == "__main__":
    unittest.main()
