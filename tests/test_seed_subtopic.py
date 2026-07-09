"""scripts/seed_topic_registry.py subtopic 시드 순수 함수 단위 테스트 (spec 068 · G2·G3).

이슈3(소분류 과병합·과코스닝) 보정 — subtopic 층을 **외부표준(IPTC/KDC) 닫힌 시드**로 고정한다.
topic 층(058·parent NULL·닫힌 27+미분류)과 대칭으로, subtopic 은 부모 topic 스코프(parent NOT NULL)에
가산 적재한다. 여기서는 시드 파싱·행 추출·기계적 규칙(FR-204)·주입형 적재(register_fn)의 결정적
순수 함수만 검증한다(실 register_topic 임베딩·실 DB 적재는 G6·RUN_DB_E2E 몫).

LLM/DB 불필요 — 시드 파일 파싱·정본 행 추출·주입형 적재의 결정적 순수 함수만.
"""
from __future__ import annotations

import unittest

from scripts.seed_topic_registry import (
    _DEFAULT_SEED_PATH,
    _DEFAULT_SUBTOPIC_SEED_PATH,
    apply_subtopic_seed,
    load_subtopic_seed,
    load_taxonomy_seed,
    run_subtopic_seed,
    subtopic_registry_entries,
    taxonomy_registry_entries,
)

# 유효 부모 topic 집합 — taxonomy 27 정본(미분류 제외). subtopic 은 미분류에 달지 않는다.
_VALID_PARENTS = {
    e["topic_ko"]
    for e in taxonomy_registry_entries(load_taxonomy_seed(_DEFAULT_SEED_PATH))
    if e["topic_ko"] != "미분류"
}


class TestLoadSubtopicSeed(unittest.TestCase):
    """subtopic_seed.json 파싱 — 버전·출처 기록·subtopics 배열."""

    def test_parses_version_source_and_subtopics(self):
        seed = load_subtopic_seed(_DEFAULT_SUBTOPIC_SEED_PATH)
        self.assertIn("version", seed)
        self.assertTrue(str(seed["version"]).strip(), "버전 기록 누락")
        self.assertIn("source", seed)
        self.assertTrue(str(seed["source"]).strip(), "출처(IPTC/KDC) 기록 누락")
        self.assertIn("subtopics", seed)
        self.assertIsInstance(seed["subtopics"], list)


class TestSubtopicRegistryEntries(unittest.TestCase):
    """실 시드 → 적재 행 [{parent_topic, subtopic_ko, subtopic_en}] — 27 topic 커버·순서·형상."""

    def setUp(self):
        self.seed = load_subtopic_seed(_DEFAULT_SUBTOPIC_SEED_PATH)
        self.entries = subtopic_registry_entries(self.seed)

    def test_shape_and_str_enforced(self):
        # parent_topic·subtopic_ko 는 str() 강제, subtopic_en 은 None 허용(정본 미확정 여지).
        for e in self.entries:
            self.assertIsInstance(e["parent_topic"], str)
            self.assertIsInstance(e["subtopic_ko"], str)
            self.assertTrue(e["parent_topic"].strip())
            self.assertTrue(e["subtopic_ko"].strip())
            self.assertTrue(
                e["subtopic_en"] is None or isinstance(e["subtopic_en"], str)
            )

    def test_all_parents_in_27_topic_set(self):
        for e in self.entries:
            self.assertIn(
                e["parent_topic"], _VALID_PARENTS, f"부모 topic 이 27 밖: {e}"
            )

    def test_covers_all_27_topics(self):
        # subtopic 이 27 topic 전부를 덮어야 한다(미분류 제외).
        covered = {e["parent_topic"] for e in self.entries}
        self.assertEqual(covered, _VALID_PARENTS, "27 topic 중 미커버 존재")

    def test_row_count_about_170(self):
        # 확정 큐레이션: topic 당 5~7 → 약 170행(사람 확정본).
        self.assertGreaterEqual(len(self.entries), 150)
        self.assertLessEqual(len(self.entries), 190)

    def test_each_topic_has_5_to_8_subtopics(self):
        # 입도 균형(FINAL.md 규칙 ③): topic 당 5~7(큰 topic 은 더 잘게).
        from collections import Counter

        cnt = Counter(e["parent_topic"] for e in self.entries)
        for topic, n in cnt.items():
            self.assertGreaterEqual(n, 5, f"{topic} subtopic {n}개(<5)")
            self.assertLessEqual(n, 8, f"{topic} subtopic {n}개(>8)")

    def test_no_duplicate_within_parent(self):
        # 같은 부모 스코프 안에서 subtopic_ko 중복 없음(ON CONFLICT (parent,ko) 정합).
        seen: set[tuple[str, str]] = set()
        for e in self.entries:
            key = (e["parent_topic"], e["subtopic_ko"])
            self.assertNotIn(key, seen, f"부모 스코프 중복: {key}")
            seen.add(key)

    def test_preserves_file_order_first(self):
        # 파일 순서 보존: 첫 topic 은 음식·요리(taxonomy 순서).
        self.assertEqual(self.entries[0]["parent_topic"], "음식·요리")

    def test_no_subtopic_repeats_parent_name(self):
        # FR-204 결과: 확정 시드에는 부모 topic 명 반복 subtopic 이 없어야 한다(배제 0).
        for e in self.entries:
            self.assertNotIn(
                e["subtopic_ko"],
                e["parent_topic"],
                f"subtopic 이 부모명 부분문자열: {e}",
            )


class TestSubtopicRegistryEntriesRules(unittest.TestCase):
    """FR-204 기계적 규칙 — 부모 검증(27 밖 예외) + 부모명 반복(부분문자열) 배제(합성 시드)."""

    def test_parent_outside_27_raises(self):
        seed = {
            "version": "vtest",
            "subtopics": [
                {"topic_ko": "존재하지않는토픽", "subtopic_ko": "x", "subtopic_en": "y"},
            ],
        }
        with self.assertRaises(ValueError):
            subtopic_registry_entries(seed)

    def test_parent_unclassified_raises(self):
        # 미분류(catch-all)에는 subtopic 을 달지 않는다 → 27 밖으로 취급, 예외.
        seed = {
            "version": "vtest",
            "subtopics": [
                {"topic_ko": "미분류", "subtopic_ko": "무엇", "subtopic_en": "z"},
            ],
        }
        with self.assertRaises(ValueError):
            subtopic_registry_entries(seed)

    def test_subtopic_substring_of_parent_excluded(self):
        # 부모명 부분문자열(음식 ⊂ 음식·요리)·동일 라벨은 무의미 소분류 → 원천 배제.
        seed = {
            "version": "vtest",
            "subtopics": [
                {"topic_ko": "음식·요리", "subtopic_ko": "요리·레시피", "subtopic_en": "cooking_recipes"},
                {"topic_ko": "음식·요리", "subtopic_ko": "음식", "subtopic_en": "food"},
                {"topic_ko": "음식·요리", "subtopic_ko": "음식·요리", "subtopic_en": "same"},
            ],
        }
        entries = subtopic_registry_entries(seed)
        kos = [e["subtopic_ko"] for e in entries]
        self.assertEqual(kos, ["요리·레시피"], "부모명 부분문자열/동일 배제 실패")

    def test_str_coercion_on_non_str_fields(self):
        # 라벨 str() 강제(graph_query 관례) — 숫자 등이 들어와도 str 로.
        seed = {
            "version": "vtest",
            "subtopics": [
                {"topic_ko": "과학", "subtopic_ko": 12345, "subtopic_en": None},
            ],
        }
        entries = subtopic_registry_entries(seed)
        self.assertEqual(entries[0]["subtopic_ko"], "12345")
        self.assertIsNone(entries[0]["subtopic_en"])  # None 은 None 유지(str('None') 금지)


class TestApplySubtopicSeed(unittest.TestCase):
    """subtopic 적재 — register_fn(subtopic_ko, subtopic_en, source='taxonomy', parent_topic=<topic>) 호출."""

    def _seed(self):
        return {
            "version": "vtest",
            "subtopics": [
                {"topic_ko": "여행·지역", "subtopic_ko": "국내여행·지역탐방", "subtopic_en": "domestic_travel"},
                {"topic_ko": "여행·지역", "subtopic_ko": "해외여행", "subtopic_en": "international_travel"},
                {"topic_ko": "음식·요리", "subtopic_ko": "제과·제빵·디저트", "subtopic_en": "baking_dessert"},
            ],
        }

    def test_registers_each_with_parent_scope(self):
        calls: list[tuple] = []

        def fake_register(conn, ko, en, *, source, parent_topic):  # noqa: ANN001
            calls.append((ko, en, source, parent_topic))

        counts = apply_subtopic_seed(None, self._seed(), register_fn=fake_register)
        self.assertEqual(
            calls,
            [
                ("국내여행·지역탐방", "domestic_travel", "taxonomy", "여행·지역"),
                ("해외여행", "international_travel", "taxonomy", "여행·지역"),
                ("제과·제빵·디저트", "baking_dessert", "taxonomy", "음식·요리"),
            ],
        )
        self.assertEqual(counts, {"n_subtopic": 3})

    def test_full_seed_count(self):
        seed = load_subtopic_seed(_DEFAULT_SUBTOPIC_SEED_PATH)
        n = 0

        def fake_register(conn, ko, en, *, source, parent_topic):  # noqa: ANN001
            nonlocal n
            n += 1

        counts = apply_subtopic_seed(None, seed, register_fn=fake_register)
        self.assertEqual(n, len(subtopic_registry_entries(seed)))
        self.assertEqual(counts["n_subtopic"], n)


class _FakeCursor:
    def __init__(self, log: list[str]):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):  # noqa: ANN001
        self._log.append(sql)


class _FakeConn:
    def __init__(self):
        self.executed: list[str] = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _FakeCursor(self.executed)

    def commit(self):
        self.committed = True


class _FakeDB:
    """connection() 만 제공하는 최소 DB — dry-run 시 호출되면 안 됨을 검증."""

    def __init__(self):
        self.conn = _FakeConn()
        self.connection_calls = 0

    def connection(self):
        self.connection_calls += 1
        return self.conn


class TestRunSubtopicSeedIndependent(unittest.TestCase):
    """--subtopics 배선 — topic 시드와 독립·subtopic 층 삭제 없음·가산 적재·dry-run DB 미접촉."""

    def _seed(self):
        return {
            "version": "vtest",
            "subtopics": [
                {"topic_ko": "동물", "subtopic_ko": "반려동물", "subtopic_en": "pets"},
                {"topic_ko": "동물", "subtopic_ko": "야생동물·멸종위기종", "subtopic_en": "wildlife_endangered"},
            ],
        }

    def test_dry_run_does_not_touch_db(self):
        db = _FakeDB()
        counts = run_subtopic_seed(db, self._seed(), apply=False)
        self.assertEqual(counts, {"n_subtopic": 2})
        self.assertEqual(db.connection_calls, 0, "dry-run 인데 DB 접촉")

    def test_apply_registers_and_commits_without_delete(self):
        db = _FakeDB()
        calls: list[tuple] = []

        def fake_register(conn, ko, en, *, source, parent_topic):  # noqa: ANN001
            calls.append((ko, parent_topic))

        counts = run_subtopic_seed(
            db, self._seed(), apply=True, register_fn=fake_register
        )
        self.assertEqual(counts, {"n_subtopic": 2})
        self.assertTrue(db.conn.committed, "커밋 누락")
        # subtopic 층은 삭제하지 않는다(가산) — DELETE/TRUNCATE 가 실행되면 안 됨.
        joined = " ".join(db.conn.executed).upper()
        self.assertNotIn("DELETE", joined)
        self.assertNotIn("TRUNCATE", joined)
        self.assertEqual(calls, [("반려동물", "동물"), ("야생동물·멸종위기종", "동물")])


if __name__ == "__main__":
    unittest.main()
