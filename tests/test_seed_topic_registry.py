"""scripts/seed_topic_registry.py 순수 함수 단위 테스트 (spec 058 v2 T902 · taxonomy 시드).

v2 개정(2026-07-07·닫힌 분류체계 전환): v1 replay/from-draft 로직은 폐기됐다(ADR
`2026-07-07-topic-closed-taxonomy-pivot.md`). 시드는 이제 **taxonomy_seed.json**(27+기타 닫힌
분류체계 정본)을 그대로 topic_registry(parent_topic=NULL·source='taxonomy')에 적재한다.

LLM/DB 불필요 — 시드 파일 파싱·정본 행 추출·주입형 적재(register_fn)의 결정적 순수 함수만 검증한다.
(실제 TRUNCATE·실 DB 적재는 임퓨어 경로이며 여기서 다루지 않는다.)
"""
from __future__ import annotations

import unittest

from scripts.seed_topic_registry import (
    _DEFAULT_SEED_PATH,
    apply_taxonomy_seed,
    load_taxonomy_seed,
    taxonomy_registry_entries,
)

# taxonomy_draft.md §1 확정 분류체계(27+기타) — 시드 파일의 정합성 교차검증용.
_EXPECTED_TOPIC_KOS = {
    "음식·요리", "스포츠·레저", "예술·공예", "음악", "미디어·엔터테인먼트",
    "자연·환경", "동물", "과학", "IT·기술", "교통·모빌리티", "여행·지역",
    "역사·문화유산", "생활·취미", "경제·산업", "정치·사회", "법·범죄",
    "군사·안보", "재난·안전", "교육·지식", "건강·의학", "사람·일상",
    "종교·신앙", "언어·어학", "문학·도서", "패션·뷰티", "가족·육아",
    "직업·커리어", "기타",
}


class TestLoadTaxonomySeed(unittest.TestCase):
    """taxonomy_seed.json 파싱 — 버전 기록·topics 배열."""

    def test_parses_version_and_topics(self):
        seed = load_taxonomy_seed(_DEFAULT_SEED_PATH)
        self.assertIn("version", seed)
        self.assertTrue(str(seed["version"]).strip(), "버전 기록 누락")
        self.assertIn("topics", seed)
        self.assertIsInstance(seed["topics"], list)


class TestTaxonomyRegistryEntries(unittest.TestCase):
    """시드 → registry 적재 행 [{topic_ko, topic_en}] — 28행(기타 포함)·순서 보존·중복 0."""

    def setUp(self):
        self.seed = load_taxonomy_seed(_DEFAULT_SEED_PATH)
        self.entries = taxonomy_registry_entries(self.seed)

    def test_has_28_rows_including_etc(self):
        self.assertEqual(len(self.entries), 28)
        kos = {e["topic_ko"] for e in self.entries}
        self.assertIn("기타", kos)  # 탈출구(강제 배정 금지)

    def test_matches_taxonomy_draft_topic_kos(self):
        kos = {e["topic_ko"] for e in self.entries}
        self.assertEqual(kos, _EXPECTED_TOPIC_KOS)

    def test_no_duplicate_topic_ko(self):
        kos = [e["topic_ko"] for e in self.entries]
        self.assertEqual(len(kos), len(set(kos)), "topic_ko 중복 존재")

    def test_all_have_nonempty_topic_en(self):
        for e in self.entries:
            self.assertTrue(str(e["topic_en"]).strip(), f"{e['topic_ko']} topic_en 비어있음")

    def test_preserves_draft_order_first_and_last(self):
        # 표 순서 보존: 첫 행 음식·요리 → 마지막 행 기타(탈출구).
        self.assertEqual(self.entries[0]["topic_ko"], "음식·요리")
        self.assertEqual(self.entries[-1]["topic_ko"], "기타")


class TestApplyTaxonomySeed(unittest.TestCase):
    """taxonomy 적재 — register_topic(parent_topic=None·source='taxonomy') 만 호출(alias 0·LLM/kNN 0)."""

    def _draft(self):
        return {
            "version": "v2",
            "topics": [
                {"topic_ko": "음식·요리", "topic_en": "food_cooking"},
                {"topic_ko": "기타", "topic_en": "etc"},
            ],
        }

    def test_registers_parent_null_source_taxonomy_no_alias(self):
        reg_calls: list[tuple] = []

        def fake_register(conn, ko, en, *, source, parent_topic):  # noqa: ANN001
            reg_calls.append((ko, en, source, parent_topic))

        counts = apply_taxonomy_seed(None, self._draft(), register_fn=fake_register)
        # 각 topic 은 parent_topic=None(topic 층)·source='taxonomy' 로 등록
        self.assertEqual(
            reg_calls,
            [
                ("음식·요리", "food_cooking", "taxonomy", None),
                ("기타", "etc", "taxonomy", None),
            ],
        )
        # alias 는 쓰지 않는다(닫힌 분류체계는 쌍별 병합 없음 — 결과 alias 0)
        self.assertEqual(counts, {"n_registry": 2, "n_alias": 0})

    def test_full_seed_registers_28(self):
        seed = load_taxonomy_seed(_DEFAULT_SEED_PATH)
        n = 0

        def fake_register(conn, ko, en, *, source, parent_topic):  # noqa: ANN001
            nonlocal n
            n += 1

        counts = apply_taxonomy_seed(None, seed, register_fn=fake_register)
        self.assertEqual(n, 28)
        self.assertEqual(counts["n_registry"], 28)
        self.assertEqual(counts["n_alias"], 0)


if __name__ == "__main__":
    unittest.main()
