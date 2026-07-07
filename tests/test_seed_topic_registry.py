"""scripts/seed_topic_registry.py 순수 함수 단위 테스트 (spec 058 v2 T902 · taxonomy 시드).

v2 개정(2026-07-07·닫힌 분류체계 전환): v1 replay/from-draft 로직은 폐기됐다(ADR
`2026-07-07-topic-closed-taxonomy-pivot.md`). 시드는 이제 **taxonomy_seed.json**(27+미분류 닫힌
분류체계 정본)을 그대로 topic_registry(parent_topic=NULL·source='taxonomy')에 적재한다.

alias 선시드(2026-07-07·G12 driver 확정): registry 시드 직후 **taxonomy_alias_seed.json**(§3
커버리지 매핑·raw_ko→canonical)을 topic 층 alias(parent NULL·decided_by='seed')로 적재해 백필
topic 분류를 LLM 재분류 없이 결정적으로 만든다.

LLM/DB 불필요 — 시드 파일 파싱·정본 행 추출·주입형 적재(register_fn/freeze_fn)의 결정적 순수 함수만 검증한다.
(실제 TRUNCATE·실 DB 적재는 임퓨어 경로이며 여기서 다루지 않는다.)
"""
from __future__ import annotations

import unittest

from scripts.seed_topic_registry import (
    _DEFAULT_ALIAS_SEED_PATH,
    _DEFAULT_SEED_PATH,
    _delete_topic_layer,
    alias_seed_entries,
    apply_alias_seed,
    apply_taxonomy_seed,
    load_alias_seed,
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
    "직업·커리어", "미분류",
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
        self.assertIn("미분류", kos)  # 탈출구(강제 배정 금지·en=unclassified)

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
        # 표 순서 보존: 첫 행 음식·요리 → 마지막 행 미분류(탈출구).
        self.assertEqual(self.entries[0]["topic_ko"], "음식·요리")
        self.assertEqual(self.entries[-1]["topic_ko"], "미분류")


class TestApplyTaxonomySeed(unittest.TestCase):
    """taxonomy 적재 — register_topic(parent_topic=None·source='taxonomy') 만 호출(alias 0·LLM/kNN 0)."""

    def _draft(self):
        return {
            "version": "v2",
            "topics": [
                {"topic_ko": "음식·요리", "topic_en": "food_cooking"},
                {"topic_ko": "미분류", "topic_en": "unclassified"},
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
                ("미분류", "unclassified", "taxonomy", None),
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


class TestLoadAliasSeed(unittest.TestCase):
    """taxonomy_alias_seed.json 파싱 — 버전 기록·aliases 배열(§3 커버리지 매핑)."""

    def test_parses_version_and_aliases(self):
        seed = load_alias_seed(_DEFAULT_ALIAS_SEED_PATH)
        self.assertIn("version", seed)
        self.assertTrue(str(seed["version"]).strip(), "버전 기록 누락")
        self.assertIn("aliases", seed)
        self.assertIsInstance(seed["aliases"], list)
        self.assertGreater(len(seed["aliases"]), 100)  # §3 raw ~120 → 자기참조 제외 ~117


class TestAliasSeedEntries(unittest.TestCase):
    """alias 선시드 행 [{raw_ko, canonical_ko}] — 라벨 str()·raw_ko 유일·canonical 은 정본·자기참조 0."""

    def setUp(self):
        self.alias_seed = load_alias_seed(_DEFAULT_ALIAS_SEED_PATH)
        self.entries = alias_seed_entries(self.alias_seed)
        self.canon_kos = {e["topic_ko"] for e in taxonomy_registry_entries(load_taxonomy_seed(_DEFAULT_SEED_PATH))}

    def test_shape_and_str(self):
        for e in self.entries:
            self.assertIsInstance(e["raw_ko"], str)
            self.assertIsInstance(e["canonical_ko"], str)
            self.assertTrue(e["raw_ko"].strip() and e["canonical_ko"].strip())

    def test_raw_ko_unique(self):
        raws = [e["raw_ko"] for e in self.entries]
        self.assertEqual(len(raws), len(set(raws)), "raw_ko 중복 존재")

    def test_all_canonicals_are_taxonomy_topics(self):
        # 모든 canonical_ko 는 닫힌 정본(28)에 속해야 한다(alias 히트 → registry en 조회 성립).
        for e in self.entries:
            self.assertIn(e["canonical_ko"], self.canon_kos, f"정본 아님: {e}")

    def test_no_self_referential(self):
        # raw_ko == canonical_ko 는 닫힌 집합 정확일치가 처리하므로 alias 로 두지 않는다(음악/과학/동물).
        for e in self.entries:
            self.assertNotEqual(e["raw_ko"], e["canonical_ko"])

    def test_energy_maps_to_economy_industry(self):
        # §3 검토 확정(F3): 에너지 → 경제·산업(과거 LLM 이 과학으로 오분류하던 케이스를 결정적 고정).
        m = {e["raw_ko"]: e["canonical_ko"] for e in self.entries}
        self.assertEqual(m.get("에너지"), "경제·산업")
        self.assertEqual(m.get("금융"), "경제·산업")


class TestApplyAliasSeed(unittest.TestCase):
    """alias 적재 — freeze_fn(raw, canonical, 'seed', parent_topic=None) 만 호출(topic 층·멱등)."""

    def test_freezes_parent_null_decided_by_seed(self):
        seed = {
            "version": "v2",
            "aliases": [
                {"raw_ko": "에너지", "canonical_ko": "경제·산업"},
                {"raw_ko": "천문", "canonical_ko": "과학"},
            ],
        }
        calls: list[tuple] = []

        def fake_freeze(conn, raw_ko, canonical_ko, decided_by, *, parent_topic):  # noqa: ANN001
            calls.append((raw_ko, canonical_ko, decided_by, parent_topic))

        counts = apply_alias_seed(None, seed, freeze_fn=fake_freeze)
        self.assertEqual(
            calls,
            [
                ("에너지", "경제·산업", "seed", None),
                ("천문", "과학", "seed", None),
            ],
        )
        self.assertEqual(counts, {"n_alias": 2})

    def test_full_alias_seed_count(self):
        seed = load_alias_seed(_DEFAULT_ALIAS_SEED_PATH)
        n = 0

        def fake_freeze(conn, raw_ko, canonical_ko, decided_by, *, parent_topic):  # noqa: ANN001
            nonlocal n
            n += 1

        counts = apply_alias_seed(None, seed, freeze_fn=fake_freeze)
        self.assertEqual(counts["n_alias"], len(seed["aliases"]))
        self.assertEqual(n, len(seed["aliases"]))


class _FakeCursor:
    """execute 된 SQL 을 로그에 기록하는 최소 커서(컨텍스트 매니저)."""

    def __init__(self, log: list[str]):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):  # noqa: ANN001
        self._log.append(sql)


class _FakeConn:
    """cursor() 만 제공하는 최소 커넥션(순수 단위 — 실 DB 없이 SQL 관측)."""

    def __init__(self):
        self.executed: list[str] = []

    def cursor(self):
        return _FakeCursor(self.executed)


class TestDeleteTopicLayerScoped(unittest.TestCase):
    """🔴 시드 정리는 **topic 층(parent_topic IS NULL)만** 스코프 삭제 — subtopic 층 보존.

    과거 ``TRUNCATE topic_alias, topic_registry`` 는 v297 로 자란 subtopic 층(백필 성장
    레이어·결정성 캐시·parent NOT NULL)까지 통째로 날려 governance §4 '전역 재빌드 없음'과
    상충하고, 재실행 시 프로덕션 subtopic 을 소실시켰다. → 삭제를 parent_topic IS NULL 로 스코프.
    """

    def test_scoped_delete_parent_null_only_no_truncate(self):
        conn = _FakeConn()
        _delete_topic_layer(conn)
        joined = " ".join(conn.executed)
        # topic 층만: topic_alias·topic_registry 각각 parent_topic IS NULL 로 DELETE(2문).
        self.assertIn("topic_alias", joined)
        self.assertIn("topic_registry", joined)
        self.assertEqual(joined.upper().count("DELETE"), 2, f"DELETE 2문 아님: {conn.executed}")
        # 두 삭제 모두 subtopic 층(parent NOT NULL)을 건드리지 않게 parent_topic IS NULL 로 스코프.
        for sql in conn.executed:
            self.assertIn("parent_topic IS NULL", sql, f"스코프 없음(subtopic 위험): {sql}")
        # TRUNCATE 는 subtopic 층까지 날리므로 절대 쓰지 않는다(회귀 방지).
        self.assertNotIn("TRUNCATE", joined.upper())

    def test_deletes_alias_before_registry(self):
        # 읽기 순서(자식 alias → 부모 registry) 유지 — alias 를 먼저 지운다.
        conn = _FakeConn()
        _delete_topic_layer(conn)
        first_alias = next(i for i, s in enumerate(conn.executed) if "topic_alias" in s)
        first_registry = next(i for i, s in enumerate(conn.executed) if "topic_registry" in s)
        self.assertLess(first_alias, first_registry)


if __name__ == "__main__":
    unittest.main()
