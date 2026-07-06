"""058 G2 — ``topic_canonicalize`` seam 단위 테스트 (mock, DB·LLM 불필요).

검증 의도 (FR-201~204·FR-701~703·SC-02·SC-05)
    관계 topic 자유기입 라벨을 성장하는 정본 레지스트리로 수렴시키는 해소 파이프라인:
      정확일치(alias 캐시·LLM 0) → 임베딩 kNN 후보 → LLM 재사용/신규 판정 → alias 동결.
    - **결정성(헌법 3조)**: kNN 정렬 타이브레이커(거리 asc → topic_ko asc), 캐시 히트 재실행 LLM 0.
    - **LLM 단일 seam**: ``judge_topic`` 은 후보 K개만 프롬프트에 주입하고 ``client=`` 주입으로
      네트워크 없이 검증한다(``src.llm.client.complete_json`` 경유·temp=0).
    - **임베딩 불변식(034 교훈)**: ``register_topic`` 은 0-노름(빈 콘텐츠) 임베딩을 거부(ValueError) —
      NULL/0-노름 topic 은 kNN 불가시 → 동의어 재난립.

mock 패턴은 ``tests/test_topic_query.py``(cursor mock·_mock_conn) 동형. canonicalize_topic 은
헬퍼(lookup_alias·knn_topic_candidates·judge_topic·register_topic·_freeze_alias·_lookup_topic_en)를
모듈 위치에서 patch 해 분기만 순수 검증한다.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch

_MOD = "src.relations.topic_canonicalize"


def _mock_conn(rows):
    """``conn.cursor(row_factory=dict_row)`` 컨텍스트매니저를 흉내내는 mock conn(topic_query 동형).

    ``__enter__`` 가 cur 를 돌려주고 ``fetchall``/``fetchone`` 이 주입 행을 반환한다.
    ``execute`` 인자는 ``cur.execute.call_args`` 로 캡처해 SQL 부분문자열·바인딩을 검증한다.
    """
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.fetchall.return_value = rows
    cur.fetchone.return_value = rows[0] if rows else None
    conn.cursor.return_value = cur
    return conn, cur


def _compact_sql(cur):
    """마지막 execute 의 SQL 을 공백 정규화해 견고한 부분문자열 검사용으로 반환."""
    return " ".join(cur.execute.call_args[0][0].split())


class TestLookupAlias(unittest.TestCase):
    """alias 정확일치 캐시 룩업 — 히트/미스·바인딩."""

    def test_exact_hit_returns_canonical_as_str(self) -> None:
        from src.relations.topic_canonicalize import lookup_alias

        conn, cur = _mock_conn([{"canonical_ko": "요리"}])
        out = lookup_alias(conn, "식품")
        self.assertEqual(out, "요리")
        self.assertIsInstance(out, str)
        # raw_ko 는 %s 바인딩(인젝션 0)
        self.assertIn("topic_alias", _compact_sql(cur))
        self.assertEqual(cur.execute.call_args[0][1], ("식품",))

    def test_miss_returns_none(self) -> None:
        from src.relations.topic_canonicalize import lookup_alias

        conn, _ = _mock_conn([])
        self.assertIsNone(lookup_alias(conn, "없는라벨"))


class TestKnnTopicCandidates(unittest.TestCase):
    """임베딩 kNN 후보 회수 — 결정적 정렬·0-노름 제외·빈 레지스트리·str·k."""

    @patch(f"{_MOD}._embed_label", return_value=[0.1] * 1536)
    def test_deterministic_sort_and_zero_norm_excluded(self, m_embed) -> None:
        from src.relations.topic_canonicalize import knn_topic_candidates

        conn, cur = _mock_conn([{"topic_ko": "요리"}, {"topic_ko": "음식"}])
        out = knn_topic_candidates(conn, "식품", k=5)

        self.assertEqual(out, ["요리", "음식"])
        sql = _compact_sql(cur)
        # pgvector 코사인 거리 <=> + 결정적 정렬(거리 asc → topic_ko asc)
        self.assertIn("<=>", sql)
        self.assertIn("ORDER BY", sql)
        self.assertIn("topic_ko", sql)
        # 034: 0-노름/NULL 레지스트리 임베딩 제외(kNN 불가시 오염 차단)
        self.assertIn("vector_norm(embedding) > 0", sql)
        self.assertIn("embedding IS NOT NULL", sql)
        # 라벨 임베딩 seam 재사용(질의 임베딩)
        m_embed.assert_called_once_with("식품")
        # k 는 LIMIT 바인딩
        self.assertIn(5, cur.execute.call_args[0][1])

    @patch(f"{_MOD}._embed_label", return_value=[0.1] * 1536)
    def test_empty_registry_returns_empty(self, _m) -> None:
        from src.relations.topic_canonicalize import knn_topic_candidates

        conn, _ = _mock_conn([])
        self.assertEqual(knn_topic_candidates(conn, "양자컴퓨팅"), [])

    @patch(f"{_MOD}._embed_label", return_value=[0.1] * 1536)
    def test_topic_ko_coerced_to_str(self, _m) -> None:
        from src.relations.topic_canonicalize import knn_topic_candidates

        conn, _ = _mock_conn([{"topic_ko": 123}])
        out = knn_topic_candidates(conn, "x")
        self.assertEqual(out, ["123"])
        self.assertIsInstance(out[0], str)


class TestRegisterTopic(unittest.TestCase):
    """정본 등록 — 임베딩 저장·0-노름 거부(불변식)·ON CONFLICT 멱등·source."""

    @patch(f"{_MOD}._embed_label", return_value=[0.2] * 1536)
    def test_inserts_with_embedding_cast_and_on_conflict(self, m_embed) -> None:
        from src.relations.topic_canonicalize import register_topic

        conn, cur = _mock_conn([])
        register_topic(conn, "양자컴퓨팅", "quantum")

        m_embed.assert_called_once_with("양자컴퓨팅")
        sql = _compact_sql(cur)
        self.assertIn("INSERT INTO topic_registry", sql)
        self.assertIn("::vector(1536)", sql)
        self.assertIn("ON CONFLICT (topic_ko) DO NOTHING", sql)
        params = cur.execute.call_args[0][1]
        self.assertIn("양자컴퓨팅", params)
        self.assertIn("quantum", params)
        self.assertIn("auto", params)  # 기본 source

    @patch(f"{_MOD}._embed_label", return_value=[0.2] * 1536)
    def test_source_override(self, _m) -> None:
        from src.relations.topic_canonicalize import register_topic

        conn, cur = _mock_conn([])
        register_topic(conn, "요리", "cooking", source="seed")
        self.assertIn("seed", cur.execute.call_args[0][1])

    @patch(f"{_MOD}._embed_label", return_value=[0.0] * 1536)
    def test_zero_norm_embedding_rejected(self, _m) -> None:
        # 034 교훈: 0-노름 임베딩은 kNN 불가시 → 동의어 재난립. 앱에서 금지(ValueError).
        from src.relations.topic_canonicalize import register_topic

        conn, cur = _mock_conn([])
        with self.assertRaises(ValueError):
            register_topic(conn, "빈라벨", "empty")
        cur.execute.assert_not_called()  # INSERT 하지 않음


class TestJudgeTopic(unittest.TestCase):
    """LLM 재사용/신규 판정 — 후보 K개만 주입·매칭/NEW·후보밖 방어·빈후보 LLM 0."""

    def test_empty_candidates_no_llm_call_returns_none(self) -> None:
        from src.relations.topic_canonicalize import judge_topic

        client = MagicMock()
        self.assertIsNone(judge_topic("양자컴퓨팅", [], client=client))
        client.chat.completions.create.assert_not_called()

    def test_match_returns_candidate_and_only_candidates_in_prompt(self) -> None:
        from src.relations.topic_canonicalize import judge_topic

        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"match": "요리"}'))]
        )
        out = judge_topic("식품", ["요리", "음식"], client=client)
        self.assertEqual(out, "요리")
        # 후보 K개만 프롬프트에 주입(전체 레지스트리 주입 금지·FR-203)
        prompt = client.chat.completions.create.call_args[1]["messages"][0]["content"]
        self.assertIn("요리", prompt)
        self.assertIn("음식", prompt)
        self.assertIn("식품", prompt)

    def test_new_verdict_returns_none(self) -> None:
        from src.relations.topic_canonicalize import judge_topic

        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"match": "NEW"}'))]
        )
        self.assertIsNone(judge_topic("양자컴퓨팅", ["요리", "음식"], client=client))

    def test_hallucinated_label_not_in_candidates_returns_none(self) -> None:
        # 후보 목록에 없는 라벨을 LLM 이 지어내면 안전하게 None(신규) — 오병합 방지·결정성.
        from src.relations.topic_canonicalize import judge_topic

        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"match": "우주"}'))]
        )
        self.assertIsNone(judge_topic("양자컴퓨팅", ["요리", "음식"], client=client))


class TestCanonicalizeTopic(unittest.TestCase):
    """canonicalize_topic 오케스트레이션 — passthrough·정확일치·kNN 매칭·신규·결정성."""

    def test_empty_raw_ko_passthrough_without_normalization(self) -> None:
        from src.relations.topic_canonicalize import canonicalize_topic

        out = canonicalize_topic(object(), "", None)
        self.assertEqual(out["canonical_ko"], "")
        self.assertEqual(out["canonical_en"], "general")
        self.assertEqual(out["decided_by"], "passthrough")

        out2 = canonicalize_topic(object(), None, "food")
        self.assertIsNone(out2["canonical_ko"])
        self.assertEqual(out2["canonical_en"], "food")
        self.assertEqual(out2["decided_by"], "passthrough")

    @patch(f"{_MOD}._lookup_topic_en", return_value="cooking")
    @patch(f"{_MOD}.knn_topic_candidates")
    @patch(f"{_MOD}.judge_topic")
    @patch(f"{_MOD}.lookup_alias", return_value="요리")
    def test_alias_exact_hit_skips_llm(self, m_alias, m_judge, m_knn, m_en) -> None:
        # T201: alias 정확일치 → canonical 반환 + judge_topic(LLM)·kNN 미호출.
        from src.relations.topic_canonicalize import canonicalize_topic

        conn = object()
        out = canonicalize_topic(conn, "식품", "food")

        self.assertEqual(out, {"canonical_ko": "요리", "canonical_en": "cooking", "decided_by": "exact"})
        m_judge.assert_not_called()   # LLM 0(캐시 히트)
        m_knn.assert_not_called()

    @patch(f"{_MOD}.register_topic")
    @patch(f"{_MOD}._lookup_topic_en", return_value="cooking")
    @patch(f"{_MOD}._freeze_alias")
    @patch(f"{_MOD}.judge_topic", return_value="요리")
    @patch(f"{_MOD}.knn_topic_candidates", return_value=["요리", "음식"])
    @patch(f"{_MOD}.lookup_alias", return_value=None)
    def test_miss_knn_judge_match_freezes_alias(
        self, m_alias, m_knn, m_judge, m_freeze, m_en, m_register
    ) -> None:
        # T203/204: 미스 → kNN 후보 → judge 매칭 → alias 동결(decided_by=llm)·정본 반환.
        from src.relations.topic_canonicalize import canonicalize_topic

        conn = object()
        out = canonicalize_topic(conn, "식품", "food")

        m_knn.assert_called_once_with(conn, "식품")
        m_judge.assert_called_once_with("식품", ["요리", "음식"], client=None)
        m_freeze.assert_called_once_with(conn, "식품", "요리", "llm")
        m_register.assert_not_called()   # 재사용 → 신규 등록 없음
        self.assertEqual(out, {"canonical_ko": "요리", "canonical_en": "cooking", "decided_by": "llm"})

    @patch(f"{_MOD}._lookup_topic_en")
    @patch(f"{_MOD}._freeze_alias")
    @patch(f"{_MOD}.register_topic")
    @patch(f"{_MOD}.judge_topic", return_value=None)
    @patch(f"{_MOD}.knn_topic_candidates", return_value=[])
    @patch(f"{_MOD}.lookup_alias", return_value=None)
    def test_new_topic_registers_and_freezes_self_alias(
        self, m_alias, m_knn, m_judge, m_register, m_freeze, m_en
    ) -> None:
        # T205/206: judge=None(NEW) → register_topic(임베딩 저장) + self-alias 동결 → 신규 정본 반환.
        from src.relations.topic_canonicalize import canonicalize_topic

        conn = object()
        out = canonicalize_topic(conn, "양자컴퓨팅", "quantum computing")

        m_register.assert_called_once_with(conn, "양자컴퓨팅", "quantum computing", source="auto")
        m_freeze.assert_called_once_with(conn, "양자컴퓨팅", "양자컴퓨팅", "llm")
        m_en.assert_not_called()   # 신규 en 은 raw_en 사용(레지스트리 재조회 불필요)
        self.assertEqual(
            out,
            {"canonical_ko": "양자컴퓨팅", "canonical_en": "quantum computing", "decided_by": "llm"},
        )

    def test_new_topic_defaults_en_to_general_when_raw_en_none(self) -> None:
        # raw_en 미제공(None) NEW → topic_en 은 "general" 로 정본화(FR-204).
        from src.relations.topic_canonicalize import canonicalize_topic

        conn = object()
        with patch(f"{_MOD}.lookup_alias", return_value=None), \
             patch(f"{_MOD}.knn_topic_candidates", return_value=[]), \
             patch(f"{_MOD}.judge_topic", return_value=None), \
             patch(f"{_MOD}.register_topic") as m_register, \
             patch(f"{_MOD}._freeze_alias"):
            out = canonicalize_topic(conn, "양자컴퓨팅", None)
        self.assertEqual(out["canonical_en"], "general")
        m_register.assert_called_once_with(conn, "양자컴퓨팅", "general", source="auto")

    def test_same_raw_twice_second_is_cache_hit_llm_zero(self) -> None:
        # T207: 같은 raw 2회 → 2번째 alias 히트·LLM(judge) 0회·동일 정본(결정성·FR-701/SC-05).
        from src.relations.topic_canonicalize import canonicalize_topic

        frozen: dict[str, str] = {}       # topic_alias 인메모리 모사
        registry_en: dict[str, str] = {}  # topic_registry(topic_ko→topic_en) 모사

        def fake_lookup_alias(conn, raw):
            return frozen.get(raw)

        def fake_freeze(conn, raw, canon, decided):
            frozen.setdefault(raw, canon)

        def fake_register(conn, topic_ko, topic_en, source="auto"):
            registry_en[topic_ko] = topic_en

        def fake_lookup_en(conn, topic_ko):
            return registry_en.get(topic_ko)

        with patch(f"{_MOD}.lookup_alias", side_effect=fake_lookup_alias), \
             patch(f"{_MOD}.knn_topic_candidates", return_value=[]) as m_knn, \
             patch(f"{_MOD}.judge_topic", return_value=None) as m_judge, \
             patch(f"{_MOD}.register_topic", side_effect=fake_register) as m_register, \
             patch(f"{_MOD}._freeze_alias", side_effect=fake_freeze), \
             patch(f"{_MOD}._lookup_topic_en", side_effect=fake_lookup_en):
            conn = object()
            first = canonicalize_topic(conn, "양자컴퓨팅", "quantum computing")
            second = canonicalize_topic(conn, "양자컴퓨팅", "quantum computing")

        # 2번째는 캐시 히트: judge(LLM)·kNN·register 각 1회만(첫 호출에서만)
        self.assertEqual(m_judge.call_count, 1)
        self.assertEqual(m_knn.call_count, 1)
        self.assertEqual(m_register.call_count, 1)
        # 정본 결과는 동일(결정성) — decided_by 만 llm→exact(캐시 근거 표식)로 다름
        self.assertEqual(first["canonical_ko"], second["canonical_ko"])
        self.assertEqual(first["canonical_en"], second["canonical_en"])
        self.assertEqual(second["decided_by"], "exact")


if __name__ == "__main__":
    unittest.main()
