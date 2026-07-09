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

    def test_root_scope_filters_parent_null(self) -> None:
        # v297: topic 층(기본) 룩업은 parent_topic IS NULL 스코프로 조여야 한다(subtopic 오히트 방지).
        from src.relations.topic_canonicalize import lookup_alias

        conn, cur = _mock_conn([{"canonical_ko": "음식·요리"}])
        lookup_alias(conn, "식품")
        self.assertIn("parent_topic IS NULL", _compact_sql(cur))

    def test_subtopic_scope_binds_parent(self) -> None:
        # v297: subtopic 층 룩업은 (parent_topic, raw_ko) 스코프 — 동음이의 보존(교통>사고 ≠ 사회>사고).
        from src.relations.topic_canonicalize import lookup_alias

        conn, cur = _mock_conn([{"canonical_ko": "김밥"}])
        out = lookup_alias(conn, "노리마키", parent_topic="음식·요리")
        self.assertEqual(out, "김밥")
        self.assertIn("parent_topic = %s", _compact_sql(cur))
        self.assertEqual(cur.execute.call_args[0][1], ("노리마키", "음식·요리"))


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

    @patch(f"{_MOD}._embed_label", return_value=[0.1] * 1536)
    def test_subtopic_scope_filters_parent(self, _m) -> None:
        # v297: subtopic kNN 후보는 **같은 부모 스코프**만(오병합 폭발 반경 버킷 한정·C3).
        from src.relations.topic_canonicalize import knn_topic_candidates

        conn, cur = _mock_conn([{"topic_ko": "김밥"}])
        out = knn_topic_candidates(conn, "노리마키", parent_topic="음식·요리")
        self.assertEqual(out, ["김밥"])
        sql = _compact_sql(cur)
        self.assertIn("parent_topic = %s", sql)
        self.assertIn("음식·요리", cur.execute.call_args[0][1])


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
        # v297: topic 층(parent_topic None) 등록은 부분 유니크 인덱스(parent NULL) 술어로 인퍼런스.
        self.assertIn("ON CONFLICT (topic_ko) WHERE parent_topic IS NULL DO NOTHING", sql)
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

    @patch(f"{_MOD}._embed_label", return_value=[0.2] * 1536)
    def test_subtopic_layer_uses_parent_scope_on_conflict(self, _m) -> None:
        # v297 subtopic 층 등록: 부분 유니크 인덱스 (parent_topic, topic_ko) 술어 인퍼런스·부모 바인딩.
        from src.relations.topic_canonicalize import register_topic

        conn, cur = _mock_conn([])
        register_topic(conn, "김밥", None, parent_topic="음식·요리")
        sql = _compact_sql(cur)
        self.assertIn(
            "ON CONFLICT (parent_topic, topic_ko) WHERE parent_topic IS NOT NULL DO NOTHING", sql
        )
        self.assertIn("음식·요리", cur.execute.call_args[0][1])

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


class TestJudgePromptSynonymOnly(unittest.TestCase):
    """judge 프롬프트가 **동의어/동일 개념 한정** 판정으로 조여졌는지 문자열 검사(사용자 결정).

    왜 이 가드가 필요한가
        직전 replay 는 judge 가 상위 카테고리·is-a·광역 부모까지 흡수해(등산→스포츠·사진→예술·
        물리학→과학) "주제가 너무 넓어지는" 문제가 재발했다. 이를 막으려면 프롬프트가 **같은 것을
        다른 말로 부르는 동의어일 때만** 매칭하고, 상위분류·종류(is-a)·부분-전체·단순 연관은 NEW 로
        보내도록 지침·예시를 명시해야 한다. temp=0·complete_json·단일 seam(2조)은 유지된다.
    """

    def test_prompt_states_synonym_only_criterion(self) -> None:
        # 동의어/동일 개념 한정 지침이 프롬프트에 명시돼야 한다.
        from src.relations.topic_canonicalize import _JUDGE_PROMPT

        self.assertIn("동의어", _JUDGE_PROMPT)
        # "같은 것을 다른 말로 부르는가" 취지의 기준
        self.assertIn("바꿔", _JUDGE_PROMPT)

    def test_prompt_forbids_hierarchy_and_is_a(self) -> None:
        # 상위분류·is-a(종류/장르/분야)·연관은 매칭 금지(→ NEW) 임을 못박아야 한다.
        from src.relations.topic_canonicalize import _JUDGE_PROMPT

        self.assertIn("상위", _JUDGE_PROMPT)
        self.assertIn("is-a", _JUDGE_PROMPT)
        self.assertIn("NEW", _JUDGE_PROMPT)

    def test_prompt_contains_positive_synonym_example(self) -> None:
        # 긍정(동의어) 예시가 있어야 한다 — 예: 등산 == 산악.
        from src.relations.topic_canonicalize import _JUDGE_PROMPT

        self.assertIn("등산", _JUDGE_PROMPT)
        self.assertIn("산악", _JUDGE_PROMPT)

    def test_prompt_contains_negative_broad_examples(self) -> None:
        # 부정(상위분류·is-a) 예시가 있어야 한다 — 스포츠·예술·과학 광역흡수 재발 방지.
        from src.relations.topic_canonicalize import _JUDGE_PROMPT

        self.assertIn("스포츠", _JUDGE_PROMPT)
        self.assertIn("예술", _JUDGE_PROMPT)
        self.assertIn("과학", _JUDGE_PROMPT)


class TestFetchCanonicalTopics(unittest.TestCase):
    """닫힌 정본 topic 집합 조회 — parent NULL·source='taxonomy'·{ko:en}·결정적 정렬(v2)."""

    def test_returns_ko_en_map_scoped_to_taxonomy_root(self) -> None:
        from src.relations.topic_canonicalize import _fetch_canonical_topics

        conn, cur = _mock_conn(
            [{"topic_ko": "음식·요리", "topic_en": "food_cooking"}, {"topic_ko": "미분류", "topic_en": "unclassified"}]
        )
        out = _fetch_canonical_topics(conn)
        self.assertEqual(out, {"음식·요리": "food_cooking", "미분류": "unclassified"})
        sql = _compact_sql(cur)
        self.assertIn("topic_registry", sql)
        self.assertIn("parent_topic IS NULL", sql)   # 닫힌 topic 층만
        self.assertIn("source", sql)                  # source='taxonomy' 필터
        self.assertIn("ORDER BY", sql)                # 결정적 정렬(프롬프트 후보 순서 안정)

    def test_none_en_preserved(self) -> None:
        from src.relations.topic_canonicalize import _fetch_canonical_topics

        conn, _ = _mock_conn([{"topic_ko": "음악", "topic_en": None}])
        self.assertEqual(_fetch_canonical_topics(conn), {"음악": None})


class TestClassifyTopic(unittest.TestCase):
    """자유 topic → 닫힌 범주 분류(동의어 판정 아님)·후보밖/누락 응답 → 미분류·client 주입(v2)."""

    _CATS = ["미분류", "스포츠·레저", "음식·요리"]

    def test_classifies_into_category_and_lists_all(self) -> None:
        from src.relations.topic_canonicalize import classify_topic

        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"category": "음식·요리"}'))]
        )
        out = classify_topic("김치찌개", "kimchi stew", self._CATS, client=client)
        self.assertEqual(out, "음식·요리")
        prompt = client.chat.completions.create.call_args[1]["messages"][0]["content"]
        for c in self._CATS:                       # 닫힌 목록 전부 프롬프트에(후보=27전체·FR-201v2)
            self.assertIn(c, prompt)
        self.assertIn("김치찌개", prompt)
        self.assertIn("kimchi stew", prompt)

    def test_hallucinated_or_missing_falls_back_to_etc(self) -> None:
        # 목록 밖 라벨·NEW·누락(강제배정·오배정 방지) → 안전하게 '미분류'(결정성).
        from src.relations.topic_canonicalize import classify_topic

        for content in ('{"category": "우주"}', '{"category": "NEW"}', "{}"):
            client = MagicMock()
            client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=content))]
            )
            self.assertEqual(classify_topic("x", None, self._CATS, client=client), "미분류")


class TestClassifyPrompt(unittest.TestCase):
    """분류 프롬프트 가드 — '분류(동의어 아님)'·목록 강제·기타 폴백 지침이 담겨야 한다(v2)."""

    def test_prompt_states_classification_and_etc_escape(self) -> None:
        from src.relations.topic_canonicalize import _CLASSIFY_PROMPT

        self.assertIn("분류", _CLASSIFY_PROMPT)    # 동의어 판정이 아니라 '분류'
        self.assertIn("미분류", _CLASSIFY_PROMPT)    # 확신 없으면 미분류(강제배정 금지)

    def test_prompt_forbids_inventing_labels(self) -> None:
        from src.relations.topic_canonicalize import _CLASSIFY_PROMPT

        self.assertIn("목록", _CLASSIFY_PROMPT)    # 목록 밖 라벨 생성 금지


class TestCanonicalizeTopic(unittest.TestCase):
    """canonicalize_topic v2 — 닫힌 목록(정확일치·alias 캐시·분류 폴백·기타)·신규등록 없음·결정성."""

    def test_empty_raw_ko_passthrough_without_normalization(self) -> None:
        from src.relations.topic_canonicalize import canonicalize_topic

        out = canonicalize_topic(object(), "", None)
        self.assertEqual(out, {"canonical_ko": "", "canonical_en": "general", "decided_by": "passthrough"})

        out2 = canonicalize_topic(object(), None, "food")
        self.assertEqual(out2, {"canonical_ko": None, "canonical_en": "food", "decided_by": "passthrough"})

    @patch(f"{_MOD}.classify_topic")
    @patch(f"{_MOD}._fetch_canonical_topics", return_value={})
    def test_empty_registry_passthrough(self, m_fetch, m_classify) -> None:
        # 레지스트리 미시드 → 원본 유지(동작 보존·G4 T401 하위호환)·분류 LLM 0.
        from src.relations.topic_canonicalize import canonicalize_topic

        out = canonicalize_topic(object(), "식품", "food")
        self.assertEqual(out, {"canonical_ko": "식품", "canonical_en": "food", "decided_by": "passthrough"})
        m_classify.assert_not_called()

    @patch(f"{_MOD}.classify_topic")
    @patch(f"{_MOD}.lookup_alias")
    @patch(f"{_MOD}._fetch_canonical_topics", return_value={"음식·요리": "food_cooking", "미분류": "unclassified"})
    def test_exact_match_to_closed_set_passes_through_no_llm(self, m_fetch, m_alias, m_classify) -> None:
        # 2) 닫힌 정본 집합 정확일치 → 그대로(exact)·분류 LLM 0·alias 룩업 이전 반환.
        from src.relations.topic_canonicalize import canonicalize_topic

        out = canonicalize_topic(object(), "음식·요리", "food_cooking")
        self.assertEqual(out, {"canonical_ko": "음식·요리", "canonical_en": "food_cooking", "decided_by": "exact"})
        m_classify.assert_not_called()
        m_alias.assert_not_called()

    @patch(f"{_MOD}.register_topic")
    @patch(f"{_MOD}.classify_topic")
    @patch(f"{_MOD}.lookup_alias", return_value="음식·요리")
    @patch(f"{_MOD}._fetch_canonical_topics", return_value={"음식·요리": "food_cooking", "미분류": "unclassified"})
    def test_alias_cache_hit_returns_canonical_no_llm(self, m_fetch, m_alias, m_classify, m_register) -> None:
        # 3) alias 캐시(parent NULL) 히트 → canonical + registry en·분류 LLM 0·신규등록 없음.
        from src.relations.topic_canonicalize import canonicalize_topic

        conn = object()
        out = canonicalize_topic(conn, "식품", "food")
        self.assertEqual(out, {"canonical_ko": "음식·요리", "canonical_en": "food_cooking", "decided_by": "exact"})
        m_alias.assert_called_once_with(conn, "식품")
        m_classify.assert_not_called()
        m_register.assert_not_called()

    @patch(f"{_MOD}.register_topic")
    @patch(f"{_MOD}._freeze_alias")
    @patch(f"{_MOD}.classify_topic", return_value="음식·요리")
    @patch(f"{_MOD}.lookup_alias", return_value=None)
    @patch(f"{_MOD}._fetch_canonical_topics", return_value={"음식·요리": "food_cooking", "미분류": "unclassified"})
    def test_offlist_miss_classifies_and_freezes_alias(
        self, m_fetch, m_alias, m_classify, m_freeze, m_register
    ) -> None:
        # 4) 미스 → 분류 LLM(후보=닫힌목록·client 주입) → 매칭 범주·alias 동결(classify)·신규등록 없음.
        from src.relations.topic_canonicalize import canonicalize_topic

        conn = object()
        client = object()
        out = canonicalize_topic(conn, "김치찌개", "kimchi stew", client=client)
        self.assertEqual(out, {"canonical_ko": "음식·요리", "canonical_en": "food_cooking", "decided_by": "classify"})
        args, kwargs = m_classify.call_args
        self.assertEqual(args[0], "김치찌개")
        self.assertEqual(args[1], "kimchi stew")
        self.assertIn("음식·요리", args[2])          # 후보 = 닫힌 목록 전체
        self.assertIn("미분류", args[2])
        self.assertEqual(kwargs.get("client"), client)
        m_freeze.assert_called_once_with(conn, "김치찌개", "음식·요리", "classify")
        m_register.assert_not_called()               # topic 층 신규 등록 없음(v2)

    @patch(f"{_MOD}.register_topic")
    @patch(f"{_MOD}._freeze_alias")
    @patch(f"{_MOD}.classify_topic", return_value="미분류")
    @patch(f"{_MOD}.lookup_alias", return_value=None)
    @patch(f"{_MOD}._fetch_canonical_topics", return_value={"음식·요리": "food_cooking", "미분류": "unclassified"})
    def test_ambiguous_falls_back_to_etc_and_logs(
        self, m_fetch, m_alias, m_classify, m_freeze, m_register
    ) -> None:
        # 4) 애매 → 기타 폴백·alias 동결·제안 라벨(원본) 로그(거버넌스 §4·가산 확장 근거)·신규등록 없음.
        from src.relations.topic_canonicalize import canonicalize_topic

        conn = object()
        with self.assertLogs("src.relations.topic_canonicalize", level="INFO") as cm:
            out = canonicalize_topic(conn, "블록체인딜라이트", "blockchain delight")
        self.assertEqual(out, {"canonical_ko": "미분류", "canonical_en": "unclassified", "decided_by": "classify"})
        m_freeze.assert_called_once_with(conn, "블록체인딜라이트", "미분류", "classify")
        m_register.assert_not_called()
        self.assertTrue(any("블록체인딜라이트" in line for line in cm.output))

    def test_no_new_topic_registration_in_any_path(self) -> None:
        # v2 불변: topic 층은 고정(닫힌 27+기타) — 어떤 경로에서도 register_topic 을 부르지 않는다(쌍별 등록 제거).
        from src.relations.topic_canonicalize import canonicalize_topic

        with patch(f"{_MOD}._fetch_canonical_topics", return_value={"미분류": "unclassified"}), \
             patch(f"{_MOD}.lookup_alias", return_value=None), \
             patch(f"{_MOD}.classify_topic", return_value="미분류"), \
             patch(f"{_MOD}._freeze_alias"), \
             patch(f"{_MOD}.register_topic") as m_register:
            canonicalize_topic(object(), "임의라벨", None)
        m_register.assert_not_called()

    def test_same_raw_twice_second_is_cache_hit_llm_zero(self) -> None:
        # 결정성(SC-04v2): 같은 raw 2회 → 2번째 alias 히트·분류 LLM 0·동일 정본.
        from src.relations.topic_canonicalize import canonicalize_topic

        canonical = {"음식·요리": "food_cooking", "미분류": "unclassified"}
        frozen: dict[str, str] = {}

        def fake_lookup_alias(conn, raw, *, parent_topic=None):
            return frozen.get(raw)

        def fake_freeze(conn, raw, canon, decided, *, parent_topic=None):
            frozen.setdefault(raw, canon)

        with patch(f"{_MOD}._fetch_canonical_topics", return_value=canonical), \
             patch(f"{_MOD}.lookup_alias", side_effect=fake_lookup_alias), \
             patch(f"{_MOD}.classify_topic", return_value="음식·요리") as m_classify, \
             patch(f"{_MOD}._freeze_alias", side_effect=fake_freeze):
            conn = object()
            first = canonicalize_topic(conn, "김치찌개", "kimchi stew")
            second = canonicalize_topic(conn, "김치찌개", "kimchi stew")

        self.assertEqual(m_classify.call_count, 1)   # 2번째는 캐시 히트(LLM 0)
        self.assertEqual(first["canonical_ko"], second["canonical_ko"])
        self.assertEqual(first["canonical_en"], second["canonical_en"])
        self.assertEqual(first["decided_by"], "classify")
        self.assertEqual(second["decided_by"], "exact")


class TestCanonicalizeSubtopic(unittest.TestCase):
    """canonicalize_subtopic v2 — 부모 스코프 해소(FR-202v2)·C7 규칙(모달리티·부모 범주명 비움).

    규칙 순서:
      0. None/빈 → None · 모달리티 블랙리스트 → None · raw_sub == 부모 범주명 → None(C7).
      1. (부모, raw) alias 정확일치 → 그 정본(부모 스코프 캐시).
      2. 미스 → 같은 부모 스코프 kNN 후보.
      3. 동의어-한정 judge(기존 judge_topic 재사용·후보 same-parent 만) → 매칭 or NEW.
      4. NEW → register_topic(부모 스코프)·매칭=그 정본. **반드시 같은 parent 스코프로 register→freeze**
         (스코프 불변식·plan). alias 동결(부모, raw).
    """

    def test_none_or_empty_returns_none(self) -> None:
        # 0: None/빈/공백만 → None. conn 접근 없이 조기 반환(object() 로 DB 미접근 확인).
        from src.relations.topic_canonicalize import canonicalize_subtopic

        self.assertIsNone(canonicalize_subtopic(object(), "음식·요리", None))
        self.assertIsNone(canonicalize_subtopic(object(), "음식·요리", ""))
        self.assertIsNone(canonicalize_subtopic(object(), "음식·요리", "   "))

    def test_modality_words_ko_and_en_return_none(self) -> None:
        # C7 (모달리티): 매체어(텍스트/오디오/영상/이미지 + en·대소문자 무관) → None.
        from src.relations.topic_canonicalize import canonicalize_subtopic

        for w in [
            "텍스트", "오디오", "영상", "이미지",
            "text", "audio", "video", "image",
            "TEXT", "Video", "AUDIO", "Image",
        ]:
            self.assertIsNone(canonicalize_subtopic(object(), "음식·요리", w), msg=w)

    def test_modality_first_token_return_none(self) -> None:
        # 정규화(한 어절) 후 매체어면 차단: "audio file" → "audio" → None.
        from src.relations.topic_canonicalize import canonicalize_subtopic

        self.assertIsNone(canonicalize_subtopic(object(), "음식·요리", "audio file"))
        self.assertIsNone(canonicalize_subtopic(object(), "음식·요리", "text 데이터"))

    def test_subtopic_equal_parent_category_name_returns_none(self) -> None:
        # C7 (부모 범주명 비움): raw_sub == 부모 범주명 → None. registry/alias 조회 전에 차단(비용 0).
        from src.relations.topic_canonicalize import canonicalize_subtopic

        with patch(f"{_MOD}.lookup_alias") as m_alias, \
             patch(f"{_MOD}.knn_topic_candidates") as m_knn:
            self.assertIsNone(canonicalize_subtopic(object(), "음식·요리", "음식·요리"))
            m_alias.assert_not_called()
            m_knn.assert_not_called()

    def test_modality_check_precedes_registry_lookup(self) -> None:
        # 모달리티어는 registry/alias 조회 전에 차단(비용 0·결정성) — 룩업/kNN 미호출.
        from src.relations.topic_canonicalize import canonicalize_subtopic

        with patch(f"{_MOD}.lookup_alias") as m_alias, \
             patch(f"{_MOD}.knn_topic_candidates") as m_knn:
            self.assertIsNone(canonicalize_subtopic(object(), "음식·요리", "텍스트"))
            m_alias.assert_not_called()
            m_knn.assert_not_called()

    @patch(f"{_MOD}.knn_topic_candidates")
    @patch(f"{_MOD}.lookup_alias", return_value="김밥")
    def test_parent_scoped_alias_exact_hit(self, m_alias, m_knn) -> None:
        # 1: (부모, raw) alias 정확일치 → 그 정본·kNN/judge 미호출.
        from src.relations.topic_canonicalize import canonicalize_subtopic

        conn = object()
        out = canonicalize_subtopic(conn, "음식·요리", "김밥 만들기")
        self.assertEqual(out, "김밥")
        m_alias.assert_called_once_with(conn, "김밥", parent_topic="음식·요리")  # 정규화 후 부모 스코프 룩업
        m_knn.assert_not_called()

    @patch(f"{_MOD}.register_topic")
    @patch(f"{_MOD}._freeze_alias")
    @patch(f"{_MOD}.judge_topic", return_value="김밥")
    @patch(f"{_MOD}.knn_topic_candidates", return_value=["김밥", "주먹밥"])
    @patch(f"{_MOD}.lookup_alias", return_value=None)
    def test_miss_same_parent_knn_judge_match_freezes(
        self, m_alias, m_knn, m_judge, m_freeze, m_register
    ) -> None:
        # 2·3: 미스 → 같은 부모 kNN → 동의어 judge 매칭 → alias 동결(부모 스코프)·정본 반환·신규등록 없음.
        from src.relations.topic_canonicalize import canonicalize_subtopic

        conn = object()
        client = object()
        out = canonicalize_subtopic(conn, "음식·요리", "노리마키", client=client)
        self.assertEqual(out, "김밥")
        m_knn.assert_called_once_with(conn, "노리마키", parent_topic="음식·요리")  # 같은 부모 스코프
        m_judge.assert_called_once_with("노리마키", ["김밥", "주먹밥"], client=client)
        m_freeze.assert_called_once_with(conn, "노리마키", "김밥", "llm", parent_topic="음식·요리")
        m_register.assert_not_called()

    @patch(f"{_MOD}.register_topic")
    @patch(f"{_MOD}._freeze_alias")
    @patch(f"{_MOD}.judge_topic", return_value=None)
    @patch(f"{_MOD}.knn_topic_candidates", return_value=[])
    @patch(f"{_MOD}.lookup_alias", return_value=None)
    def test_new_subtopic_registers_and_freezes_same_parent_scope(
        self, m_alias, m_knn, m_judge, m_freeze, m_register
    ) -> None:
        # 4: NEW → register/freeze 를 **동일 parent 스코프**로(스코프 불변식·FK 완화 앱 보증). alias 동결(부모,raw).
        from src.relations.topic_canonicalize import canonicalize_subtopic

        conn = object()
        out = canonicalize_subtopic(conn, "음식·요리", "떡볶이")
        self.assertEqual(out, "떡볶이")
        # register 는 부모 스코프 topic_ko=정규화 라벨
        r_args, r_kwargs = m_register.call_args
        self.assertEqual(r_args[1], "떡볶이")
        self.assertEqual(r_kwargs.get("parent_topic"), "음식·요리")
        # freeze 도 동일 부모 스코프(불변식)
        f_args, f_kwargs = m_freeze.call_args
        self.assertEqual(f_args[0], conn)
        self.assertEqual(f_args[1], "떡볶이")
        self.assertEqual(f_kwargs.get("parent_topic"), "음식·요리")

    @patch(f"{_MOD}._freeze_alias")
    @patch(f"{_MOD}.judge_topic", return_value=None)
    @patch(f"{_MOD}.knn_topic_candidates", return_value=[])
    @patch(f"{_MOD}.register_topic")
    @patch(f"{_MOD}.lookup_alias", return_value=None)
    def test_new_subtopic_register_before_freeze_order(
        self, m_alias, m_register, m_knn, m_judge, m_freeze
    ) -> None:
        # 스코프 불변식 보증: register(정본 등록) → freeze(alias 동결) 순서(FK 완화 앱 불변식).
        from src.relations.topic_canonicalize import canonicalize_subtopic

        mgr = MagicMock()
        mgr.attach_mock(m_register, "register")
        mgr.attach_mock(m_freeze, "freeze")
        canonicalize_subtopic(object(), "음식·요리", "떡볶이")
        order = [c[0] for c in mgr.mock_calls]
        self.assertLess(order.index("register"), order.index("freeze"))

    def test_subtopic_deterministic_second_is_cache_hit(self) -> None:
        # 결정성(SC-04v2): 같은 (부모,raw) 2회 → 2번째 alias 히트·judge/register LLM 0·동일 정본.
        from src.relations.topic_canonicalize import canonicalize_subtopic

        frozen: dict[tuple, str] = {}

        def fake_lookup_alias(conn, raw, *, parent_topic=None):
            return frozen.get((parent_topic, raw))

        def fake_freeze(conn, raw, canon, decided, *, parent_topic=None):
            frozen.setdefault((parent_topic, raw), canon)

        with patch(f"{_MOD}.lookup_alias", side_effect=fake_lookup_alias), \
             patch(f"{_MOD}.knn_topic_candidates", return_value=[]) as m_knn, \
             patch(f"{_MOD}.judge_topic", return_value=None) as m_judge, \
             patch(f"{_MOD}.register_topic") as m_register, \
             patch(f"{_MOD}._freeze_alias", side_effect=fake_freeze):
            conn = object()
            first = canonicalize_subtopic(conn, "음식·요리", "떡볶이")
            second = canonicalize_subtopic(conn, "음식·요리", "떡볶이")

        self.assertEqual(first, "떡볶이")
        self.assertEqual(second, "떡볶이")
        self.assertEqual(m_judge.call_count, 1)      # 2번째 캐시 히트(LLM 0)
        self.assertEqual(m_register.call_count, 1)
        self.assertEqual(m_knn.call_count, 1)


class TestSubtopicPreferReuse(unittest.TestCase):
    """065 T603 — subtopic 재사용 완화(``prefer_reuse``): 부모 스코프 기존 subtopic 재사용 우선.

    기본(``prefer_reuse=False``·058 관계 경로)은 동의어-한정 ``judge_topic`` 으로 불변.
    065 자기주제(``prefer_reuse=True``)는 재사용-우선 판정(``judge_subtopic_reuse``)으로 완화해
    코스닝된 카테고리(도시여행 등)를 부모 topic 안에서 재사용시켜 과편화(싱글턴)를 줄인다.
    """

    @patch(f"{_MOD}.register_topic")
    @patch(f"{_MOD}._freeze_alias")
    @patch(f"{_MOD}.judge_subtopic_reuse", return_value="도시여행")
    @patch(f"{_MOD}.judge_topic")
    @patch(f"{_MOD}.knn_topic_candidates", return_value=["도시여행", "전통건축"])
    @patch(f"{_MOD}.lookup_alias", return_value=None)
    def test_prefer_reuse_uses_reuse_judge_and_reuses(
        self, m_alias, m_knn, m_judge, m_reuse, m_freeze, m_register
    ) -> None:
        from src.relations.topic_canonicalize import canonicalize_subtopic

        conn = object()
        client = object()
        out = canonicalize_subtopic(
            conn, "여행", "파리자유여행", client=client, prefer_reuse=True
        )
        self.assertEqual(out, "도시여행")  # 부모 기존 subtopic 재사용
        # 부모 스코프 kNN 이 기존 subtopic 후보를 회수해 재사용 판정에 넘긴다.
        m_knn.assert_called_once_with(conn, "파리자유여행", parent_topic="여행")
        m_reuse.assert_called_once_with("파리자유여행", ["도시여행", "전통건축"], client=client)
        m_judge.assert_not_called()  # prefer_reuse 경로는 완화 judge 사용(strict 동의어 아님)
        m_register.assert_not_called()  # 재사용이므로 신규 등록 없음
        m_freeze.assert_called_once_with(
            conn, "파리자유여행", "도시여행", "llm", parent_topic="여행"
        )

    @patch(f"{_MOD}.register_topic")
    @patch(f"{_MOD}._freeze_alias")
    @patch(f"{_MOD}.judge_subtopic_reuse")
    @patch(f"{_MOD}.judge_topic", return_value="김밥")
    @patch(f"{_MOD}.knn_topic_candidates", return_value=["김밥"])
    @patch(f"{_MOD}.lookup_alias", return_value=None)
    def test_default_prefer_reuse_false_uses_strict_judge(
        self, m_alias, m_knn, m_judge, m_reuse, m_freeze, m_register
    ) -> None:
        # 058 관계 경로(기본): 완화 미적용 — strict judge_topic 그대로(동작 보존).
        from src.relations.topic_canonicalize import canonicalize_subtopic

        out = canonicalize_subtopic(object(), "음식·요리", "노리마키")
        self.assertEqual(out, "김밥")
        m_judge.assert_called_once()
        m_reuse.assert_not_called()

    def test_judge_subtopic_reuse_prompt_prefers_category_reuse(self) -> None:
        # 재사용-우선 프롬프트는 "같은 일반 카테고리면 재사용" 취지를 담아야 한다(코스닝 유도).
        from src.relations import topic_canonicalize as tc

        self.assertTrue(hasattr(tc, "judge_subtopic_reuse"))
        self.assertIn("카테고리", tc._SUBTOPIC_REUSE_PROMPT)

    def test_judge_subtopic_reuse_no_candidates_returns_none_no_llm(self) -> None:
        from src.relations.topic_canonicalize import judge_subtopic_reuse

        client = MagicMock()
        self.assertIsNone(judge_subtopic_reuse("도시여행", [], client=client))
        client.chat.completions.create.assert_not_called()

    def test_judge_subtopic_reuse_out_of_candidate_returns_none(self) -> None:
        # LLM 이 후보 밖 라벨을 지어내면 None(오병합 방지·결정성) — 후보 목록 안일 때만 재사용.
        from src.relations.topic_canonicalize import judge_subtopic_reuse

        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"match": "엉뚱한것"}'))]
        )
        self.assertIsNone(judge_subtopic_reuse("도시여행", ["전통건축"], client=client))


class TestModalityBlacklist(unittest.TestCase):
    """모달리티 블랙리스트 단일 출처(plan §계약)."""

    def test_blacklist_is_single_source_module_constant(self) -> None:
        from src.relations.topic_canonicalize import _MODALITY_BLACKLIST

        self.assertEqual(
            set(_MODALITY_BLACKLIST),
            {"텍스트", "오디오", "영상", "이미지", "text", "audio", "video", "image"},
        )


if __name__ == "__main__":
    unittest.main()
