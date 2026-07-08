"""065 자산 자기주제(aboutness) 정본화 — 분류 코어 단위 테스트 (mock, DB·LLM 불필요).

검증 의도 (FR-101/102·FR-201~204)
    자산 스스로의 (topic, subtopic) 정본을 하이브리드(임베딩 kNN → LLM 닫힌 확정 → 058 canonicalize)로
    부여하는 seam. DB/LLM 없이 mock conn·mock client 로 분기·SQL 형상·결정성만 검증한다.
    - **결정성(헌법 3조)**: temp=0 + 닫힌 topic 후보 + 멱등 upsert → 같은 입력 같은 출력.
    - **LLM 단일 seam(헌법 6조)**: ``src.llm.client.complete_json``·``client=`` 주입.
    - **닫힌집합 검증(FR-203)**: LLM 이 후보 밖 topic 을 답하면 1회 재질의 후 실패 시 미부여(강제 매핑 금지).

mock 패턴은 ``tests/test_topic_canonicalize.py``(cursor mock·_mock_conn)·``tests/test_topic_query.py`` 동형.
classify_asset_topic 은 헬퍼(knn_topic_candidates·canonicalize_subtopic·_lookup_topic_en)를
**asset_topic 모듈 위치에서** patch 해 분기만 순수 검증한다.
"""
from __future__ import annotations

import os
import re
import unittest
from unittest.mock import MagicMock, patch

_MOD = "src.classify.asset_topic"

# 마이그레이션 파일 경로(레포 루트 기준·CI 무관). 이 테스트 파일: tests/test_asset_topic_classify.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SQL_PATH = os.path.join(_REPO_ROOT, "migrations", "sql", "299_asset_topic.sql")
_ALEMBIC_PATH = os.path.join(
    _REPO_ROOT, "migrations", "alembic", "versions", "v299_asset_topic.py"
)


class TestMigrationV299(unittest.TestCase):
    """T101 — v299 asset_topic DDL 파일 존재 + 필수 컬럼/제약 문자열(파일 파싱·DB 불요)."""

    def test_sql_file_exists(self) -> None:
        self.assertTrue(
            os.path.isfile(_SQL_PATH), f"299_asset_topic.sql 이 없다: {_SQL_PATH}"
        )

    def test_sql_defines_asset_topic_table_and_columns(self) -> None:
        with open(_SQL_PATH, encoding="utf-8") as fh:
            sql = fh.read().lower()
        # 테이블·PK·필수 컬럼·정책버전·인덱스가 DDL 에 문자열로 존재해야 한다.
        self.assertIn("create table", sql)
        self.assertIn("asset_topic", sql)
        self.assertIn("asset_id", sql)
        self.assertIn("topic_ko", sql)
        self.assertIn("policy_version", sql)
        # 자산 삭제 시 자기주제 행 동반 삭제(FR-101) — ON DELETE CASCADE.
        self.assertIn("on delete cascade", sql)
        # 파생 조인·패싯용 (topic_ko, subtopic_ko) 인덱스.
        self.assertIn("idx_asset_topic_pair", sql)

    def test_alembic_revision_chains_and_reversible(self) -> None:
        with open(_ALEMBIC_PATH, encoding="utf-8") as fh:
            src = fh.read()
        # down_revision 이 실제 v298 revision id 로 체인 연결.
        self.assertIn("v298_labels_schema_object", src)
        # run_sql_file 관례로 SQL 실행 + downgrade 는 DROP TABLE.
        self.assertIn("run_sql_file", src)
        self.assertIn("299_asset_topic.sql", src)
        self.assertRegex(src, r"(?i)drop\s+table\s+if\s+exists\s+asset_topic")
        # revision id 는 alembic_version.version_num(VARCHAR(32)) 제약 — 32자 이하.
        m = re.search(r'^revision\s*=\s*["\']([^"\']+)["\']', src, re.MULTILINE)
        self.assertIsNotNone(m, "revision id 를 찾지 못했다")
        self.assertLessEqual(len(m.group(1)), 32)


import json  # noqa: E402  (테스트 헬퍼용 — 상단 import 블록 아래 배치)

_FIXTURE_PATH = os.path.join(
    _REPO_ROOT, "tests", "fixtures", "topics", "same_topic_groups_contract.json"
)


def _mock_conn_seq(fetchone_val=None, fetchall_val=None):
    """``conn.cursor(...)`` 컨텍스트매니저 mock — fetchone/fetchall 을 각각 통제.

    ``__enter__`` 가 cur 를 돌려주고 fetchone/fetchall 이 주입값을 반환한다. 같은 cur 를 여러 query 가
    공유하므로, 서로 다른 접근자(fetchone vs fetchall)를 쓰는 2단 쿼리를 한 mock 으로 검증할 수 있다.
    """
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.fetchone.return_value = fetchone_val
    cur.fetchall.return_value = fetchall_val if fetchall_val is not None else []
    conn.cursor.return_value = cur
    return conn, cur


def _client_once(content: str):
    """complete_json 이 호출하는 client.chat.completions.create 를 흉내(고정 응답·재호출도 동일)."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )
    return client


def _client_seq(*contents: str):
    """호출마다 다른 응답(재질의 시나리오용) — side_effect 순차 반환."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=c))]) for c in contents
    ]
    return client


class TestBuildSelfText(unittest.TestCase):
    """T201 — 자기 텍스트 결정적 구성(summary → keywords → labels)·빈/None 안전."""

    def test_deterministic_order_summary_keywords_labels(self) -> None:
        from src.classify.asset_topic import build_self_text

        # 겹치지 않는 토큰으로 순서만 검증(부분문자열 충돌 방지).
        out = build_self_text(
            "요약본문",
            ["키워드하나", "키워드둘"],
            [{"label": "sport", "score": 0.9}, {"label": "ball", "score": 0.8}],
        )
        # summary 가 맨 앞, 그다음 keywords, 그다음 labels 순서(재실행 동일).
        self.assertTrue(out.startswith("요약본문"))
        self.assertLess(out.index("요약본문"), out.index("키워드하나"))
        self.assertLess(out.index("키워드하나"), out.index("sport"))
        self.assertLess(out.index("sport"), out.index("ball"))

    def test_labels_dict_takes_label_only(self) -> None:
        from src.classify.asset_topic import build_self_text

        out = build_self_text(None, None, [{"label": "cat", "score": 0.5}])
        self.assertEqual(out, "cat")  # score 는 제외, label 만

    def test_all_empty_returns_empty_string(self) -> None:
        from src.classify.asset_topic import build_self_text

        self.assertEqual(build_self_text(None, None, None), "")
        self.assertEqual(build_self_text("", [], []), "")
        self.assertEqual(build_self_text("  ", ["  ", None], [{}]), "")

    def test_none_keywords_and_labels_safe(self) -> None:
        from src.classify.asset_topic import build_self_text

        self.assertEqual(build_self_text("요약만", None), "요약만")


class TestTopicCandidatesForSelfText(unittest.TestCase):
    """T202 — kNN 재사용 어댑터(058 topic 층 kNN)·빈 텍스트/후보 [] 처리."""

    def test_empty_text_returns_empty_no_knn(self) -> None:
        from src.classify import asset_topic

        with patch.object(asset_topic, "knn_topic_candidates") as m_knn:
            out = asset_topic.topic_candidates_for_self_text(object(), "")
            self.assertEqual(out, [])
            m_knn.assert_not_called()  # 빈 텍스트면 임베딩·kNN 자체를 건너뜀

    def test_delegates_to_058_knn_topic_layer(self) -> None:
        from src.classify import asset_topic

        conn = object()
        with patch.object(
            asset_topic, "knn_topic_candidates", return_value=["스포츠·레저", "예술"]
        ) as m_knn:
            out = asset_topic.topic_candidates_for_self_text(conn, "농구 경기", k=5)
            self.assertEqual(out, ["스포츠·레저", "예술"])
            # topic 층 kNN = parent_topic=None(058 프리미티브 재사용).
            args, kwargs = m_knn.call_args
            self.assertIs(args[0], conn)
            self.assertEqual(args[1], "농구 경기")
            self.assertEqual(kwargs.get("parent_topic"), None)

    def test_empty_candidates_passthrough(self) -> None:
        from src.classify import asset_topic

        with patch.object(asset_topic, "knn_topic_candidates", return_value=[]):
            self.assertEqual(
                asset_topic.topic_candidates_for_self_text(object(), "미시드"), []
            )


_TOPIC_JSON = (
    '{"topic_ko": "스포츠·레저", "topic_en": "sports_leisure", '
    '"subtopic_ko": "농구", "subtopic_en": "basketball", "confidence": 0.9}'
)


class TestClassifyAssetTopic(unittest.TestCase):
    """T203 — 하이브리드 판정: 후보 검증·재질의·미부여·결정성·canonicalize 인자."""

    def _patched(self, candidates, sub_return="농구", en_return="sports_leisure"):
        """knn·canonicalize_subtopic·_lookup_topic_en 을 asset_topic 위치에서 patch 하는 컨텍스트 묶음."""
        from src.classify import asset_topic

        p_knn = patch.object(
            asset_topic, "topic_candidates_for_self_text", return_value=candidates
        )
        p_sub = patch.object(
            asset_topic, "canonicalize_subtopic", return_value=sub_return
        )
        p_en = patch.object(asset_topic, "_lookup_topic_en", return_value=en_return)
        return p_knn, p_sub, p_en

    def test_candidate_topic_returns_dict(self) -> None:
        from src.classify import asset_topic

        p_knn, p_sub, p_en = self._patched(["스포츠·레저", "예술"])
        client = _client_once(_TOPIC_JSON)
        with p_knn, p_sub, p_en:
            out = asset_topic.classify_asset_topic(
                MagicMock(), "A1", self_text="농구 경기 영상", client=client
            )
        self.assertEqual(out["topic_ko"], "스포츠·레저")
        self.assertEqual(out["topic_en"], "sports_leisure")
        self.assertEqual(out["subtopic_ko"], "농구")
        self.assertEqual(out["subtopic_en"], "basketball")
        self.assertEqual(out["confidence"], 0.9)
        self.assertEqual(out["decided_by"], "hybrid")
        # 후보 내 topic → 재질의 없음(LLM 1회).
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_out_of_candidate_then_requery_success(self) -> None:
        from src.classify import asset_topic

        p_knn, p_sub, p_en = self._patched(["스포츠·레저"])
        # 1차: 후보 밖('정치') → 2차 재질의: 후보 내('스포츠·레저').
        bad = '{"topic_ko": "정치", "subtopic_ko": "선거", "confidence": 0.7}'
        client = _client_seq(bad, _TOPIC_JSON)
        with p_knn, p_sub, p_en:
            out = asset_topic.classify_asset_topic(
                MagicMock(), "A2", self_text="농구", client=client
            )
        self.assertEqual(out["topic_ko"], "스포츠·레저")
        self.assertEqual(client.chat.completions.create.call_count, 2)  # 정확히 1회 재질의

    def test_two_failures_return_none(self) -> None:
        from src.classify import asset_topic

        p_knn, p_sub, p_en = self._patched(["스포츠·레저"])
        bad = '{"topic_ko": "정치", "confidence": 0.7}'
        client = _client_seq(bad, bad)
        with p_knn, p_sub, p_en:
            # canonicalize·upsert 가 호출되지 않아야 함(강제 매핑 금지).
            with patch.object(asset_topic, "canonicalize_subtopic") as m_sub:
                out = asset_topic.classify_asset_topic(
                    MagicMock(), "A3", self_text="농구", client=client
                )
                self.assertIsNone(out)
                m_sub.assert_not_called()
        self.assertEqual(client.chat.completions.create.call_count, 2)  # 2회 실패 후 중단

    def test_no_text_returns_none_no_llm(self) -> None:
        from src.classify import asset_topic

        client = _client_once(_TOPIC_JSON)
        with patch.object(asset_topic, "topic_candidates_for_self_text") as m_knn:
            out = asset_topic.classify_asset_topic(
                MagicMock(), "A4", self_text="", client=client
            )
        self.assertIsNone(out)
        client.chat.completions.create.assert_not_called()  # 텍스트 없음 → LLM 미호출
        m_knn.assert_not_called()

    def test_empty_candidates_returns_none_no_llm(self) -> None:
        from src.classify import asset_topic

        p_knn, p_sub, p_en = self._patched([])  # 레지스트리 미시드 가드
        client = _client_once(_TOPIC_JSON)
        with p_knn, p_sub, p_en:
            out = asset_topic.classify_asset_topic(
                MagicMock(), "A5", self_text="농구", client=client
            )
        self.assertIsNone(out)
        client.chat.completions.create.assert_not_called()

    def test_same_input_deterministic_output(self) -> None:
        from src.classify import asset_topic

        p_knn, p_sub, p_en = self._patched(["스포츠·레저"])
        with p_knn, p_sub, p_en:
            out1 = asset_topic.classify_asset_topic(
                MagicMock(), "A6", self_text="농구", client=_client_once(_TOPIC_JSON)
            )
            out2 = asset_topic.classify_asset_topic(
                MagicMock(), "A6", self_text="농구", client=_client_once(_TOPIC_JSON)
            )
        self.assertEqual(out1, out2)  # temp=0·닫힌 후보 → 결정적

    def test_canonicalize_subtopic_called_with_confirmed_topic(self) -> None:
        from src.classify import asset_topic

        p_knn, _p_sub, p_en = self._patched(["스포츠·레저"])
        client = _client_once(_TOPIC_JSON)
        with p_knn, p_en, patch.object(
            asset_topic, "canonicalize_subtopic", return_value="농구"
        ) as m_sub:
            conn = MagicMock()
            asset_topic.classify_asset_topic(conn, "A7", self_text="농구", client=client)
            # (conn, 확정 topic_ko, LLM raw subtopic, client=client) 로 058 정본화 호출.
            args, kwargs = m_sub.call_args
            self.assertIs(args[0], conn)
            self.assertEqual(args[1], "스포츠·레저")
            self.assertEqual(args[2], "농구")
            self.assertIs(kwargs.get("client"), client)

    def test_upsert_sql_shape_on_conflict(self) -> None:
        from src.classify import asset_topic

        p_knn, p_sub, p_en = self._patched(["스포츠·레저"])
        conn, cur = _mock_conn_seq()
        with p_knn, p_sub, p_en:
            asset_topic.classify_asset_topic(
                conn, "A8", self_text="농구", client=_client_once(_TOPIC_JSON)
            )
        sql = " ".join(cur.execute.call_args[0][0].split()).lower()
        self.assertIn("insert into asset_topic", sql)
        self.assertIn("on conflict (asset_id) do update", sql)
        self.assertIn("updated_at = now()", sql)
        # policy_version 이 파라미터로 기록되는지(FR-601).
        params = cur.execute.call_args[0][1]
        self.assertIn(asset_topic.POLICY_VERSION, params)


class TestFetchAssetTopic(unittest.TestCase):
    """T204 — 정본 읽기(구 project_asset_topics 형상)·부재 []."""

    def _fixture_keys(self):
        with open(_FIXTURE_PATH, encoding="utf-8") as fh:
            fx = json.load(fh)
        return set(fx["project_asset_topics_shape"][0].keys())

    def test_row_present_returns_weight_one_shape(self) -> None:
        from src.classify.asset_topic import fetch_asset_topic

        row = {
            "topic_ko": "스포츠·레저",
            "subtopic_ko": "농구",
            "topic_en": "sports_leisure",
            "subtopic_en": "basketball",
        }
        conn, _ = _mock_conn_seq(fetchone_val=row)
        out = fetch_asset_topic(conn, "A1")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["weight"], 1)
        # 필드명 수준 fixture(구 투영 형상)와 동일해야 소비처 무변경 스왑 가능.
        self.assertEqual(set(out[0].keys()), self._fixture_keys())
        self.assertEqual(out[0]["topic_ko"], "스포츠·레저")

    def test_absent_returns_empty(self) -> None:
        from src.classify.asset_topic import fetch_asset_topic

        conn, _ = _mock_conn_seq(fetchone_val=None)
        self.assertEqual(fetch_asset_topic(conn, "missing"), [])


class TestFindSameTopicGroups(unittest.TestCase):
    """T204 — 같은 (topic,subtopic) 자산 집계(구 find_topic_neighbor_groups 형상)."""

    def _fixture_group_keys(self):
        with open(_FIXTURE_PATH, encoding="utf-8") as fh:
            fx = json.load(fh)
        g = fx["find_topic_neighbor_groups_shape"][0]
        sub = g["subtopics"][0]
        asset = sub["assets"][0]
        return set(g.keys()), set(sub.keys()), set(asset.keys())

    def test_no_target_topic_returns_empty(self) -> None:
        from src.classify.asset_topic import find_same_topic_groups

        conn, _ = _mock_conn_seq(fetchone_val=None)  # 대상 자산 asset_topic 행 없음
        self.assertEqual(find_same_topic_groups(conn, "A"), [])

    def test_pair_match_shape_equals_contract(self) -> None:
        from src.classify.asset_topic import find_same_topic_groups

        target = {"topic_ko": "스포츠·레저", "subtopic_ko": "농구"}
        cand_rows = [
            {"asset_id": "019f-1", "topic_ko": "스포츠·레저", "subtopic_ko": "농구",
             "fs_path": "/d/019f-1__wikipedia_농구_5166.txt", "modality": "text",
             "already_linked": True},
            {"asset_id": "019f-2", "topic_ko": "스포츠·레저", "subtopic_ko": "농구",
             "fs_path": "/d/019f-2__Kim_Tae-sul_(농구).JPG", "modality": "image",
             "already_linked": False},
        ]
        conn, cur = _mock_conn_seq(fetchone_val=target, fetchall_val=cand_rows)
        out = find_same_topic_groups(conn, "TARGET")

        self.assertEqual(len(out), 1)
        gk, sk, ak = self._fixture_group_keys()
        self.assertEqual(set(out[0].keys()), gk)
        self.assertEqual(out[0]["topic_ko"], "스포츠·레저")
        self.assertEqual(out[0]["asset_count"], 2)
        sub = out[0]["subtopics"][0]
        self.assertEqual(set(sub.keys()), sk)
        self.assertEqual(sub["subtopic_ko"], "농구")
        self.assertEqual(sub["asset_count"], 2)
        asset0 = sub["assets"][0]
        self.assertEqual(set(asset0.keys()), ak)
        # file_name 은 fs_path basename(관례)·asset_id 는 str.
        self.assertTrue(asset0["file_name"].endswith(".txt") or asset0["file_name"].endswith(".JPG"))
        self.assertIsInstance(asset0["asset_id"], str)
        # subtopic 있는 대상 → 후보 쿼리에 subtopic 필터가 걸려야(같은 쌍 매칭).
        cand_sql = " ".join(cur.execute.call_args_list[1][0][0].split()).lower()
        self.assertIn("subtopic_ko", cand_sql)

    def test_topic_only_match_when_target_subtopic_none(self) -> None:
        from src.classify.asset_topic import find_same_topic_groups

        target = {"topic_ko": "음식·요리", "subtopic_ko": None}
        cand_rows = [
            {"asset_id": "b1", "topic_ko": "음식·요리", "subtopic_ko": "제빵",
             "fs_path": "/d/b1__bread.jpg", "modality": "image", "already_linked": False},
            {"asset_id": "b2", "topic_ko": "음식·요리", "subtopic_ko": None,
             "fs_path": "/d/b2__food.txt", "modality": "text", "already_linked": True},
        ]
        conn, cur = _mock_conn_seq(fetchone_val=target, fetchall_val=cand_rows)
        out = find_same_topic_groups(conn, "TARGET")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["topic_ko"], "음식·요리")
        # 두 하위주제 버킷(제빵·None) 모두 등장 — topic 단독 매칭.
        subs = {s["subtopic_ko"] for s in out[0]["subtopics"]}
        self.assertEqual(subs, {"제빵", None})
        # 대상 subtopic 이 None → 후보 쿼리에 subtopic 필터가 없어야(topic 단독).
        cand_sql = " ".join(cur.execute.call_args_list[1][0][0].split())
        self.assertNotIn("at.subtopic_ko = %s", cand_sql)


if __name__ == "__main__":
    unittest.main()
