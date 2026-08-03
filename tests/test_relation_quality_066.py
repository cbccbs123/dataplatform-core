"""066 관계 품질 개선 — 무내용/미부여 제외 + 자기주제 LLM 투입(soft) 단위 테스트.

DB·LLM 실호출 0(mock conn·seam 주입). spec 066 FR-101/102/103·201/202/203.

테스트 전략(docs/테스트_가이드.md)
  - G1(무내용/미부여 제외): 후보 SELECT 에 ``EXISTS(asset_topic)`` 절 포함(T101),
    미부여 소스는 후보검색·LLM 미호출로 (0,0,0,0) 반환(T102) — 상위가 기존 흐름으로 isolated 종결.
  - G2(자기주제 LLM 투입): 후보 조회에 topic_ko/subtopic_ko 동반(T201·LEFT JOIN),
    프롬프트에 소스·후보 주제 표기 + soft 지시(T202). 하드 배제 아님(정상 크로스-주제 보존).
  - 관계 골격(상태기계·sync_graph_edges·review·scan) 불변 — 여기선 후보·진입·프롬프트 3지점만 검증.
"""

from __future__ import annotations

import types
import unittest
import uuid
from unittest import mock

from src.relations.asset_candidates import find_embedding_candidates
from src.relations.prompt import build_relation_proposal_prompt

_SRC = "018f0000-0000-7000-8000-000000000001"
_T1 = "018f0000-0000-7000-8000-000000000007"
_T2 = "018f0000-0000-7000-8000-000000000008"


def _mock_conn(rows):
    """asset_candidates 가 쓰는 ``conn.cursor(row_factory=dict_row)`` 컨텍스트 매니저 흉내."""
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = rows
    return conn, cur


# ── 순수 단위용 최소 cfg 더블(propose_relations_for_asset 이 읽는 설정만) ─────────
_FAKE_CFG = types.SimpleNamespace(relations=types.SimpleNamespace(
    top_k=10,
    min_sim=0.2,
    path_top_k=10,
    auto_approve_min=0.75,
    auto_approve_emb_min=0.0,
    # 081 게이트. 이 파일이 검증하는 것은 066(후보·진입·프롬프트)이므로 **게이트는 끈 값**을 준다 —
    # 켜면 이 테스트들이 081 동작까지 함께 재게 되어 실패 원인이 흐려진다.
    persist_min_conf_similarity=0.0,
    auto_approve_exclude_kinds="",
    review_exempt_kinds="",
))


class _SingleConnDB:
    """propose_relations_for_asset 의 execute_in_transaction(_run) 만 흉내(실 DB 없이 _run 실행)."""

    def execute_in_transaction(self, fn, *, idempotent: bool = True):
        return fn(mock.MagicMock())


# ── T101 [FR-101] 후보 SELECT 미부여(asset_topic 없음) 배제 ─────────────────────
class TestCandidateExcludesUnassigned(unittest.TestCase):
    """find_embedding_candidates 최종 SELECT 가 미부여 후보를 EXISTS(asset_topic) 로 배제한다."""

    def test_sql_has_asset_topic_exists_clause(self) -> None:
        # registered 필터 옆에 미부여 후보 배제 절이 있어야 남에게 무내용 자산이 노출되지 않는다.
        conn, cur = _mock_conn([])
        find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5)
        norm = " ".join(cur.execute.call_args.args[0].split())
        self.assertIn("EXISTS", norm)
        self.assertIn("asset_topic", norm)
        # a.asset_id 기준 상관 서브쿼리(자산 단위 부여 여부).
        self.assertIn("at.asset_id = a.asset_id", norm)

    def test_existing_ordering_and_params_unchanged(self) -> None:
        # 회귀: 정렬·파라미터 순서·min_sim·top_k 불변(EXISTS/JOIN 추가로 흔들리면 안 됨).
        conn, cur = _mock_conn([])
        find_embedding_candidates(conn, source_asset_id=_SRC, top_k=7, min_sim=0.3)
        norm = " ".join(cur.execute.call_args.args[0].split())
        self.assertIn("ORDER BY p.best_sim DESC, p.id ASC", norm)
        params = cur.execute.call_args.args[1]
        self.assertEqual(params[0], _SRC)
        self.assertEqual(params[2], _SRC)
        self.assertEqual(params[3], 0.3)  # min_sim
        self.assertEqual(params[4], 7)    # top_k


# ── T201 [FR-201] 후보 조회에 주제(topic_ko/subtopic_ko) 동반 ────────────────────
class TestCandidateCarriesTopic(unittest.TestCase):
    """find_embedding_candidates 가 후보별 asset_topic(topic_ko/subtopic_ko)을 LEFT JOIN 으로 실어온다."""

    def test_sql_left_joins_asset_topic(self) -> None:
        conn, cur = _mock_conn([])
        find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5)
        norm = " ".join(cur.execute.call_args.args[0].split())
        self.assertIn("LEFT JOIN asset_topic", norm)
        self.assertIn("t.topic_ko", norm)
        self.assertIn("t.subtopic_ko", norm)

    def test_returned_dict_has_topic_fields(self) -> None:
        rows = [
            {"id": uuid.UUID(_T1), "file_uri": "/d/a.png", "media_type": "image",
             "emb_score": 0.91, "summary": "요약A", "keywords": "", "topic_ko": "여행·관광", "subtopic_ko": "타지마할"},
            {"id": uuid.UUID(_T2), "file_uri": "/d/b.txt", "media_type": "txt",
             "emb_score": 0.42, "summary": None, "keywords": None, "topic_ko": None, "subtopic_ko": None},
        ]
        conn, _ = _mock_conn(rows)
        out = find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5)
        self.assertEqual(out[0]["topic_ko"], "여행·관광")
        self.assertEqual(out[0]["subtopic_ko"], "타지마할")
        # 기존 키 불변(하위호환) — id/summary 등 그대로.
        self.assertEqual(out[0]["id"], _T1)
        self.assertEqual(out[1]["summary"], "")  # None → ''
        # 방어적으로 None 허용(EXISTS 로 미부여는 이미 빠지나 LEFT JOIN 방어).
        self.assertIsNone(out[1]["topic_ko"])


# ── T102 [FR-102/103] 미부여 소스 스킵 → (0,0,0,0) 반환(후보검색·LLM 미호출) ──────
class TestUnassignedSourceSkips(unittest.TestCase):
    """propose_relations_for_asset: 소스 asset_topic 없음이면 후보검색·LLM 미호출·(0,0,0,0) 반환."""

    def test_unassigned_source_returns_zero_without_candidates_or_llm(self) -> None:
        from src.relations import asset_entry as ae

        with mock.patch.object(ae, "get_current_settings", return_value=_FAKE_CFG), \
             mock.patch.object(ae, "_fetch_source_row",
                               return_value={"fs_path": "/d/x.wav", "modality": "audio", "summary": ""}), \
             mock.patch.object(ae, "fetch_asset_topic", return_value=[]) as ft, \
             mock.patch.object(ae, "find_embedding_candidates") as fc, \
             mock.patch.object(ae, "propose_edges_json") as llm, \
             mock.patch.object(ae, "sync_graph_edges") as sync, \
             mock.patch.object(ae, "record_lineage") as lineage:
            result = ae.propose_relations_for_asset(_SingleConnDB(), _SRC, top_k=5)

        self.assertEqual(result, (0, 0, 0, 0))
        ft.assert_called_once()
        fc.assert_not_called()   # 후보검색 스킵
        llm.assert_not_called()  # LLM 스킵(토큰 낭비 0)
        sync.assert_not_called()
        lineage.assert_not_called()

    def test_assigned_source_proceeds_normally(self) -> None:
        # 소스 부여면 기존 경로 정상(후보검색·LLM 호출·엣지 반환).
        from src.relations import asset_entry as ae

        emb_rows = [{"id": _T1, "file_uri": "/d/a.txt", "media_type": "txt",
                     "emb_score": 0.83, "summary": "", "keywords": "", "topic_ko": "여행·관광", "subtopic_ko": "타지마할"}]
        with mock.patch.object(ae, "get_current_settings", return_value=_FAKE_CFG), \
             mock.patch.object(ae, "_fetch_source_row",
                               return_value={"fs_path": "/d/a.txt", "modality": "txt", "summary": "요약"}), \
             mock.patch.object(ae, "fetch_asset_topic",
                               return_value=[{"topic_ko": "음식·요리", "subtopic_ko": "라면"}]), \
             mock.patch.object(ae, "find_embedding_candidates", return_value=emb_rows) as fc, \
             mock.patch.object(ae, "find_path_signal_candidates", return_value=[]), \
             mock.patch.object(ae, "fetch_active_relation_kinds", return_value=[]), \
             mock.patch.object(ae, "register_new_relation_kinds", return_value=(0, 0)), \
             mock.patch.object(ae, "sync_graph_edges", return_value=(1, 0)) as sync, \
             mock.patch.object(ae, "record_lineage", return_value=None):
            result = ae.propose_relations_for_asset(
                _SingleConnDB(), _SRC, top_k=5, llm_fn=lambda _p: {"edges": []})

        fc.assert_called_once()   # 후보검색 정상 호출
        sync.assert_called_once()
        self.assertEqual(result, (0, 0, 1, 0))


class TestSourceTopicWiredToPrompt(unittest.TestCase):
    """T202 배선 — propose_relations_for_asset 가 소스 주제를 build_relation_proposal_prompt 로 전달."""

    def test_source_topic_passed_to_prompt(self) -> None:
        from src.relations import asset_entry as ae

        captured: dict = {}

        def _fake_prompt(*, source_summary, source_media_type, candidates,
                         relation_kinds_catalog, source_topic=None, source_keywords=None,
                         source_filename=None):
            captured["source_topic"] = source_topic
            captured["source_filename"] = source_filename
            return "PROMPT"

        with mock.patch.object(ae, "get_current_settings", return_value=_FAKE_CFG), \
             mock.patch.object(ae, "_fetch_source_row",
                               return_value={"fs_path": "/d/a.txt", "modality": "txt", "summary": "요약"}), \
             mock.patch.object(ae, "fetch_asset_topic",
                               return_value=[{"topic_ko": "음식·요리", "subtopic_ko": "라면"}]), \
             mock.patch.object(ae, "find_embedding_candidates", return_value=[]), \
             mock.patch.object(ae, "find_path_signal_candidates", return_value=[]), \
             mock.patch.object(ae, "fetch_active_relation_kinds", return_value=[]), \
             mock.patch.object(ae, "build_relation_proposal_prompt", side_effect=_fake_prompt), \
             mock.patch.object(ae, "register_new_relation_kinds", return_value=(0, 0)), \
             mock.patch.object(ae, "sync_graph_edges", return_value=(0, 0)), \
             mock.patch.object(ae, "record_lineage", return_value=None):
            ae.propose_relations_for_asset(
                _SingleConnDB(), _SRC, top_k=5, llm_fn=lambda _p: {"edges": []})

        self.assertEqual(captured["source_topic"],
                         {"topic_ko": "음식·요리", "subtopic_ko": "라면"})
        # 소스 파일명도 함께 전달된다(2026-08-03 채택) — **basename 만**이어야 한다.
        # 디렉터리 경로가 새면 LLM 입력에 환경 의존·개인정보가 들어간다(헌법 3조·10조).
        self.assertEqual(captured["source_filename"], "a.txt")


# ── T202 [FR-202/203] 프롬프트 주제 표기 + soft 지시 ────────────────────────────
class TestPromptTopicAndSoftGuidance(unittest.TestCase):
    """build_relation_proposal_prompt 가 source_topic 을 수용하고 소스·후보 주제 + soft 지시를 담는다."""

    def _build(self, *, source_topic=None) -> str:
        return build_relation_proposal_prompt(
            source_summary="소스 요약",
            source_media_type="txt",
            source_topic=source_topic,
            candidates=[
                {
                    "id": _T1,
                    "file_uri": "/data/foo/bar.txt",
                    "media_type": "txt",
                    "emb_score": 0.42,
                    "summary": "후보 요약",
                    "topic_ko": "여행·관광",
                    "subtopic_ko": "타지마할",
                }
            ],
            relation_kinds_catalog=[
                {"type_code": "same_domain", "type_name": "같은 도메인", "description": "같은 분야"},
                {"type_code": "duplicate_near", "type_name": "근접중복", "description": "유사"},
            ],
        )

    def test_accepts_source_topic_param(self) -> None:
        # source_topic 키워드 파라미터를 수용(TypeError 없이 문자열 반환).
        prompt = self._build(source_topic={"topic_ko": "음식·요리", "subtopic_ko": "라면"})
        self.assertIsInstance(prompt, str)

    def test_source_topic_rendered(self) -> None:
        prompt = self._build(source_topic={"topic_ko": "음식·요리", "subtopic_ko": "라면"})
        self.assertIn("음식·요리", prompt)
        self.assertIn("라면", prompt)

    def test_candidate_topic_rendered(self) -> None:
        # 각 후보의 topic_ko/subtopic_ko 가 후보 표기에 실린다.
        prompt = self._build(source_topic={"topic_ko": "음식·요리", "subtopic_ko": "라면"})
        self.assertIn("여행·관광", prompt)
        self.assertIn("타지마할", prompt)

    def test_soft_guidance_present(self) -> None:
        # soft 지시: 주제 다르면 same_domain 지양, 단 내용 실연관 시 내용 우선(하드 배제 아님).
        prompt = self._build(source_topic={"topic_ko": "음식·요리", "subtopic_ko": "라면"})
        self.assertIn("same_domain", prompt)
        self.assertIn("내용", prompt)  # 내용 우선(soft)
        # 참고 신호(하드 배제 아님)임을 명시.
        self.assertIn("참고", prompt)

    def test_source_topic_none_is_backward_compatible(self) -> None:
        # source_topic 미지정(None)이어도 기존처럼 조립된다(하위호환).
        prompt = self._build(source_topic=None)
        self.assertIsInstance(prompt, str)
        self.assertIn("소스 요약", prompt)


if __name__ == "__main__":
    unittest.main()
