"""T3-1 관계 후보 검색 단위 테스트 (asset_candidates). DB·LLM 불필요.

엣지 영속화는 단계 C에서 graph_edge 로 이관됨(graph_persist) — tests/test_graph_persist.py 참조.

[008 그룹3] run_relations 의 cross_asset 슬롯 resolve 전환(T009~T012) 단위 테스트를 함께 둔다 —
일반 팩 슬롯 경유 결과가 기존 propose_relations_for_asset 위임과 **동치**(FR-001·SC-001),
미배선 cross_asset 전략 가드(FR-002), 도메인 폴백(FR-003) 검증. DB·LLM 불필요(seam 주입).
"""
from __future__ import annotations

import unittest
import uuid
from unittest import mock

from src.relations.asset_candidates import _channels_param, find_embedding_candidates
from src.relations.llm_propose import parse_and_normalize_edges

_SRC = "018f0000-0000-7000-8000-000000000001"
_T1 = "018f0000-0000-7000-8000-000000000007"
_T2 = "018f0000-0000-7000-8000-000000000008"


def _edge(confidence: object) -> dict:
    """confidence 값만 바꿔 가며 검증할 최소 LLM 엣지 dict."""
    return {
        "target_media_item_id": _T1,
        "relation_type_code": "same_series",
        "confidence": confidence,
        "reason": "테스트",
    }


class TestConfidenceClamp(unittest.TestCase):
    """T004 [US4, FR-010, #2] — confidence 를 [0,1] 로 클램프하고, 비정상값은 결정적 0.0."""

    def _conf(self, raw: object) -> float:
        out = parse_and_normalize_edges({"edges": [_edge(raw)]})
        self.assertEqual(len(out), 1)
        return out[0]["confidence"]

    def test_above_one_clamped_to_one(self) -> None:
        # 1.5 → 1.0: 자동승인 임계 판정이 1.0 을 넘지 않도록 상한 클램프.
        self.assertEqual(self._conf(1.5), 1.0)

    def test_below_zero_clamped_to_zero(self) -> None:
        # -0.3 → 0.0: 음수 confidence 를 하한 0.0 으로 클램프.
        self.assertEqual(self._conf(-0.3), 0.0)

    def test_in_range_preserved(self) -> None:
        # 정상 범위 값은 그대로 보존.
        self.assertEqual(self._conf(0.42), 0.42)

    def test_nan_falls_back_to_zero(self) -> None:
        # NaN → 0.0: 비교 불가능한 값은 결정적 기본값으로(헌법 3조).
        self.assertEqual(self._conf(float("nan")), 0.0)

    def test_unparsable_string_falls_back_to_zero(self) -> None:
        # 파싱 불가 문자열 → 0.0.
        self.assertEqual(self._conf("높음"), 0.0)

    def test_missing_falls_back_to_zero(self) -> None:
        # confidence 키 누락 → 0.0.
        edge = _edge(0.5)
        del edge["confidence"]
        out = parse_and_normalize_edges({"edges": [edge]})
        self.assertEqual(out[0]["confidence"], 0.0)

    def test_numeric_string_in_range_parsed(self) -> None:
        # 숫자 문자열도 float 로 파싱되어 보존.
        self.assertEqual(self._conf("0.8"), 0.8)


def _mock_conn(rows):
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = rows
    return conn, cur


class TestChannelsParam(unittest.TestCase):
    def test_st_clip_both(self) -> None:
        self.assertEqual(_channels_param("st"), ["st"])
        self.assertEqual(_channels_param("clip"), ["clip"])
        self.assertEqual(set(_channels_param("both")), {"st", "clip"})


class TestFindCandidates(unittest.TestCase):
    def test_maps_rows_to_str_id_candidates(self) -> None:
        rows = [
            {"id": uuid.UUID(_T1), "file_uri": "/d/a.png", "media_type": "image", "emb_score": 0.91, "summary": "요약A"},
            {"id": uuid.UUID(_T2), "file_uri": "/d/b.txt", "media_type": "txt", "emb_score": 0.42, "summary": None},
        ]
        conn, cur = _mock_conn(rows)
        out = find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5, embedding_kind="both")
        self.assertEqual([c["id"] for c in out], [_T1, _T2])
        self.assertTrue(all(isinstance(c["id"], str) for c in out))
        self.assertEqual(out[1]["summary"], "")  # None → ''
        params = cur.execute.call_args.args[1]
        self.assertEqual(params[0], _SRC)
        self.assertEqual(set(params[1]), {"st", "clip"})
        self.assertEqual(params[3], 0.0)  # min_sim 기본값
        self.assertEqual(params[4], 5)    # top_k


class TestDeterministicOrdering(unittest.TestCase):
    """T003 [US2, FR-007] — best_sim 동률 시 후보 id ASC tiebreaker 로 결정적 정렬(헌법 3조)."""

    def test_order_by_has_id_tiebreaker(self) -> None:
        # ORDER BY 가 best_sim DESC 만이면 동률 후보 순서가 비결정적 — id ASC 보조 정렬 필수.
        conn, cur = _mock_conn([])
        find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5)
        sql = cur.execute.call_args.args[0]
        # 정규화: 공백을 단일화해 줄바꿈·들여쓰기에 무관하게 부분 문자열 검사.
        norm = " ".join(sql.split())
        self.assertIn("ORDER BY p.best_sim DESC, p.id ASC", norm)

    def test_id_tiebreaker_is_after_best_sim(self) -> None:
        # best_sim 가 1순위, id 가 2순위여야 유사도 우선순위가 보존된다.
        conn, cur = _mock_conn([])
        find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5)
        norm = " ".join(cur.execute.call_args.args[0].split())
        order_clause = norm[norm.index("ORDER BY"):]
        self.assertLess(order_clause.index("best_sim"), order_clause.index("p.id"))


# ── [008 그룹3] T009~T012: run_relations cross_asset 슬롯 resolve 전환 ──────────
from src.pipeline.packs import GENERAL_PACK, MEDICAL_PACK, for_domain  # noqa: E402
from src.pipeline.registry import StrategyRegistry  # noqa: E402


class TestResolveCrossAssetSlots(unittest.TestCase):
    """T009·T010 [US1, FR-001·FR-002] — 팩의 cross_asset 슬롯을 레지스트리에서 resolve.

    슬롯 resolve 는 "의료 ER(단계 D)이 코드 분기 없이 끼워질 자리 만들기"이며, 일반 팩의
    슬롯이 모두 등록돼 있는지(미배선 가드)를 진입부에서 검증하는 chokepoint 다.
    """

    def _registry_with_defaults(self) -> StrategyRegistry:
        from src.pipeline.builtins import register_defaults
        reg = StrategyRegistry()
        register_defaults(reg)
        return reg

    def test_general_pack_slots_all_resolvable(self) -> None:
        # 일반 팩의 cross_asset 슬롯(candidates/score/persist_edges)이 전부 등록돼 있어야 한다.
        from src.app.run_relations import _resolve_cross_asset_slots
        reg = self._registry_with_defaults()
        resolved = _resolve_cross_asset_slots(GENERAL_PACK, registry=reg)
        # candidates/score/persist_edges 는 Callable 로 resolve 된다.
        for slot in ("candidates", "score", "persist_edges"):
            self.assertTrue(callable(resolved[slot]), f"슬롯 {slot} 미해결")

    def test_unwired_strategy_raises_notimplemented(self) -> None:
        # 미등록 cross_asset 전략(의료 단계 D 전 상태)을 가리키는 팩은 명시적 오류로 차단(FR-002).
        from src.app.run_relations import _resolve_cross_asset_slots
        from src.pipeline.packs import DomainPack
        reg = self._registry_with_defaults()
        unwired = DomainPack(
            name="medical",
            per_asset=dict(MEDICAL_PACK.per_asset),
            cross_asset={  # candidates 가 미등록 전략을 가리킴
                "candidates": "blocking_5keys",
                "score": "llm_propose",
                "decide": "confidence",
                "persist_edges": "graph_upsert",
            },
            policy="medical_strict",
        )
        with self.assertRaises(NotImplementedError):
            _resolve_cross_asset_slots(unwired, registry=reg)


class _FakeDB:
    """run_relations 가 _fetch_domain_label 로 호출하는 db.execute_in_transaction 만 흉내낸다.

    실제 트랜잭션 대신, domain_label 조회용 콜백에 자산별 라벨을 돌려주는 가짜 커서를 주입한다.
    """

    def __init__(self, labels: dict[str, str]) -> None:
        self._labels = labels
        self._current: str | None = None

    def execute_in_transaction(self, fn, *, idempotent: bool = True):
        # _fetch_domain_label 의 _run(conn) 만 사용 — domain_label SELECT 1행을 반환하는 conn mock.
        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value

        def _execute(sql, params):
            aid = params[0]
            label = self._labels.get(aid)
            cur.fetchone.return_value = (label,) if label is not None else None

        cur.execute.side_effect = _execute
        return fn(conn)


class TestRunRelationsSlotRouting(unittest.TestCase):
    """T009·T010·T011 [US1] — run_relations 가 도메인 팩 슬롯 resolve 로 라우팅하되,
    일반 경로는 기존 propose_relations_for_asset 위임과 **동치**(FR-001)."""

    _A_GEN = "018f0000-0000-7000-8000-000000000011"
    _A_MED = "018f0000-0000-7000-8000-000000000012"
    _A_REVIEW = "018f0000-0000-7000-8000-000000000013"

    def test_general_delegates_to_propose_with_same_args(self) -> None:
        # FR-001/SC-001: 일반 자산은 기존 propose_relations_for_asset 에 동일 인자로 위임되어
        # 결과(엣지 수치)가 슬롯 미경유 기존 경로와 동일해야 한다.
        from src.app import run_relations as rr
        db = _FakeDB({self._A_GEN: "general"})
        with mock.patch.object(
            rr, "propose_relations_for_asset", return_value=(1, 2, 3, 4)
        ) as m:
            result = rr.run_relations(
                [self._A_GEN], db=db, top_k=7, embedding_kind="st"
            )
        m.assert_called_once_with(db, self._A_GEN, top_k=7, embedding_kind="st")
        # done 에 (asset_id, edges_upserted, edges_skipped) = (aid, 3, 4) 가 그대로 실려야 한다.
        self.assertEqual(result["done"], [(self._A_GEN, 3, 4)])
        self.assertEqual(result["failed"], [])

    def test_review_label_falls_back_to_general(self) -> None:
        # FR-003: review/미지정 라벨은 일반 팩으로 보수적 폴백 → propose 위임 경로를 탄다.
        from src.app import run_relations as rr
        db = _FakeDB({self._A_REVIEW: "review"})
        with mock.patch.object(
            rr, "propose_relations_for_asset", return_value=(0, 0, 0, 0)
        ) as m:
            result = rr.run_relations([self._A_REVIEW], db=db)
        m.assert_called_once()
        self.assertEqual(result["failed"], [])
        self.assertEqual(len(result["done"]), 1)

    def test_unspecified_label_falls_back_to_general(self) -> None:
        # FR-003: domain_label NULL/자산 미존재 → 'general' 폴백.
        from src.app import run_relations as rr
        db = _FakeDB({})  # 라벨 없음 → fetchone None → 'general'
        with mock.patch.object(
            rr, "propose_relations_for_asset", return_value=(0, 0, 1, 0)
        ) as m:
            result = rr.run_relations([self._A_GEN], db=db)
        m.assert_called_once()
        self.assertEqual(result["done"], [(self._A_GEN, 1, 0)])

    def test_unwired_medical_isolated_as_failed_batch_continues(self) -> None:
        # FR-002: 의료 cross_asset 전략 미배선이면 그 자산만 failed 격리, 일반 자산은 계속 처리.
        from src.app import run_relations as rr
        db = _FakeDB({self._A_MED: "medical", self._A_GEN: "general"})

        # 의료 팩이 미등록 전략을 가리키도록 for_domain 을 패치(단계 D 전 미배선 시뮬레이션).
        from src.pipeline.packs import DomainPack
        unwired_medical = DomainPack(
            name="medical",
            per_asset=dict(MEDICAL_PACK.per_asset),
            cross_asset={
                "candidates": "blocking_5keys",  # 미등록
                "score": "llm_propose",
                "decide": "confidence",
                "persist_edges": "graph_upsert",
            },
            policy="medical_strict",
        )

        def _for_domain(label: str) -> DomainPack:
            return unwired_medical if label == "medical" else GENERAL_PACK

        with mock.patch.object(rr, "for_domain", side_effect=_for_domain), \
             mock.patch.object(
                 rr, "propose_relations_for_asset", return_value=(0, 0, 5, 1)
             ) as m:
            result = rr.run_relations([self._A_MED, self._A_GEN], db=db)

        # 의료는 failed 로 격리, 일반은 done. 배치는 중단되지 않는다.
        failed_ids = [aid for aid, _ in result["failed"]]
        self.assertIn(self._A_MED, failed_ids)
        self.assertEqual(result["done"], [(self._A_GEN, 5, 1)])
        # 일반 자산에 대해서만 propose 위임이 일어난다(의료는 resolve 단계에서 차단).
        m.assert_called_once_with(db, self._A_GEN, top_k=None, embedding_kind="both")


# ── [008 그룹5] T017: cross-asset end-to-end 후보 흐름 통합 가드 ─────────────
class _SingleConnDB:
    """propose_relations_for_asset 의 execute_in_transaction(_run, ...) 만 흉내내는 가짜 DB.

    실 DB·트랜잭션 없이 _run(conn) 을 그대로 실행해, 그 안의 seam(후보 조회·LLM·엣지 upsert)을
    mock 으로 가로채 union→candidate_ids 흐름을 순수 단위로 검증하기 위한 최소 더블.
    """

    def execute_in_transaction(self, fn, *, idempotent: bool = True):
        return fn(mock.MagicMock())


# get_current_settings 가 요구하는 env 17개 없이도 순수 단위로 돌리기 위한 최소 cfg 더블.
# propose_relations_for_asset 이 읽는 4개 설정만 채운다(나머지는 본 경로에서 미사용).
import types  # noqa: E402

_FAKE_CFG = types.SimpleNamespace(
    relation_top_k=10,
    relation_min_sim=0.2,
    relation_path_top_k=10,
    relation_auto_approve_min=0.75,
)


class TestCrossAssetCandidateFlowIntegration(unittest.TestCase):
    """T017 [US4, FR-011, SC-007, #10] — cross-asset 경로 **통합** 관점 회귀 가드.

    기존 단위 테스트와 **중복되지 않는** end-to-end seam 흐름만 본다:
      - TestUnionCandidates(test_path_signal.py): union_candidates 순수 함수 자체 → 본 테스트는
        그 union 결과가 **propose_relations_for_asset 안에서 sync_graph_edges 의 allowed_target_ids
        (=candidate_ids)로 실제로 흐르는지**(FR-006 환각 화이트리스트 자동 확장)를 검증.
      - TestConfidenceClamp: parse_and_normalize_edges 단위 → 중복 안 함.
      - TestRunRelationsSlotRouting: 슬롯 라우팅·폴백 → 중복 안 함.
    DB·LLM 불필요(execute_in_transaction·후보 seam·LLM seam 전부 주입/mock).
    """

    def test_path_candidate_id_flows_into_allowed_target_ids(self) -> None:
        # FR-006/SC-004: min_sim 미달이라 임베딩 후보엔 없던 경로 신호 후보(_T2)가
        # union 되어 sync_graph_edges 의 allowed_target_ids(화이트리스트)에 자동 포함돼야 한다.
        from src.relations import asset_entry as ae

        emb_rows = [
            {"id": _T1, "file_uri": "/d/a.txt", "media_type": "txt",
             "emb_score": 0.83, "summary": ""},
        ]
        path_rows = [
            {"id": _T2, "file_uri": "/d/a_summary.txt", "media_type": "txt",
             "emb_score": 0.0, "summary": ""},  # 경로 신호 전용(min_sim 미달 가정)
        ]
        captured: dict = {}

        def _fake_sync(conn, *, source_asset_id, edges, allowed_target_ids, auto_approve_min):
            captured["allowed"] = allowed_target_ids
            return 1, 0

        with mock.patch.object(ae, "get_current_settings", return_value=_FAKE_CFG), \
             mock.patch.object(ae, "_fetch_source_row",
                               return_value={"fs_path": "/d/a.txt", "modality": "txt", "summary": ""}), \
             mock.patch.object(ae, "find_embedding_candidates", return_value=emb_rows), \
             mock.patch.object(ae, "find_path_signal_candidates", return_value=path_rows), \
             mock.patch.object(ae, "fetch_active_relation_kinds", return_value=[]), \
             mock.patch.object(ae, "register_new_relation_kinds", return_value=(0, 0)), \
             mock.patch.object(ae, "sync_graph_edges", side_effect=_fake_sync), \
             mock.patch.object(ae, "record_lineage", return_value=None):
            ae.propose_relations_for_asset(
                _SingleConnDB(), _SRC, top_k=5,
                llm_fn=lambda _prompt: {"edges": []},
            )

        allowed = captured["allowed"]
        # 임베딩 후보(_T1)와 경로 신호 후보(_T2)가 모두 화이트리스트에 들어가야 환각 차단을 통과.
        self.assertIn(_T1, allowed)
        self.assertIn(_T2, allowed)  # FR-006: 경로 신호 후보 자동 확장

    def test_overlap_keeps_embedding_emb_score_in_prompt_candidates(self) -> None:
        # C-3 회귀: 같은 asset_id 가 임베딩·경로 양쪽에 있으면 임베딩 실측 emb_score 가
        # 프롬프트 후보로 흐른다(0.0 sentinel 로 덮어쓰지 않음). build_prompt 에 넘어간
        # candidates 를 가로채 검증한다.
        from src.relations import asset_entry as ae

        emb_rows = [
            {"id": _T1, "file_uri": "/d/a.txt", "media_type": "txt",
             "emb_score": 0.77, "summary": ""},
        ]
        path_rows = [
            {"id": _T1, "file_uri": "/d/a.txt", "media_type": "txt",
             "emb_score": 0.0, "summary": ""},  # 같은 _T1 (겹침)
        ]
        captured: dict = {}

        def _fake_prompt(*, source_summary, source_media_type, candidates, relation_kinds_catalog):
            captured["candidates"] = list(candidates)
            return "PROMPT"

        with mock.patch.object(ae, "get_current_settings", return_value=_FAKE_CFG), \
             mock.patch.object(ae, "_fetch_source_row",
                               return_value={"fs_path": "/d/a.txt", "modality": "txt", "summary": ""}), \
             mock.patch.object(ae, "find_embedding_candidates", return_value=emb_rows), \
             mock.patch.object(ae, "find_path_signal_candidates", return_value=path_rows), \
             mock.patch.object(ae, "fetch_active_relation_kinds", return_value=[]), \
             mock.patch.object(ae, "register_new_relation_kinds", return_value=(0, 0)), \
             mock.patch.object(ae, "build_relation_proposal_prompt", side_effect=_fake_prompt), \
             mock.patch.object(ae, "sync_graph_edges", return_value=(0, 0)), \
             mock.patch.object(ae, "record_lineage", return_value=None):
            ae.propose_relations_for_asset(
                _SingleConnDB(), _SRC, top_k=5,
                llm_fn=lambda _prompt: {"edges": []},
            )

        cands = captured["candidates"]
        # 겹친 _T1 은 1건만, emb_score 는 임베딩 실측값(0.77) 유지.
        t1 = [c for c in cands if c["id"] == _T1]
        self.assertEqual(len(t1), 1)
        self.assertEqual(t1[0]["emb_score"], 0.77)


if __name__ == "__main__":
    unittest.main()
