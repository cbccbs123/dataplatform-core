"""재색인 오케스트레이션 ``reindex_asset_topics`` 단위 테스트 (mock, DB·OS·LLM 불필요).

검증 의도 (056 G5 · FR-301~304·SC-02)
    - ``reindex_asset_topics(db, asset_ids=[A])`` 는 **A 및 A 의 active 이웃**을 대상으로
      ``update_asset_topics`` 를 호출한다(target = A ∪ 이웃·중복 제거). 한 엣지는 양끝 자산의
      active-only 주제 투영을 동시에 바꾸므로, 변경 자산과 그 이웃을 함께 재색인해야 이웃이
      증분 색인에서 stale 로 남지 않는다.
    - best-effort·격리: 한 자산의 OS 갱신이 실패해도 예외를 삼키고 ``failed`` 로 세며, 나머지
      자산은 계속 처리하고 함수는 ``{updated, failed}`` 를 **예외 없이** 반환한다(SC-003 결).
    - seam 재사용: ``fetch_active_relations_for_asset``·``project_asset_topics``·
      ``update_asset_topics``·OS 클라이언트 팩토리(``get_client``)·``get_current_settings`` 를
      **topic_reindex 모듈 위치에서** patch 해 실 DB·OS 없이 동작을 통제한다.

가짜 ``db`` 는 ``db.transaction()`` 컨텍스트가 mock conn 을 yield 하도록만 흉내낸다(PG 읽기 seam).
"""
from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_MOD = "src.search.topic_reindex"


class _FakeDB:
    """``reindex_asset_topics`` 가 PG 읽기에 쓰는 ``db.transaction()`` 만 흉내내는 최소 더블.

    ``with db.transaction() as conn:`` 이 mock conn 을 돌려주고, 그 conn 은 patch 된
    ``fetch_active_relations_for_asset``/``project_asset_topics`` 로만 소비되므로 통과용이면 충분하다.
    """

    def __init__(self, conn=None):
        self.conn = conn if conn is not None else MagicMock()
        self.tx_calls = 0

    @contextmanager
    def transaction(self):
        self.tx_calls += 1
        yield self.conn


def _settings():
    return SimpleNamespace(opensearch_url="http://localhost:9200", opensearch_index="assets")


def _topic(topic_ko="요리"):
    return {
        "topic_ko": topic_ko,
        "subtopic_ko": "제빵",
        "topic_en": "cooking",
        "subtopic_en": "baking",
        "weight": 1,
    }


def _updated_asset_ids(mock_update) -> set[str]:
    """update_asset_topics(client, index, asset_id, topics) 호출들에서 asset_id 만 뽑는다."""
    out: set[str] = set()
    for c in mock_update.call_args_list:
        # asset_id 는 3번째 위치 인자(또는 kwarg). 어느 쪽이든 안전하게 집는다.
        if len(c.args) >= 3:
            out.add(str(c.args[2]))
        elif "asset_id" in c.kwargs:
            out.add(str(c.kwargs["asset_id"]))
    return out


class TestReindexAssetTopics(unittest.TestCase):
    """T501 — target 수집(자산 ∪ active 이웃·dedup) · 부분 업데이트 · best-effort 격리."""

    @patch(f"{_MOD}.get_current_settings")
    @patch(f"{_MOD}.get_client")
    @patch(f"{_MOD}.update_asset_topics")
    @patch(f"{_MOD}.project_asset_topics")
    @patch(f"{_MOD}.fetch_active_relations_for_asset")
    def test_reindexes_asset_and_active_neighbors(
        self, m_fetch, m_project, m_update, m_client, m_settings
    ) -> None:
        from src.search.topic_reindex import reindex_asset_topics

        m_settings.return_value = _settings()
        m_client.return_value = MagicMock()
        # A 의 active 이웃 = [B]; 이웃 B 는 1홉만 수집(B 의 이웃은 조회하지 않음).
        m_fetch.side_effect = lambda conn, *, asset_id, status="active": (
            [{"asset_id": "B"}] if asset_id == "A" else []
        )
        m_project.side_effect = lambda conn, *, asset_id: [_topic()]

        stats = reindex_asset_topics(_FakeDB(), asset_ids=["A"])

        # target = A ∪ 이웃(B) → A·B 모두 부분 업데이트, 각 1회씩(dedup).
        self.assertEqual(_updated_asset_ids(m_update), {"A", "B"})
        self.assertEqual(m_update.call_count, 2)
        self.assertEqual(stats, {"updated": 2, "failed": 0})

    @patch(f"{_MOD}.get_current_settings")
    @patch(f"{_MOD}.get_client")
    @patch(f"{_MOD}.update_asset_topics")
    @patch(f"{_MOD}.project_asset_topics")
    @patch(f"{_MOD}.fetch_active_relations_for_asset")
    def test_target_deduped_when_neighbor_also_in_input(
        self, m_fetch, m_project, m_update, m_client, m_settings
    ) -> None:
        # 입력에 A·B 가 모두 있고 A 의 이웃도 B → target 은 {A,B}(중복 제거) → update 2회.
        from src.search.topic_reindex import reindex_asset_topics

        m_settings.return_value = _settings()
        m_client.return_value = MagicMock()
        m_fetch.side_effect = lambda conn, *, asset_id, status="active": (
            [{"asset_id": "B"}] if asset_id == "A" else []
        )
        m_project.side_effect = lambda conn, *, asset_id: [_topic()]

        stats = reindex_asset_topics(_FakeDB(), asset_ids=["A", "B"])

        self.assertEqual(_updated_asset_ids(m_update), {"A", "B"})
        self.assertEqual(m_update.call_count, 2)
        self.assertEqual(stats, {"updated": 2, "failed": 0})

    @patch(f"{_MOD}.get_current_settings")
    @patch(f"{_MOD}.get_client")
    @patch(f"{_MOD}.update_asset_topics")
    @patch(f"{_MOD}.project_asset_topics")
    @patch(f"{_MOD}.fetch_active_relations_for_asset")
    def test_os_failure_for_one_asset_is_swallowed_and_counted(
        self, m_fetch, m_project, m_update, m_client, m_settings
    ) -> None:
        from src.search.topic_reindex import reindex_asset_topics

        m_settings.return_value = _settings()
        m_client.return_value = MagicMock()
        m_fetch.side_effect = lambda conn, *, asset_id, status="active": (
            [{"asset_id": "B"}] if asset_id == "A" else []
        )
        m_project.side_effect = lambda conn, *, asset_id: [_topic()]

        def _update(client, index, asset_id, topics):
            if asset_id == "A":
                raise RuntimeError("OS down")

        m_update.side_effect = _update

        # 예외가 전파되면 이 호출이 raise → 테스트 실패. 격리 성공이면 정상 반환.
        stats = reindex_asset_topics(_FakeDB(), asset_ids=["A"])

        # A 는 실패(failed), B 는 정상 처리(updated). 함수는 예외 없이 집계를 반환.
        self.assertEqual(stats, {"updated": 1, "failed": 1})
        self.assertIn("B", _updated_asset_ids(m_update))

    @patch(f"{_MOD}.get_current_settings")
    @patch(f"{_MOD}.get_client")
    @patch(f"{_MOD}.update_asset_topics")
    @patch(f"{_MOD}.project_asset_topics")
    @patch(f"{_MOD}.fetch_active_relations_for_asset")
    def test_empty_asset_ids_is_noop(
        self, m_fetch, m_project, m_update, m_client, m_settings
    ) -> None:
        # 빈 입력 → 아무 조회·클라이언트 생성·업데이트도 하지 않고 {0,0}.
        from src.search.topic_reindex import reindex_asset_topics

        stats = reindex_asset_topics(_FakeDB(), asset_ids=[])

        self.assertEqual(stats, {"updated": 0, "failed": 0})
        m_fetch.assert_not_called()
        m_client.assert_not_called()
        m_update.assert_not_called()

    @patch(f"{_MOD}.get_current_settings")
    @patch(f"{_MOD}.get_client")
    @patch(f"{_MOD}.update_asset_topics")
    @patch(f"{_MOD}.project_asset_topics")
    @patch(f"{_MOD}.fetch_active_relations_for_asset")
    def test_pg_read_failure_is_swallowed(
        self, m_fetch, m_project, m_update, m_client, m_settings
    ) -> None:
        # PG 읽기 단계에서 예외가 나도(이웃 조회 실패) 예외를 전파하지 않고 집계만 반환한다.
        from src.search.topic_reindex import reindex_asset_topics

        m_fetch.side_effect = RuntimeError("PG unreachable")

        stats = reindex_asset_topics(_FakeDB(), asset_ids=["A"])

        self.assertEqual(stats["updated"], 0)
        self.assertGreaterEqual(stats["failed"], 1)
        m_update.assert_not_called()  # 읽기 실패 → OS 갱신 시도조차 안 함


if __name__ == "__main__":
    unittest.main()
