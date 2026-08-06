"""자산 자기주제 **read seam** 코어 단위 테스트 — fetch_asset_topic·find_same_topic_groups (mock·DB 불요).

078 레포 분리: asset_topic **read**(조회) 함수는 077에서 `src/topic/asset_topic_query.py`(코어)로 이동했고,
**write(classify)**는 파이프라인 레포로 갔다. 이 두 read 함수의 단위 검증(SQL 형상·짝 매칭·already_linked
대칭·정렬/절단)은 **코어 소속**이므로 여기서 유지한다(구 `tests/test_asset_topic_classify.py` 의
TestFetchAssetTopic·TestFindSameTopicGroups 를 코어로 재편입 — classify 이관 시 유실 방지·헌법 8조 회귀 0).
"""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXTURE_PATH = os.path.join(
    _REPO_ROOT, "tests", "fixtures", "topics", "same_topic_groups_contract.json"
)
# 이 계약 스냅샷은 **실 코퍼스 asset_id** 를 담고 있어 이 레포(공개)에 두지 않는다 — 비공개 문서
# 레포가 소유하고, 측정할 때만 이 경로로 가져온다. 그래서 부재 시 실패가 아니라 **skip** 이다
# (다른 골든 테스트들도 같은 규약: `RUN_OS_E2E` 게이트 또는 파일 존재 확인).
_HAS_FIXTURE = os.path.isfile(_FIXTURE_PATH)
_FIXTURE_REASON = f"계약 fixture 없음(비공개 문서 레포 소유): {_FIXTURE_PATH}"


def _mock_conn_seq(fetchone_val=None, fetchall_val=None):
    """``conn.cursor(...)`` 컨텍스트매니저 mock — fetchone/fetchall 을 각각 통제."""
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.fetchone.return_value = fetchone_val
    cur.fetchall.return_value = fetchall_val if fetchall_val is not None else []
    conn.cursor.return_value = cur
    return conn, cur


@unittest.skipUnless(_HAS_FIXTURE, _FIXTURE_REASON)
class TestFetchAssetTopic(unittest.TestCase):
    """T204 — 정본 읽기(구 project_asset_topics 형상)·부재 []."""

    def _fixture_keys(self):
        with open(_FIXTURE_PATH, encoding="utf-8") as fh:
            fx = json.load(fh)
        return set(fx["project_asset_topics_shape"][0].keys())

    def test_row_present_returns_weight_one_shape(self) -> None:
        from src.topic.asset_topic_query import fetch_asset_topic

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
        from src.topic.asset_topic_query import fetch_asset_topic

        conn, _ = _mock_conn_seq(fetchone_val=None)
        self.assertEqual(fetch_asset_topic(conn, "missing"), [])


@unittest.skipUnless(_HAS_FIXTURE, _FIXTURE_REASON)
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
        from src.topic.asset_topic_query import find_same_topic_groups

        conn, _ = _mock_conn_seq(fetchone_val=None)  # 대상 자산 asset_topic 행 없음
        self.assertEqual(find_same_topic_groups(conn, "A"), [])

    def test_pair_match_shape_equals_contract(self) -> None:
        from src.topic.asset_topic_query import find_same_topic_groups

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
        from src.topic.asset_topic_query import find_same_topic_groups

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


# ── 068 G4: 닫힌 subtopic 조회 + LLM 선택 (T301) ──────────────────────────────
# subtopic 도 topic 처럼 부모 topic 의 **닫힌 시드 목록**에서 LLM 이 고른다(058 열린 어휘 canonicalize
# 폐기·과병합/과코스닝 차단). 아래는 조회·선택·정본 en 조회 3개 신설 함수의 단위 검증(mock·DB/LLM 불요).
_SUB_JSON_OK = '{"subtopic_ko": "국내여행·지역탐방"}'


if __name__ == "__main__":
    unittest.main()
