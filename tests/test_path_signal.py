"""T008 [US2, SC-002·003·004] — 경로 신호(path_signal) 후보 산출·union·allowed 확장 단위 테스트.

DB·LLM 불필요(mock conn). 검증 대상:
  - C-1 stem 정규화: 확장자 + 알려진 버전/파생 접미사를 정규식으로 1회 제거한 정규화 stem.
  - C-2 결정적 정렬·LIMIT: 근접도 DESC(정확 stem 일치 > 정규화 후 일치), asset_id ASC, LIMIT path_top_k.
  - C-3 union 우선: path-only emb_score=0.0 sentinel, 임베딩 후보와 겹치면 임베딩 후보 유지.
  - FR-005: min_sim 미달 경로 후보도 후보 집합에 포함.
  - FR-006: candidate_ids(allowed_target_ids)에 경로 신호 후보 id 자동 포함.
"""
from __future__ import annotations

import unittest
import uuid
from unittest import mock

from src.relations.asset_entry import union_candidates
from src.relations.path_signal import (
    find_path_signal_candidates,
    normalize_stem,
    proximity_rank,
    split_dir_and_name,
)

_SRC = "018f0000-0000-7000-8000-000000000001"
_T1 = "018f0000-0000-7000-8000-000000000007"
_T2 = "018f0000-0000-7000-8000-000000000008"
_T3 = "018f0000-0000-7000-8000-000000000009"


class TestNormalizeStem(unittest.TestCase):
    """C-1: 확장자 + 버전/파생 접미사 1회 제거 후 정규화 stem(결정적·로케일 비의존)."""

    def test_strips_extension(self) -> None:
        self.assertEqual(normalize_stem("report.docx"), "report")

    def test_strips_version_suffix_vN(self) -> None:
        # manual_v1 / manual_v2 → 같은 정규화 stem 'manual'.
        self.assertEqual(normalize_stem("manual_v1.pdf"), normalize_stem("manual_v2.pdf"))
        self.assertEqual(normalize_stem("manual_v2.pdf"), "manual")

    def test_strips_korean_part_suffix(self) -> None:
        # _1부 / _2부 같은 연작 접미사 제거.
        self.assertEqual(normalize_stem("강의_1부.mp4"), "강의")
        self.assertEqual(normalize_stem("강의_2부.mp4"), "강의")

    def test_strips_summary_translation_transcript(self) -> None:
        self.assertEqual(normalize_stem("report_summary.txt"), "report")
        self.assertEqual(normalize_stem("문서_요약.txt"), "문서")
        self.assertEqual(normalize_stem("문서_번역.txt"), "문서")
        self.assertEqual(normalize_stem("문서_전사.txt"), "문서")

    def test_strips_copy_suffix(self) -> None:
        self.assertEqual(normalize_stem("photo_copy.png"), "photo")
        self.assertEqual(normalize_stem("사진_사본.png"), "사진")

    def test_suffix_removed_only_once(self) -> None:
        # 접미사는 1회만 제거 — 'a_summary_copy' 는 한 번만 벗겨진다(결정적·과도 정규화 방지).
        # 'copy' 가 끝이므로 'a_summary' 가 남는다(요약 접미사는 한 번 더 벗기지 않음).
        self.assertEqual(normalize_stem("a_summary_copy.txt"), "a_summary")

    def test_no_suffix_unchanged(self) -> None:
        self.assertEqual(normalize_stem("notes.md"), "notes")

    def test_case_normalized(self) -> None:
        # 로케일 비의존 소문자 정규화 — Manual.PDF 와 manual.pdf 동일 stem.
        self.assertEqual(normalize_stem("Manual.PDF"), normalize_stem("manual.pdf"))

    def test_deterministic(self) -> None:
        # 동일 입력 2회 → 동일 출력(헌법 3조).
        self.assertEqual(normalize_stem("강의_1부.mp4"), normalize_stem("강의_1부.mp4"))


class TestSplitDirAndName(unittest.TestCase):
    def test_splits_dir_and_basename(self) -> None:
        d, name = split_dir_and_name("/data/docs/report.docx")
        self.assertEqual(d, "/data/docs")
        self.assertEqual(name, "report.docx")

    def test_root_level(self) -> None:
        d, name = split_dir_and_name("/report.docx")
        self.assertEqual(d, "/")
        self.assertEqual(name, "report.docx")


class TestProximityRank(unittest.TestCase):
    """C-2: 근접도 — 정확 raw stem 일치(2) > 정규화 후 일치(1) > 불일치(None)."""

    def test_exact_raw_stem_match_is_highest(self) -> None:
        # raw stem(접미사 제거 전, 확장자만 제거)이 같으면 정확 일치 = 최고 근접도.
        r = proximity_rank(
            src_raw="report", src_norm="report", cand_raw="report", cand_norm="report"
        )
        self.assertEqual(r, 2)

    def test_normalized_match_is_lower(self) -> None:
        # raw 는 다르지만 정규화 stem 이 같으면 근접도 1.
        r = proximity_rank(
            src_raw="manual_v1", src_norm="manual", cand_raw="manual_v2", cand_norm="manual"
        )
        self.assertEqual(r, 1)

    def test_no_match_returns_none(self) -> None:
        r = proximity_rank(
            src_raw="report", src_norm="report", cand_raw="invoice", cand_norm="invoice"
        )
        self.assertIsNone(r)


def _mock_conn_with_source_and_dir(source_row, dir_rows):
    """소스 fs_path 조회 1회 + 동일 디렉터리 자산 조회 1회를 흉내내는 mock conn.

    첫 cursor().execute → fetchone()(소스 fs_path), 두 번째 execute → fetchall()(디렉터리 자산).
    실제 구현이 cursor 를 몇 번 여는지에 무관하도록, 같은 cursor 가 순차 호출되는 형태로 둔다.
    """
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = source_row
    cur.fetchall.return_value = dir_rows
    return conn, cur


class TestFindPathSignalCandidates(unittest.TestCase):
    """find_path_signal_candidates: 동일 디렉터리 + stem 매칭 후보를 결정적 정렬·LIMIT 으로 산출."""

    def test_only_same_dir_stem_matches_included(self) -> None:
        # 소스 /data/docs/report.docx 와 동일 디렉터리 + stem 일치/근접만 후보.
        src_row = {"fs_path": "/data/docs/report.docx"}
        dir_rows = [
            # 정확 stem 일치(raw) — report.pdf
            {"asset_id": uuid.UUID(_T1), "fs_path": "/data/docs/report.pdf",
             "modality": "pdf", "summary": "요약1"},
            # 정규화 일치(파생 접미사) — report_summary.txt → 정규화 'report'
            {"asset_id": uuid.UUID(_T2), "fs_path": "/data/docs/report_summary.txt",
             "modality": "txt", "summary": "요약2"},
            # 불일치 — invoice.xlsx (제외돼야 함)
            {"asset_id": uuid.UUID(_T3), "fs_path": "/data/docs/invoice.xlsx",
             "modality": "excel", "summary": "요약3"},
        ]
        conn, cur = _mock_conn_with_source_and_dir(src_row, dir_rows)
        out = find_path_signal_candidates(conn, source_asset_id=_SRC, limit=10)
        ids = [c["id"] for c in out]
        self.assertIn(_T1, ids)
        self.assertIn(_T2, ids)
        self.assertNotIn(_T3, ids)  # stem 불일치 제외

    def test_emb_score_is_zero_sentinel(self) -> None:
        # C-3: path-only 후보는 emb_score=0.0 결정적 sentinel.
        src_row = {"fs_path": "/data/docs/report.docx"}
        dir_rows = [
            {"asset_id": uuid.UUID(_T1), "fs_path": "/data/docs/report.pdf",
             "modality": "pdf", "summary": ""},
        ]
        conn, _ = _mock_conn_with_source_and_dir(src_row, dir_rows)
        out = find_path_signal_candidates(conn, source_asset_id=_SRC, limit=10)
        self.assertEqual(out[0]["emb_score"], 0.0)

    def test_deterministic_order_proximity_then_asset_id(self) -> None:
        # C-2: 근접도 DESC(정확 stem 일치 우선), 동률은 asset_id ASC.
        src_row = {"fs_path": "/data/docs/report.docx"}
        # _T2(정규화 일치, rank1) 가 행 순서상 먼저 와도, 정확 일치(rank2)인 _T1 이 앞서야 한다.
        dir_rows = [
            {"asset_id": uuid.UUID(_T2), "fs_path": "/data/docs/report_summary.txt",
             "modality": "txt", "summary": ""},
            {"asset_id": uuid.UUID(_T1), "fs_path": "/data/docs/report.pdf",
             "modality": "pdf", "summary": ""},
        ]
        conn, _ = _mock_conn_with_source_and_dir(src_row, dir_rows)
        out = find_path_signal_candidates(conn, source_asset_id=_SRC, limit=10)
        self.assertEqual([c["id"] for c in out], [_T1, _T2])

    def test_asset_id_tiebreaker_within_same_rank(self) -> None:
        # 같은 근접도(둘 다 정확 raw 일치)면 asset_id ASC.
        src_row = {"fs_path": "/data/docs/report.docx"}
        dir_rows = [
            {"asset_id": uuid.UUID(_T2), "fs_path": "/data/docs/report.pdf",
             "modality": "pdf", "summary": ""},
            {"asset_id": uuid.UUID(_T1), "fs_path": "/data/docs/report.txt",
             "modality": "txt", "summary": ""},
        ]
        conn, _ = _mock_conn_with_source_and_dir(src_row, dir_rows)
        out = find_path_signal_candidates(conn, source_asset_id=_SRC, limit=10)
        # _T1 < _T2 (문자열) → _T1 먼저.
        self.assertEqual([c["id"] for c in out], [_T1, _T2])

    def test_limit_applied(self) -> None:
        # C-2: LIMIT path_top_k — 매칭 후보가 한도 초과면 결정적으로 상위 limit 만.
        src_row = {"fs_path": "/data/docs/report.docx"}
        dir_rows = [
            {"asset_id": uuid.UUID(_T1), "fs_path": "/data/docs/report.pdf",
             "modality": "pdf", "summary": ""},
            {"asset_id": uuid.UUID(_T2), "fs_path": "/data/docs/report.txt",
             "modality": "txt", "summary": ""},
            {"asset_id": uuid.UUID(_T3), "fs_path": "/data/docs/report.md",
             "modality": "txt", "summary": ""},
        ]
        conn, _ = _mock_conn_with_source_and_dir(src_row, dir_rows)
        out = find_path_signal_candidates(conn, source_asset_id=_SRC, limit=2)
        self.assertEqual(len(out), 2)
        # asset_id ASC 정렬이므로 가장 작은 둘.
        self.assertEqual([c["id"] for c in out], [_T1, _T2])

    def test_self_excluded(self) -> None:
        # self(소스 자산)는 후보에서 제외 — SQL where 또는 코드에서 source_asset_id 제외.
        src_row = {"fs_path": "/data/docs/report.docx"}
        dir_rows = [
            # 소스 자신과 같은 asset_id 가 디렉터리 조회에 섞여 들어와도 제외돼야 한다.
            {"asset_id": uuid.UUID(_SRC), "fs_path": "/data/docs/report.docx",
             "modality": "word", "summary": ""},
            {"asset_id": uuid.UUID(_T1), "fs_path": "/data/docs/report.pdf",
             "modality": "pdf", "summary": ""},
        ]
        conn, _ = _mock_conn_with_source_and_dir(src_row, dir_rows)
        out = find_path_signal_candidates(conn, source_asset_id=_SRC, limit=10)
        ids = [c["id"] for c in out]
        self.assertNotIn(_SRC, ids)
        self.assertIn(_T1, ids)

    def test_missing_source_returns_empty(self) -> None:
        # 소스 자산이 없으면 빈 후보(조용히).
        conn, _ = _mock_conn_with_source_and_dir(None, [])
        out = find_path_signal_candidates(conn, source_asset_id=_SRC, limit=10)
        self.assertEqual(out, [])

    def test_subdirectory_excluded(self) -> None:
        # 하위 디렉터리(/data/docs/sub)는 동일 폴더가 아니므로 제외(C-2 동일 폴더만).
        src_row = {"fs_path": "/data/docs/report.docx"}
        dir_rows = [
            {"asset_id": uuid.UUID(_T1), "fs_path": "/data/docs/sub/report.pdf",
             "modality": "pdf", "summary": ""},
        ]
        conn, _ = _mock_conn_with_source_and_dir(src_row, dir_rows)
        out = find_path_signal_candidates(conn, source_asset_id=_SRC, limit=10)
        self.assertEqual(out, [])


def _emb(cid: str, score: float, uri: str = "/d/x") -> dict:
    return {"id": cid, "file_uri": uri, "media_type": "txt", "emb_score": score, "summary": ""}


def _path(cid: str, uri: str = "/d/x") -> dict:
    return {"id": cid, "file_uri": uri, "media_type": "txt", "emb_score": 0.0, "summary": ""}


class TestUnionCandidates(unittest.TestCase):
    """T006 [FR-005] — 임베딩 후보 ∪ 경로 신호 후보를 asset_id dedup union(C-3 임베딩 우선)."""

    def test_path_only_candidate_included(self) -> None:
        # FR-005: min_sim 미달이라 임베딩 후보엔 없던 경로 신호 후보가 union 에 포함된다.
        emb = [_emb(_T1, 0.83)]
        path = [_path(_T2)]
        out = union_candidates(emb, path)
        ids = [c["id"] for c in out]
        self.assertIn(_T1, ids)
        self.assertIn(_T2, ids)

    def test_overlap_keeps_embedding_candidate(self) -> None:
        # C-3: 같은 asset_id 가 양쪽에 있으면 임베딩 후보(실측 emb_score) 유지.
        emb = [_emb(_T1, 0.77)]
        path = [_path(_T1)]  # emb_score 0.0 sentinel
        out = union_candidates(emb, path)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["emb_score"], 0.77)  # 임베딩 실측값 우선(0.0 아님)

    def test_embedding_candidates_come_first(self) -> None:
        # 결정성: 임베딩 후보(정렬됨)를 먼저, 그 뒤 경로 신호 후보(정렬됨) — 안정 순서.
        emb = [_emb(_T1, 0.9)]
        path = [_path(_T2), _path(_T3)]
        out = union_candidates(emb, path)
        self.assertEqual([c["id"] for c in out], [_T1, _T2, _T3])

    def test_empty_path_returns_embedding_only(self) -> None:
        emb = [_emb(_T1, 0.5)]
        out = union_candidates(emb, [])
        self.assertEqual([c["id"] for c in out], [_T1])

    def test_dedup_within_path_preserves_first(self) -> None:
        # 경로 신호 내부 중복 asset_id 도 dedup(첫 항목 유지, 결정적).
        emb: list = []
        path = [_path(_T2, "/d/a"), _path(_T2, "/d/b")]
        out = union_candidates(emb, path)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["file_uri"], "/d/a")


if __name__ == "__main__":
    unittest.main()
