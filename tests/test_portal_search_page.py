"""포탈 검색 페이지(평탄화·cursor·페이지네이션) 순수 단위 테스트.

DB·네트워크·파일 IO 없이 완전 순수. 헌법 3조(결정성)를 정렬 tiebreak(asset_id)·
cursor sort_keys·round(6)으로 검증한다(같은 입력 2회 → 같은 출력).
"""
from __future__ import annotations

import math
import unittest


def _result(buckets: dict) -> dict:
    """search_service.search_hybrid 반환 형태(query/results/meta)를 흉내낸 입력."""
    return {"query": "q", "results": buckets, "meta": {}}


class TestFlattenRanked(unittest.TestCase):
    def test_buckets_flattened_and_sorted(self) -> None:
        # 여러 모달리티 버킷을 단일 리스트로 합치고 (-round(sim,6), asset_id) 로 정렬한다.
        from src.portal.search_page import flatten_ranked

        result = _result(
            {
                "text_documents": [
                    {"id": "a", "similarity": 0.50, "file_uri": "/x/a.txt", "summary": "A"},
                    {"id": "c", "similarity": 0.90, "file_uri": "/x/c.txt", "summary": "C"},
                ],
                "image": [
                    {"id": "b", "similarity": 0.90, "file_uri": "/x/b.png", "summary": "B"},
                ],
            }
        )
        ranked = flatten_ranked(result)
        # 0.90 동점은 asset_id 오름차순(b < c), 그다음 0.50(a).
        self.assertEqual([r["asset_id"] for r in ranked], ["b", "c", "a"])

    def test_item_shape_and_modality_label(self) -> None:
        # 각 항목은 asset_id/modality/similarity/summary/file_name 5키, 버킷키→모달리티 라벨 매핑.
        from src.portal.search_page import flatten_ranked

        result = _result(
            {
                "text_documents": [
                    {"id": "a", "similarity": 0.5, "file_uri": "/d/sub/report.txt", "summary": "S"}
                ],
            }
        )
        item = flatten_ranked(result)[0]
        self.assertEqual(
            set(item.keys()), {"asset_id", "modality", "similarity", "summary", "file_name"}
        )
        self.assertEqual(item["modality"], "text")  # text_documents → text
        self.assertEqual(item["file_name"], "report.txt")  # file_uri 의 basename
        self.assertEqual(item["summary"], "S")

    def test_all_modality_labels(self) -> None:
        # 4개 버킷 키가 각각 text/audio/image/video 로 매핑된다.
        from src.portal.search_page import flatten_ranked

        result = _result(
            {
                "text_documents": [{"id": "t", "similarity": 0.4, "file_uri": "t.txt", "summary": ""}],
                "audio": [{"id": "u", "similarity": 0.3, "file_uri": "u.wav", "summary": ""}],
                "image": [{"id": "i", "similarity": 0.2, "file_uri": "i.png", "summary": ""}],
                "video": [{"id": "v", "similarity": 0.1, "file_uri": "v.mp4", "summary": ""}],
            }
        )
        got = {r["asset_id"]: r["modality"] for r in flatten_ranked(result)}
        self.assertEqual(got, {"t": "text", "u": "audio", "i": "image", "v": "video"})

    def test_nan_none_similarity_becomes_zero(self) -> None:
        # NaN/None similarity 는 0.0 으로 정화돼 맨 뒤로, asset_id 오름차순.
        from src.portal.search_page import flatten_ranked

        result = _result(
            {
                "text_documents": [
                    {"id": "good", "similarity": 0.7, "file_uri": "g.txt", "summary": ""},
                    {"id": "z_nan", "similarity": float("nan"), "file_uri": "n.txt", "summary": ""},
                    {"id": "a_none", "similarity": None, "file_uri": "x.txt", "summary": ""},
                ],
            }
        )
        ranked = flatten_ranked(result)
        self.assertEqual([r["asset_id"] for r in ranked], ["good", "a_none", "z_nan"])
        # 정화된 similarity 가 유한 0.0.
        nan_item = next(r for r in ranked if r["asset_id"] == "z_nan")
        self.assertTrue(math.isfinite(nan_item["similarity"]))
        self.assertEqual(nan_item["similarity"], 0.0)

    def test_deterministic_same_input_twice(self) -> None:
        # 동일 입력 2회 → 동일 순서(결정성, 헌법 3조).
        from src.portal.search_page import flatten_ranked

        result = _result(
            {
                "text_documents": [
                    {"id": "k2", "similarity": 0.5, "file_uri": "k2.txt", "summary": ""},
                    {"id": "k1", "similarity": 0.5, "file_uri": "k1.txt", "summary": ""},
                ],
                "image": [{"id": "k3", "similarity": 0.5, "file_uri": "k3.png", "summary": ""}],
            }
        )
        first = [r["asset_id"] for r in flatten_ranked(result)]
        second = [r["asset_id"] for r in flatten_ranked(result)]
        self.assertEqual(first, second)
        self.assertEqual(first, ["k1", "k2", "k3"])  # 동점 → asset_id asc

    def test_exclude_domains_filters_only_matching_label(self) -> None:
        # domain_label 이 exclude_domains 에 들면 제외, 없거나 다른 라벨이면 포함.
        from src.portal.search_page import flatten_ranked

        result = _result(
            {
                "text_documents": [
                    {"id": "med", "similarity": 0.9, "file_uri": "m.txt", "summary": "", "domain_label": "medical"},
                    {"id": "gen", "similarity": 0.8, "file_uri": "g.txt", "summary": "", "domain_label": "general"},
                    {"id": "nolabel", "similarity": 0.7, "file_uri": "n.txt", "summary": ""},
                ],
            }
        )
        ranked = flatten_ranked(result, exclude_domains=frozenset({"medical"}))
        ids = [r["asset_id"] for r in ranked]
        self.assertNotIn("med", ids)  # 의료 배제(FR-014)
        self.assertIn("gen", ids)
        self.assertIn("nolabel", ids)  # domain_label 없으면 포함

    def test_empty_result(self) -> None:
        # 빈 결과 → 빈 리스트(경계).
        from src.portal.search_page import flatten_ranked

        self.assertEqual(flatten_ranked(_result({})), [])
        self.assertEqual(flatten_ranked({"query": "q", "meta": {}}), [])


class TestCursorCodec(unittest.TestCase):
    def test_roundtrip(self) -> None:
        # encode → decode 라운드트립으로 payload 가 보존된다.
        from src.portal.search_page import decode_cursor, encode_cursor

        payload = {"s": 0.873421, "id": "asset-xyz"}
        token = encode_cursor(payload)
        self.assertIsInstance(token, str)
        self.assertEqual(decode_cursor(token), payload)

    def test_encode_is_deterministic(self) -> None:
        # 키 순서 고정(sort_keys) → 동일 payload 2회 동일 토큰(결정성·헌법 3조).
        from src.portal.search_page import encode_cursor

        a = encode_cursor({"s": 0.5, "id": "z"})
        b = encode_cursor({"id": "z", "s": 0.5})  # 키 입력 순서 달라도
        self.assertEqual(a, b)

    def test_token_is_base64url(self) -> None:
        # 토큰은 URL-safe base64(+,/ 미포함) — 쿼리스트링에 그대로 실어도 안전.
        from src.portal.search_page import encode_cursor

        token = encode_cursor({"s": 0.123456, "id": "a/b+c"})
        self.assertNotIn("+", token)
        self.assertNotIn("/", token)

    def test_forged_token_raises(self) -> None:
        # base64 로 못 푸는 위조 토큰 → ValueError(API 400 매핑).
        from src.portal.search_page import decode_cursor

        with self.assertRaises(ValueError):
            decode_cursor("!!!not-base64!!!")

    def test_broken_json_raises(self) -> None:
        # base64 는 되지만 JSON 이 아닌 토큰 → ValueError.
        import base64

        from src.portal.search_page import decode_cursor

        bad = base64.urlsafe_b64encode(b"not json at all").decode("ascii")
        with self.assertRaises(ValueError):
            decode_cursor(bad)

    def test_missing_keys_raises(self) -> None:
        # 필수 키(s/id) 누락 payload → ValueError.
        from src.portal.search_page import decode_cursor, encode_cursor

        token = encode_cursor({"s": 0.5})  # id 없음
        with self.assertRaises(ValueError):
            decode_cursor(token)

    def test_empty_token_raises(self) -> None:
        # 빈 토큰 → ValueError(경계).
        from src.portal.search_page import decode_cursor

        with self.assertRaises(ValueError):
            decode_cursor("")


def _ranked(n: int) -> list[dict]:
    """유사도가 서로 다른(내림차순) 결정적 랭킹 n건. asset_id 는 id00.. 형식."""
    return [
        {
            "asset_id": f"id{i:02d}",
            "modality": "text",
            "similarity": round(1.0 - i * 0.01, 6),
            "summary": "",
            "file_name": f"f{i}.txt",
        }
        for i in range(n)
    ]


class TestPaginate(unittest.TestCase):
    def test_first_page_without_cursor(self) -> None:
        # cursor 없으면 상위 page_size 건 + next_cursor(뒤에 더 있으니 토큰 발급).
        from src.portal.search_page import paginate

        ranked = _ranked(5)
        page = paginate(ranked, cursor=None, page_size=2)
        self.assertEqual([r["asset_id"] for r in page["items"]], ["id00", "id01"])
        self.assertIsNotNone(page["next_cursor"])

    def test_next_page_no_overlap(self) -> None:
        # 첫 페이지 next_cursor 로 다음 페이지 → 겹침 0, 이어지는 항목.
        from src.portal.search_page import paginate

        ranked = _ranked(5)
        first = paginate(ranked, cursor=None, page_size=2)
        second = paginate(ranked, cursor=first["next_cursor"], page_size=2)
        first_ids = [r["asset_id"] for r in first["items"]]
        second_ids = [r["asset_id"] for r in second["items"]]
        self.assertEqual(second_ids, ["id02", "id03"])
        self.assertEqual(set(first_ids) & set(second_ids), set())  # 겹침 0

    def test_last_page_next_cursor_none(self) -> None:
        # 마지막 페이지면 next_cursor=None(끝 표시, US1 Acc 2).
        from src.portal.search_page import paginate

        ranked = _ranked(5)
        # 5건을 2,2,1 로 페이지 → 마지막 페이지(1건)는 next_cursor None.
        p1 = paginate(ranked, cursor=None, page_size=2)
        p2 = paginate(ranked, cursor=p1["next_cursor"], page_size=2)
        p3 = paginate(ranked, cursor=p2["next_cursor"], page_size=2)
        self.assertEqual([r["asset_id"] for r in p3["items"]], ["id04"])
        self.assertIsNone(p3["next_cursor"])

    def test_exact_fit_last_page_no_cursor(self) -> None:
        # 항목 수가 page_size 배수면 마지막 페이지에서 next_cursor=None(잔여 0).
        from src.portal.search_page import paginate

        ranked = _ranked(4)
        p1 = paginate(ranked, cursor=None, page_size=2)
        p2 = paginate(ranked, cursor=p1["next_cursor"], page_size=2)
        self.assertEqual([r["asset_id"] for r in p2["items"]], ["id02", "id03"])
        self.assertIsNone(p2["next_cursor"])

    def test_same_cursor_twice_same_page(self) -> None:
        # 동일 cursor 2회 호출 → 동일 페이지·순서·next_cursor(결정성·SC-002).
        from src.portal.search_page import paginate

        ranked = _ranked(6)
        first = paginate(ranked, cursor=None, page_size=2)
        a = paginate(ranked, cursor=first["next_cursor"], page_size=2)
        b = paginate(ranked, cursor=first["next_cursor"], page_size=2)
        self.assertEqual(a, b)

    def test_empty_ranked(self) -> None:
        # 빈 랭킹 → 빈 페이지, next_cursor None(경계).
        from src.portal.search_page import paginate

        page = paginate([], cursor=None, page_size=3)
        self.assertEqual(page["items"], [])
        self.assertIsNone(page["next_cursor"])

    def test_tie_similarity_paginates_by_asset_id(self) -> None:
        # 동점 유사도 다수 → asset_id tiebreak 로 cursor 경계가 안정적(겹침/누락 0).
        from src.portal.search_page import paginate

        ranked = [
            {"asset_id": f"id{i}", "modality": "text", "similarity": 0.5, "summary": "", "file_name": ""}
            for i in range(4)
        ]
        p1 = paginate(ranked, cursor=None, page_size=2)
        p2 = paginate(ranked, cursor=p1["next_cursor"], page_size=2)
        self.assertEqual([r["asset_id"] for r in p1["items"]], ["id0", "id1"])
        self.assertEqual([r["asset_id"] for r in p2["items"]], ["id2", "id3"])

    def test_forged_cursor_raises(self) -> None:
        # 위조 cursor 는 paginate 에서도 ValueError(API 400 매핑).
        from src.portal.search_page import paginate

        with self.assertRaises(ValueError):
            paginate(_ranked(3), cursor="!!!bad!!!", page_size=2)


if __name__ == "__main__":
    unittest.main()
