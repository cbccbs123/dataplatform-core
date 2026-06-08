"""포탈 FastAPI 진입점(`src/app/portal_api.py`) 단위 테스트 — DB·LLM 불필요.

전략(plan 010 D-7, G4)
    FastAPI ``TestClient`` 로 라우팅·상태코드·계약·의료배제만 검증한다. 소비 서비스 함수
    (``search_hybrid``/``fetch_asset_detail``/``resolve_download_target``/
    ``collect_bundle_assets``/``build_bundle_zip``)와 DB 실행 seam(``_run_in_db``)을
    ``unittest.mock.patch`` 로 대체해 **DB·LLM·네트워크 없이** 순수 단위로 돈다.

검증 대상
    - T022: ``/health`` 200 · ``/search`` 정상(items+next_cursor+meta) · 의료 배제(FR-014)
      · 위조 cursor → 400.
    - T023: ``/assets/{id}`` 200/404 · ``/assets/{id}/download`` Range→206+Content-Range·
      원본 없음→404/410(FR-009) · ``/assets/{id}/bundle`` → application/zip · seed 게이트 404.

주의: ``TestClient(app)`` 를 ``with`` 없이 쓰면 lifespan(init_settings)이 돌지 않으므로
``.env``·DB 없이 라우팅만 검증된다(부트스트랩은 G5 실DB e2e 책임).
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.app.portal_api import app


def _passthrough_db(callback):
    """``_run_in_db`` 대역: 가짜 conn 으로 callback 을 즉시 실행한다(DB 불필요).

    실제 DB 조회 함수들은 각 테스트에서 patch 로 대체되므로, 넘기는 conn 값은 무의미하다.
    """
    return callback(object())


def _fake_search_result() -> dict:
    """``search_hybrid`` 가 돌려주는 모달리티 버킷 결과 대역.

    의료 행(domain_label='medical')을 일부러 섞어 FR-014 배제 배선을 검증할 수 있게 한다.
    유사도 내림차순 정렬 시 의료(0.95)가 1위지만 배제되어야 한다.
    """
    return {
        "query": "회식",
        "results": {
            "text_documents": [
                {"id": "a1", "similarity": 0.9, "file_uri": "/x/a1.txt", "summary": "s1"},
                {"id": "a2", "similarity": 0.8, "file_uri": "/x/a2.txt", "summary": "s2"},
                {"id": "a3", "similarity": 0.7, "file_uri": "/x/a3.txt", "summary": "s3"},
            ],
            "image": [
                {
                    "id": "med1",
                    "similarity": 0.95,
                    "file_uri": "/x/m.png",
                    "summary": "medical",
                    "domain_label": "medical",
                },
            ],
        },
        "meta": {"fusion": "alpha"},
    }


class TestHealth(unittest.TestCase):
    """``/health`` 헬스 체크."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_returns_ok(self) -> None:
        # 설정 초기화 없이도 헬스는 200 + 환경 라벨을 돌려준다.
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["env"], os.getenv("PORTAL_API_ENV", "dev"))


class TestSearch(unittest.TestCase):
    """``/search`` — cursor 페이지·의료배제·위조 cursor."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("src.app.portal_api.search_hybrid")
    def test_search_returns_page_contract(self, mock_search) -> None:
        # 정상 검색: items + next_cursor + meta 계약. page_size=2 면 상위 2건 + next_cursor 발급.
        mock_search.return_value = _fake_search_result()
        resp = self.client.get("/search", params={"q": "회식", "page_size": 2})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("items", body)
        self.assertIn("next_cursor", body)
        self.assertIn("meta", body)
        # 의료(0.95) 배제 후 a1(0.9)·a2(0.8) 두 건이 첫 페이지.
        self.assertEqual([it["asset_id"] for it in body["items"]], ["a1", "a2"])
        self.assertIsNotNone(body["next_cursor"])
        self.assertEqual(body["meta"]["query"], "회식")

    @patch("src.app.portal_api.search_hybrid")
    def test_search_excludes_medical(self, mock_search) -> None:
        # FR-014: domain_label='medical' 자산은 검색 결과에서 배제된다(큰 page_size 로 전부 확인).
        mock_search.return_value = _fake_search_result()
        resp = self.client.get("/search", params={"q": "회식", "page_size": 10})
        self.assertEqual(resp.status_code, 200)
        ids = [it["asset_id"] for it in resp.json()["items"]]
        self.assertEqual(ids, ["a1", "a2", "a3"])
        self.assertNotIn("med1", ids)

    @patch("src.app.portal_api.search_hybrid")
    def test_search_passes_exclude_medical_to_flatten(self, mock_search) -> None:
        # FR-014 배선: flatten_ranked 가 exclude_domains={'medical'} 로 호출되는지 직접 확인.
        mock_search.return_value = _fake_search_result()
        with patch("src.app.portal_api.flatten_ranked", return_value=[]) as mock_flatten:
            self.client.get("/search", params={"q": "x", "page_size": 5})
        self.assertEqual(
            mock_flatten.call_args.kwargs["exclude_domains"], frozenset({"medical"})
        )

    @patch("src.app.portal_api.search_hybrid")
    def test_search_forged_cursor_returns_400(self, mock_search) -> None:
        # 위조/깨진 cursor 는 400(Edge Case) — decode_cursor ValueError 매핑.
        mock_search.return_value = _fake_search_result()
        resp = self.client.get("/search", params={"q": "x", "cursor": "###broken###"})
        self.assertEqual(resp.status_code, 400)

    @patch("src.app.portal_api.search_hybrid")
    def test_search_next_cursor_walks_pages(self, mock_search) -> None:
        # 발급된 next_cursor 로 다음 페이지를 요청하면 겹치지 않고 마지막에 next_cursor=None.
        mock_search.return_value = _fake_search_result()
        first = self.client.get("/search", params={"q": "x", "page_size": 2}).json()
        cur = first["next_cursor"]
        self.assertIsNotNone(cur)
        second = self.client.get(
            "/search", params={"q": "x", "page_size": 2, "cursor": cur}
        ).json()
        self.assertEqual([it["asset_id"] for it in second["items"]], ["a3"])
        self.assertIsNone(second["next_cursor"])


@patch("src.app.portal_api._run_in_db", _passthrough_db)
class TestAssetDetail(unittest.TestCase):
    """``/assets/{id}`` — 상세 200 / 노출 게이트 404."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("src.app.portal_api.fetch_asset_detail")
    def test_detail_returns_200(self, mock_detail) -> None:
        detail = {
            "asset_id": "a1",
            "modality": "text",
            "domain_label": "general",
            "status": "registered",
            "core_meta": {"k": "v"},
            "ext_meta": {"summary": "요약"},
            "tags": [],
            "embedding_channels": [{"channel": "st", "chunk_count": 3}],
            "relations": [],
        }
        mock_detail.return_value = detail
        resp = self.client.get("/assets/a1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["asset_id"], "a1")
        self.assertEqual(resp.json()["embedding_channels"][0]["chunk_count"], 3)

    @patch("src.app.portal_api.fetch_asset_detail")
    def test_detail_none_returns_404(self, mock_detail) -> None:
        # 없음/비registered/의료(FR-014) → fetch_asset_detail None → 404.
        mock_detail.return_value = None
        resp = self.client.get("/assets/nope")
        self.assertEqual(resp.status_code, 404)


@patch("src.app.portal_api._run_in_db", _passthrough_db)
class TestDownload(unittest.TestCase):
    """``/assets/{id}/download`` — 전체/Range/누락/게이트."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        # 알려진 10바이트 임시 원본 — Range 바이트 무결성 검증용.
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        self.tmp.write(b"0123456789")
        self.tmp.close()
        self.addCleanup(lambda: os.path.exists(self.tmp.name) and os.unlink(self.tmp.name))

    def _target(self, fs_path: str) -> dict:
        return {
            "asset_id": "a1",
            "fs_path": fs_path,
            "fs_uri": f"file://{fs_path}",
            "file_size": 10,
            "modality": "text",
            "file_name": "sample.txt",
        }

    @patch("src.app.portal_api.resolve_download_target")
    def test_download_full_returns_200(self, mock_resolve) -> None:
        mock_resolve.return_value = self._target(self.tmp.name)
        resp = self.client.get("/assets/a1/download")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"0123456789")
        self.assertEqual(resp.headers["accept-ranges"], "bytes")
        self.assertIn("sample.txt", resp.headers["content-disposition"])

    @patch("src.app.portal_api.resolve_download_target")
    def test_download_range_returns_206(self, mock_resolve) -> None:
        # Range 부분 요청 → 206 + Content-Range + 정확한 바이트 구간(SC-004 단위 근사).
        mock_resolve.return_value = self._target(self.tmp.name)
        resp = self.client.get(
            "/assets/a1/download", headers={"Range": "bytes=2-5"}
        )
        self.assertEqual(resp.status_code, 206)
        self.assertEqual(resp.headers["content-range"], "bytes 2-5/10")
        self.assertEqual(resp.content, b"2345")
        self.assertEqual(resp.headers["accept-ranges"], "bytes")

    @patch("src.app.portal_api.resolve_download_target")
    def test_download_range_unsatisfiable_returns_416(self, mock_resolve) -> None:
        # 파일 크기 초과 범위 → parse_range_header ValueError → 416.
        mock_resolve.return_value = self._target(self.tmp.name)
        resp = self.client.get(
            "/assets/a1/download", headers={"Range": "bytes=100-200"}
        )
        self.assertEqual(resp.status_code, 416)

    @patch("src.app.portal_api.resolve_download_target")
    def test_download_missing_file_returns_404_or_410(self, mock_resolve) -> None:
        # FR-009: DB 엔 있으나 원본 파일이 사라짐 → 자산 노출 없이 404/410.
        mock_resolve.return_value = self._target("/no/such/file/at/all.txt")
        resp = self.client.get("/assets/a1/download")
        self.assertIn(resp.status_code, (404, 410))

    @patch("src.app.portal_api.resolve_download_target")
    def test_download_none_returns_404(self, mock_resolve) -> None:
        # 비registered/의료/없음 게이트 → None → 404.
        mock_resolve.return_value = None
        resp = self.client.get("/assets/x/download")
        self.assertEqual(resp.status_code, 404)


@patch("src.app.portal_api._run_in_db", _passthrough_db)
class TestBundle(unittest.TestCase):
    """``/assets/{id}/bundle`` — zip 응답 / seed 게이트 404."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("src.app.portal_api.build_bundle_zip", return_value=b"PK\x03\x04zipbytes")
    @patch("src.app.portal_api.collect_bundle_assets")
    @patch("src.app.portal_api.resolve_download_target")
    def test_bundle_returns_zip(self, mock_resolve, mock_collect, mock_zip) -> None:
        # seed 가 게이트(registered·비의료) 통과 → ego-network zip 스트리밍.
        mock_resolve.return_value = {"asset_id": "seed", "fs_path": "/x/seed.txt"}
        mock_collect.return_value = [
            {"asset_id": "seed", "fs_path": "/x/seed.txt", "file_name": "seed.txt"}
        ]
        resp = self.client.get("/assets/seed/bundle")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "application/zip")
        self.assertIn("attachment", resp.headers["content-disposition"])
        self.assertEqual(resp.content, b"PK\x03\x04zipbytes")
        mock_collect.assert_called_once()

    @patch("src.app.portal_api.collect_bundle_assets")
    @patch("src.app.portal_api.resolve_download_target")
    def test_bundle_seed_gated_returns_404(self, mock_resolve, mock_collect) -> None:
        # 의료/비registered/없는 seed → resolve None → 404, collect 미호출.
        mock_resolve.return_value = None
        resp = self.client.get("/assets/medseed/bundle")
        self.assertEqual(resp.status_code, 404)
        mock_collect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
