"""포탈 FastAPI 진입점(`src/app/portal_api.py`) 단위 테스트 — DB·LLM 불필요.

전략(plan 010 D-7, G4)
    FastAPI ``TestClient`` 로 라우팅·상태코드·계약·의료배제만 검증한다. 소비 서비스 함수
    (``search_hybrid``/``fetch_asset_detail``/``resolve_download_target``/
    ``collect_bundle_assets``/``build_bundle_zip``)와 DB 실행 seam(``_run_in_db``)을
    ``unittest.mock.patch`` 로 대체해 **DB·LLM·네트워크 없이** 순수 단위로 돈다.

검증 대상
    - T022: ``/health`` 200 · ``/search`` 정상(query+results(모달리티별)+meta) · 버킷별 의료
      배제(FR-014) · size top-N.
    - T023: ``/assets/{id}`` 200/404 · ``/assets/{id}/download`` Range→206+Content-Range·
      원본 없음→404/410(FR-009) · ``/assets/{id}/bundle`` → application/zip · seed 게이트 404.

주의: ``TestClient(app)`` 를 ``with`` 없이 쓰면 lifespan(init_settings)이 돌지 않으므로
``.env``·DB 없이 라우팅만 검증된다(부트스트랩은 G5 실DB e2e 책임).
"""
from __future__ import annotations

import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.app.portal_api import app


def _passthrough_db(callback):
    """``_run_in_db`` 대역: 가짜 conn 으로 callback 을 즉시 실행한다(DB 불필요).

    실제 DB 조회 함수들은 각 테스트에서 patch 로 대체되므로, 넘기는 conn 값은 무의미하다.
    """
    return callback(object())


def _empty_tiers(*_args, **_kwargs):
    """registry 미조회 단위 테스트 — tier 미등록 키는 projection 통과."""
    return {}


_AUTH_DISABLED_ENV = {
    "PORTAL_AUTH_DISABLED": "1",
    "PORTAL_JWT_SECRET": "test-secret",
}


def _enable_portal_test_auth_bypass(test_case: unittest.TestCase) -> None:
    """보호 라우트 단위 테스트용 dev bypass + DB/tier mock."""
    env = patch.dict(os.environ, _AUTH_DISABLED_ENV, clear=False)
    env.start()
    test_case.addCleanup(env.stop)
    db = patch("src.app.portal_api._run_in_db", _passthrough_db)
    db.start()
    test_case.addCleanup(db.stop)


def _fake_search_result() -> dict:
    """``search_hybrid`` 가 돌려주는 모달리티 버킷 결과 대역.

    의료 행(domain_label='medical')을 image 버킷에 섞어 FR-014 버킷별 배제를 검증할 수 있게
    한다. text 는 a1>a2>a3, image 는 의료(0.95) 1건뿐이라 배제 후 빈 섹션이 된다.
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
    """``/search`` — 모달리티별 그룹 응답·버킷별 의료배제·size top-N."""

    def setUp(self) -> None:
        _enable_portal_test_auth_bypass(self)
        tiers = patch("src.app.portal_api.fetch_access_tiers", side_effect=_empty_tiers)
        tiers.start()
        self.addCleanup(tiers.stop)
        # 057-후속/065: /search 주제 패싯(FR-503)은 결과 행의 **색인 topics**(=필터 소스)로 계산하며
        # 별도 DB 주제 seam 을 호출하지 않는다(라이브 투영 미사용). 패싯 자체 검증은 test_portal_topics.
        self.client = TestClient(app)

    @patch("src.app.portal_api.search_hybrid")
    def test_search_returns_grouped_contract(self, mock_search) -> None:
        # 정상 검색: query + results(모달리티별 dict) + meta(counts). cursor/평탄 items 없음.
        mock_search.return_value = _fake_search_result()
        resp = self.client.get("/search", params={"q": "회식", "size": 10})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertNotIn("next_cursor", body)  # cursor 제거됨
        self.assertEqual(body["query"], "회식")
        # results 는 모달리티별 dict — text 섹션 안에서만 랭킹(a1>a2>a3).
        self.assertEqual([r["asset_id"] for r in body["results"]["text"]], ["a1", "a2", "a3"])
        # image 섹션은 의료 1건뿐이라 배제 후 빈 리스트(섹션 키는 존재).
        self.assertEqual(body["results"]["image"], [])
        self.assertEqual(body["meta"]["counts"], {"text": 3, "image": 0})
        self.assertEqual(body["meta"]["size"], 10)

    @patch("src.app.portal_api.search_hybrid")
    def test_search_meta_propagates_observability_when_present(self, mock_search) -> None:
        # 069 P1-4: search_hybrid meta 의 관측성 3종(os_gate·llm_verify·query_norm)을 포탈이 전파.
        r = _fake_search_result()
        r["meta"].update({
            "os_gate": {"text": {"gate_passed": True}},
            "llm_verify": {"verified": 3, "dropped": 1, "fallback": False},
            "query_norm": {"enabled": True, "method": "morph", "original": "회식 영상", "normalized": "회식"},
        })
        mock_search.return_value = r
        body = self.client.get("/search", params={"q": "회식", "size": 10}).json()
        self.assertEqual(body["meta"]["os_gate"], {"text": {"gate_passed": True}})
        self.assertEqual(body["meta"]["llm_verify"]["dropped"], 1)
        self.assertEqual(body["meta"]["query_norm"]["method"], "morph")

    @patch("src.app.portal_api.search_hybrid")
    def test_search_meta_observability_keys_absent_when_off(self, mock_search) -> None:
        # off 관례: search_hybrid meta 에 없으면 포탈 meta 에도 키 부재(빈 값 주입 금지).
        mock_search.return_value = _fake_search_result()
        body = self.client.get("/search", params={"q": "회식", "size": 10}).json()
        for k in ("os_gate", "llm_verify", "query_norm"):
            self.assertNotIn(k, body["meta"])

    @patch("src.app.portal_api.search_hybrid")
    def test_os_connection_error_returns_503(self, mock_search) -> None:
        # 069 P1-4 권고: OS 연결 실패(인프라)는 503 — 코드버그 500 과 구분(운영 알람 분리).
        from opensearchpy.exceptions import ConnectionError as OSConnectionError

        mock_search.side_effect = OSConnectionError("N/A", "conn refused", None)
        resp = self.client.get("/search", params={"q": "회식"})
        self.assertEqual(resp.status_code, 503)
        self.assertIn("OpenSearch", resp.json()["detail"])

    @patch("src.app.portal_api.search_hybrid")
    def test_search_limit_per_bucket_param_and_default(self, mock_search) -> None:
        # 후보 풀(limit_per_bucket) 요청 파라미터화: 미지정=기본 50, 지정 시 그 값이 search_hybrid 에 전달.
        mock_search.return_value = _fake_search_result()
        self.client.get("/search", params={"q": "회식", "size": 10})
        self.assertEqual(mock_search.call_args.kwargs["limit_per_bucket"], 50)  # 기본값
        self.client.get("/search", params={"q": "회식", "size": 10, "limit_per_bucket": 200})
        self.assertEqual(mock_search.call_args.kwargs["limit_per_bucket"], 200)  # 요청 지정

    @patch("src.app.portal_api.search_hybrid")
    def test_search_pool_floored_to_size(self, mock_search) -> None:
        # size 계약 보장: 요청 풀이 size 보다 얕으면 max(풀, size) 로 끌어올린다(풀<size 회귀 방지).
        mock_search.return_value = _fake_search_result()
        self.client.get("/search", params={"q": "회식", "size": 80, "limit_per_bucket": 20})
        self.assertEqual(mock_search.call_args.kwargs["limit_per_bucket"], 80)

    @patch("src.app.portal_api.search_hybrid")
    def test_search_limit_per_bucket_bounds(self, mock_search) -> None:
        # 상한(500) 초과·하한(1) 미만은 422(Query ge/le 계약).
        mock_search.return_value = _fake_search_result()
        self.assertEqual(self.client.get("/search", params={"q": "x", "limit_per_bucket": 501}).status_code, 422)
        self.assertEqual(self.client.get("/search", params={"q": "x", "limit_per_bucket": 0}).status_code, 422)

    @patch("src.app.portal_api.search_hybrid")
    def test_search_response_rows_include_topic_pairs(self, mock_search) -> None:
        # 059 FR-104: /search 응답 행에 topic_pairs(부모>자식 짝) 포함(하위호환 필드·프론트 트리용).
        # 짝 없는 행은 [] 폴백. os_hit_to_row→_shape→_project_grouped_search 경유로 전달된다.
        mock_search.return_value = {
            "query": "먹방",
            "results": {
                "text_documents": [
                    {
                        "id": "a1",
                        "similarity": 0.9,
                        "file_uri": "/x/a1.mp4",
                        "summary": "s1",
                        "topics": ["음식·요리", "IT·기술"],
                        "subtopics": ["먹방", "데이터"],
                        "topic_pairs": ["음식·요리>먹방", "IT·기술>데이터"],
                    },
                    {"id": "a2", "similarity": 0.8, "file_uri": "/x/a2.txt", "summary": "s2"},
                ],
            },
            "meta": {},
        }
        body = self.client.get("/search", params={"q": "먹방", "size": 10}).json()
        rows = body["results"]["text"]
        self.assertEqual(rows[0]["topic_pairs"], ["음식·요리>먹방", "IT·기술>데이터"])
        self.assertEqual(rows[1]["topic_pairs"], [])  # 짝 없는 행 → [] 폴백(하위호환)

    @patch("src.app.portal_api.search_hybrid")
    def test_search_excludes_medical_per_bucket(self, mock_search) -> None:
        # FR-014: domain_label='medical' 은 해당 버킷에서 배제된다(image 섹션에서 med1 사라짐).
        mock_search.return_value = _fake_search_result()
        body = self.client.get("/search", params={"q": "회식", "size": 10}).json()
        all_ids = [r["asset_id"] for rows in body["results"].values() for r in rows]
        self.assertNotIn("med1", all_ids)

    @patch("src.app.portal_api.search_hybrid")
    def test_search_passes_exclude_and_size_to_group(self, mock_search) -> None:
        # 배선: group_ranked 가 exclude_domains={'medical'} · limit_per_modality=size 로 호출되는지.
        mock_search.return_value = _fake_search_result()
        with patch("src.app.portal_api.group_ranked", return_value={}) as mock_group:
            self.client.get("/search", params={"q": "x", "size": 7})
        self.assertEqual(
            mock_group.call_args.kwargs["exclude_domains"], frozenset({"medical"})
        )
        self.assertEqual(mock_group.call_args.kwargs["limit_per_modality"], 7)

    @patch("src.app.portal_api.search_hybrid")
    def test_search_size_caps_per_modality(self, mock_search) -> None:
        # size=2 → 각 모달리티 섹션이 상위 2건으로 제한된다(섹션별 독립 top-N).
        mock_search.return_value = _fake_search_result()
        body = self.client.get("/search", params={"q": "회식", "size": 2}).json()
        self.assertEqual([r["asset_id"] for r in body["results"]["text"]], ["a1", "a2"])

    @patch("src.app.portal_api.get_current_settings")
    @patch("src.app.portal_api.search_hybrid")
    def test_search_applies_min_scores_from_settings(
        self, mock_search, mock_settings
    ) -> None:
        # ②(2026-06-09): 포탈은 settings 의 모달리티별 적합도 하한(SEARCH_MIN_SCORE_*)을
        # search_hybrid 에 전달해야 한다 — run_search/sample_search_api 와 동일하게 floor 를
        # 걸어 점수 무관 무관 결과 벽을 막는다(010 포탈의 누락 교정).
        mock_search.return_value = _fake_search_result()
        floors = {"text": 0.35, "image": 0.25, "video": 0.42, "audio": 0.35}
        mock_settings.return_value = SimpleNamespace(search_min_scores=floors)
        self.client.get("/search", params={"q": "회식", "size": 5})
        self.assertEqual(mock_search.call_args.kwargs["min_scores"], floors)

    @patch("src.app.portal_api.get_current_settings", side_effect=RuntimeError)
    @patch("src.app.portal_api.search_hybrid")
    def test_search_without_settings_falls_back_to_none(
        self, mock_search, _mock_settings
    ) -> None:
        # settings 미초기화(라우팅 단위 테스트·오설정)에서는 min_scores=None 으로 보수 폴백한다 —
        # 필터 비활성(기존 동작)이며 500 을 내지 않는다. 운영은 lifespan 이 init_settings 보장.
        mock_search.return_value = _fake_search_result()
        resp = self.client.get("/search", params={"q": "x", "size": 5})
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(mock_search.call_args.kwargs["min_scores"])

    @patch("src.app.portal_api.search_hybrid")
    def test_search_passes_mode_and_exposes_search_plan(self, mock_search) -> None:
        mock_search.return_value = {
            **_fake_search_result(),
            "meta": {
                "search_plan": {
                    "content_query": "테스트",
                    "lexical_rescue": "restricted",
                    "generic_single_term": True,
                    "mode": "auto",
                    "suggestions": ["hint"],
                },
            },
        }
        resp = self.client.get("/search", params={"q": "테스트", "mode": "auto"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_search.call_args.kwargs["search_mode"], "auto")
        plan = resp.json()["meta"]["search_plan"]
        self.assertEqual(plan["lexical_rescue"], "restricted")
        self.assertTrue(plan["generic_single_term"])

    @patch("src.app.portal_api.search_hybrid")
    def test_search_invalid_mode_400(self, mock_search) -> None:
        resp = self.client.get("/search", params={"q": "x", "mode": "invalid"})
        self.assertEqual(resp.status_code, 400)
        mock_search.assert_not_called()

    @patch("src.app.portal_api.search_hybrid")
    def test_search_passes_v1_filters(self, mock_search) -> None:
        mock_search.return_value = _fake_search_result()
        resp = self.client.get(
            "/search",
            params=[
                ("q", "회식"),
                ("file_ext", "txt"),
                ("file_ext", "pdf"),
                ("source_dataset", "wikipedia"),
                ("created_from", "2026-01-01"),
                ("created_to", "2026-06-30"),
            ],
        )
        self.assertEqual(resp.status_code, 200)
        sf = mock_search.call_args.kwargs["search_filters"]
        self.assertEqual(sf.file_exts, ("pdf", "txt"))
        self.assertEqual(sf.source_datasets, ("wikipedia",))
        meta_filters = resp.json()["meta"]["filters"]
        self.assertEqual(meta_filters["file_ext"], ["pdf", "txt"])
        self.assertEqual(meta_filters["source_dataset"], ["wikipedia"])

    @patch("src.app.portal_api.search_hybrid")
    def test_search_passes_must_include_exclude(self, mock_search) -> None:
        # 057 FR-202: 반복 쿼리 파라미터 must_include/must_exclude 를 search_hybrid 에 배선한다.
        mock_search.return_value = _fake_search_result()
        resp = self.client.get(
            "/search",
            params=[
                ("q", "충전"),
                ("must_include", "배터리"),
                ("must_include", "고속"),
                ("must_exclude", "광고"),
            ],
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_search.call_args.kwargs["must_include"], ["배터리", "고속"])
        self.assertEqual(mock_search.call_args.kwargs["must_exclude"], ["광고"])

    @patch("src.app.portal_api.search_hybrid")
    def test_search_no_lexical_filters_forwards_empty(self, mock_search) -> None:
        # 미지정이면 빈 리스트로 전달(하위호환 — OS 본문 무변경).
        mock_search.return_value = _fake_search_result()
        self.client.get("/search", params={"q": "충전", "size": 5})
        self.assertEqual(mock_search.call_args.kwargs["must_include"], [])
        self.assertEqual(mock_search.call_args.kwargs["must_exclude"], [])

    @patch("src.app.portal_api.search_hybrid")
    def test_search_invalid_date_returns_422(self, mock_search) -> None:
        resp = self.client.get(
            "/search",
            params=[("q", "테스트"), ("created_from", "not-a-date")],
        )
        self.assertEqual(resp.status_code, 422)
        mock_search.assert_not_called()


class TestAssetDetail(unittest.TestCase):
    """``/assets/{id}`` — 상세 200 / 노출 게이트 404."""

    def setUp(self) -> None:
        _enable_portal_test_auth_bypass(self)
        # 065: 자산상세는 노출 통과 시 topics·same_topic_groups 를 같은 트랜잭션에서 계산하며
        # fetch_asset_topic/find_same_topic_groups(자기주제 정본 seam)를 호출한다. object() conn 단위
        # 테스트에선 fetch_asset_detail 과 동일하게 이 seam 들을 스텁한다(보강 검증은 test_portal_topics).
        for name in ("fetch_asset_topic", "find_same_topic_groups"):
            p = patch(f"src.app.portal_api.{name}", return_value=[])
            p.start()
            self.addCleanup(p.stop)
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


class TestDownload(unittest.TestCase):
    """``/assets/{id}/download`` — 전체/Range/누락/게이트."""

    def setUp(self) -> None:
        _enable_portal_test_auth_bypass(self)
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


class TestBundle(unittest.TestCase):
    """``/assets/{id}/bundle`` — zip 응답 / seed 게이트 404."""

    def setUp(self) -> None:
        _enable_portal_test_auth_bypass(self)
        self.client = TestClient(app)

    @staticmethod
    def _mk_stream(targets):
        s = io.BytesIO(b"PK\x03\x04zipbytes")
        TestBundle._last_stream = s
        return s

    @patch("src.app.portal_api.build_bundle_zip_stream", side_effect=_mk_stream.__func__)
    @patch("src.app.portal_api.collect_bundle_assets")
    @patch("src.app.portal_api.resolve_download_target")
    def test_bundle_returns_zip(self, mock_resolve, mock_collect, mock_zip) -> None:
        # seed 가 게이트(registered·비의료) 통과 → ego-network zip 스트리밍(069 P1-2: StreamingResponse).
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
        # 리뷰 🟡2 회귀: 응답 송신 후 BackgroundTask 가 스트림을 명시 close(FD 정리 — GC 의존 금지).
        self.assertTrue(TestBundle._last_stream.closed)

    @patch("src.app.portal_api.collect_bundle_assets")
    @patch("src.app.portal_api.resolve_download_target")
    def test_bundle_seed_gated_returns_404(self, mock_resolve, mock_collect) -> None:
        # 의료/비registered/없는 seed → resolve None → 404, collect 미호출.
        mock_resolve.return_value = None
        resp = self.client.get("/assets/medseed/bundle")
        self.assertEqual(resp.status_code, 404)
        mock_collect.assert_not_called()


class TestPortalAuth(unittest.TestCase):
    """042 JWT · /me · 보호 라우트 401."""

    def setUp(self) -> None:
        from src.portal.auth.verifier import _reset_verifier_for_tests

        _reset_verifier_for_tests()
        self._env = patch.dict(
            os.environ,
            {"PORTAL_AUTH_DISABLED": "0", "PORTAL_JWT_SECRET": "test-secret"},
            clear=False,
        )
        self._env.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        from src.portal.auth.verifier import _reset_verifier_for_tests

        self._env.stop()
        _reset_verifier_for_tests()

    def test_search_without_token_returns_401(self) -> None:
        resp = self.client.get("/search", params={"q": "x"})
        self.assertEqual(resp.status_code, 401)

    def test_auth_token_disabled_when_auth_enabled(self) -> None:
        resp = self.client.post("/auth/token", json={"username": "alice"})
        self.assertEqual(resp.status_code, 404)

    def test_me_with_valid_token(self) -> None:
        from src.portal.auth.dev_issuer import issue_dev_token

        token = issue_dev_token(user_id="alice")
        me_resp = self.client.get("/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me_resp.status_code, 200)
        body = me_resp.json()
        self.assertEqual(body["user_id"], "alice")
        self.assertEqual(body["clearance"], "authorized")


class TestPortalAuthDevToken(unittest.TestCase):
    """042 dev /auth/token — auth disabled 일 때만."""

    def setUp(self) -> None:
        from src.portal.auth.verifier import _reset_verifier_for_tests

        _reset_verifier_for_tests()
        self._env = patch.dict(
            os.environ,
            {"PORTAL_AUTH_DISABLED": "1", "PORTAL_JWT_SECRET": "test-secret"},
            clear=False,
        )
        self._env.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        from src.portal.auth.verifier import _reset_verifier_for_tests

        self._env.stop()
        _reset_verifier_for_tests()

    def test_auth_token_issues_jwt(self) -> None:
        token_resp = self.client.post("/auth/token", json={"username": "alice"})
        self.assertEqual(token_resp.status_code, 200)
        token = token_resp.json()["access_token"]
        me_resp = self.client.get("/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me_resp.json()["clearance"], "authorized")


class TestPortalOpenApiSecurity(unittest.TestCase):
    """Swagger /docs — HTTPBearer Authorize 버튼(OpenAPI securitySchemes)."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_openapi_exposes_http_bearer_security(self) -> None:
        spec = self.client.get("/openapi.json").json()
        schemes = spec.get("components", {}).get("securitySchemes", {})
        self.assertIn("HTTPBearer", schemes)
        self.assertEqual(schemes["HTTPBearer"]["scheme"], "bearer")
        search = spec["paths"]["/search"]["get"]
        self.assertIn({"HTTPBearer": []}, search.get("security", []))
        params = search.get("parameters", [])
        self.assertFalse(any(p.get("name") == "authorization" for p in params))


class TestAssetThumbnail(unittest.TestCase):
    """GET /assets/{id}/thumbnail — 썸네일 게이트(의료 배제·유형·파일 부재)·응답 계약(057-후속)."""

    def setUp(self) -> None:
        _enable_portal_test_auth_bypass(self)
        self.client = TestClient(app)

    @patch("src.app.portal_api.cached_thumbnail", return_value=b"\xff\xd8\xff\xe0JPG")
    @patch("src.app.portal_api.resolve_download_target")
    def test_image_returns_jpeg(self, mock_resolve, _gen) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            mock_resolve.return_value = {
                "asset_id": "a1", "fs_path": f.name, "modality": "image", "file_name": "a.png"}
            r = self.client.get("/assets/a1/thumbnail")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/jpeg")
        self.assertEqual(r.content, b"\xff\xd8\xff\xe0JPG")
        self.assertIn("max-age", r.headers.get("cache-control", ""))

    @patch("src.app.portal_api.cached_thumbnail", return_value=b"HERO")
    @patch("src.app.portal_api.resolve_download_target")
    def test_size_query_passed_through(self, mock_resolve, mock_cached) -> None:
        # ?size=detail → cached_thumbnail(size="detail") 로 전달(상세 히어로 640).
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            mock_resolve.return_value = {"asset_id": "a1", "fs_path": f.name, "modality": "image"}
            r = self.client.get("/assets/a1/thumbnail?size=detail")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(mock_cached.call_args.kwargs.get("size"), "detail")

    @patch("src.app.portal_api.resolve_download_target", return_value=None)
    def test_medical_or_missing_returns_404(self, _resolve) -> None:
        # 의료/비registered/없음 → resolve_download_target None → 404 (의료 썸네일=PHI 원천 차단)
        self.assertEqual(self.client.get("/assets/a1/thumbnail").status_code, 404)

    @patch("src.app.portal_api.resolve_download_target")
    def test_audio_returns_404(self, mock_resolve) -> None:
        mock_resolve.return_value = {"asset_id": "a1", "fs_path": "/x/a.mp3", "modality": "audio"}
        self.assertEqual(self.client.get("/assets/a1/thumbnail").status_code, 404)

    @patch("src.app.portal_api.resolve_download_target")
    def test_missing_file_returns_410(self, mock_resolve) -> None:
        mock_resolve.return_value = {
            "asset_id": "a1", "fs_path": "/nonexistent/x.png", "modality": "image"}
        self.assertEqual(self.client.get("/assets/a1/thumbnail").status_code, 410)

    @patch("src.app.portal_api.cached_thumbnail", return_value=None)
    @patch("src.app.portal_api.resolve_download_target")
    def test_generation_failure_returns_404(self, mock_resolve, _gen) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            mock_resolve.return_value = {"asset_id": "a1", "fs_path": f.name, "modality": "image"}
            r = self.client.get("/assets/a1/thumbnail")
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
