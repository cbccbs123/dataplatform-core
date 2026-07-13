"""006/037 검색 서비스 진입점(search_service.search_hybrid) — 호출부 독립 함수의 단위 테스트.

037 OpenSearch 전용 정리: PG(media_search) 백엔드 분기가 제거돼 검색 read path 는 OS 단일 경로다.
실제 OS 검색(search_assets_os)·OS 클라이언트(get_client)는 가짜 seam(_os_search_fn·_os_client_fn)
주입으로 대체해 서비스 계층의 모달리티 필터·응답 모양·배선만 네트워크 없이 검증한다(실 동작은 e2e).
027: search_assets_os 가 (buckets, gate_meta) 튜플을 돌려주므로 가짜 seam 도 튜플을 반환한다.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

from src.config import search_constants
from src.search.search_service import search_hybrid


def _recording_os(
    buckets: dict[str, list[dict[str, object]]],
    gate_meta: dict[str, dict[str, object]] | None = None,
) -> tuple[object, dict[str, object]]:
    """주입용 가짜 OS 검색 seam(027 (buckets, gate_meta) 튜플 계약). 호출 인자를 캡처하고 반환한다."""
    captured: dict[str, object] = {}

    def _os(
        client: object, query: str, **kw: object
    ) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
        captured["client"] = client
        captured["query"] = query
        captured.update(kw)
        return buckets, (gate_meta or {})

    return _os, captured


def _all_buckets_os() -> tuple[object, dict[str, object]]:
    """text/audio/image/video 4 버킷을 채운 가짜 OS seam(전체 모달리티 응답 모양 검증용)."""
    return _recording_os(
        {
            "text": [{"id": "os_t", "similarity": 0.9}],
            "audio": [{"id": "os_a", "similarity": 0.8}],
            "image": [{"id": "os_i", "similarity": 0.7}],
            "video": [{"id": "os_v", "similarity": 0.6}],
        }
    )


class TestSearchHybridService(unittest.TestCase):
    """OS 단일 경로의 모달리티 필터·응답 모양·meta 통과(기본 동작)."""

    def test_returns_all_buckets_by_default(self) -> None:
        fake_os, _ = _all_buckets_os()
        out = search_hybrid("질의", _os_search_fn=fake_os, _os_client_fn=lambda: "C")
        self.assertEqual(
            set(out["results"].keys()),
            {"text_documents", "audio", "image", "video"},
        )
        self.assertEqual(out["query"], "질의")

    def test_filters_to_requested_modalities(self) -> None:
        fake_os, _ = _all_buckets_os()
        out = search_hybrid(
            "질의", modalities=["text", "image"], _os_search_fn=fake_os, _os_client_fn=lambda: "C"
        )
        self.assertEqual(set(out["results"].keys()), {"text_documents", "image"})
        self.assertEqual(out["results"]["image"], [{"id": "os_i", "similarity": 0.7}])

    def test_unknown_modality_raises(self) -> None:
        fake_os, _ = _all_buckets_os()
        with self.assertRaises(ValueError):
            search_hybrid(
                "질의", modalities=["bogus"], _os_search_fn=fake_os, _os_client_fn=lambda: "C"
            )

    def test_meta_is_backend_opensearch(self) -> None:
        # OS 경로 응답 meta 는 backend=opensearch + os_gate 관측키.
        fake_os, _ = _recording_os({"text": [{"id": "os_t"}]}, {"text": {"gate_passed": True}})
        out = search_hybrid(
            "질의", modalities=["text"], _os_search_fn=fake_os, _os_client_fn=lambda: "C"
        )
        self.assertEqual(out["meta"]["backend"], "opensearch")
        self.assertIn("os_gate", out["meta"])

    def test_unsupported_backend_raises(self) -> None:
        # 037: 'opensearch' 외 값(과거 'pg' 등)은 미지원 백엔드로 즉시 ValueError(fail-fast).
        fake_os, _ = _all_buckets_os()
        with self.assertRaises(ValueError):
            search_hybrid(
                "질의", backend="pg", _os_search_fn=fake_os, _os_client_fn=lambda: "C"
            )


# ──────────────────────────────────────────────────────────────────────────
# 037: text·audio·image·video **모두 OS**(020 인덱스, nori BM25 캡션·라벨 + embedding kNN + 클라이언트
# 융합). 요청 모달리티 전체를 한 번의 OS 호출로 검색해 4 버킷을 만든다(FR-003). image/video 도 020
# assets 인덱스(캡션 nori + KoSimCSE 임베딩)에서 OS 하이브리드로 검색한다(CLIP 아님). LLM 질의 구조화
# 미접촉 → 멀티모달 LLM 0(FR-002·SC-004). meta={"backend":"opensearch","os_gate":…}. OS 미도달 명확
# 실패(FR-007·SC-006).
# ──────────────────────────────────────────────────────────────────────────


class TestBackendOpenSearchBuckets(unittest.TestCase):
    """(FR-003·SC-005) text·audio·image·video **모두 OS**(한 번의 OS 호출이 4 버킷)."""

    def test_all_modalities_come_from_os(self) -> None:
        fake_os, os_cap = _all_buckets_os()
        client_calls: list[object] = []

        def fake_client() -> str:
            client_calls.append(1)
            return "FAKE_CLIENT"

        out = search_hybrid("질의", _os_search_fn=fake_os, _os_client_fn=fake_client)
        # 응답 키가 표준 4 버킷(SC-005)
        self.assertEqual(
            set(out["results"].keys()), {"text_documents", "audio", "image", "video"}
        )
        # 전 버킷이 OS 에서 옴(image·video 도 OS, FR-003)
        self.assertEqual(out["results"]["text_documents"], [{"id": "os_t", "similarity": 0.9}])
        self.assertEqual(out["results"]["audio"], [{"id": "os_a", "similarity": 0.8}])
        self.assertEqual(out["results"]["image"], [{"id": "os_i", "similarity": 0.7}])
        self.assertEqual(out["results"]["video"], [{"id": "os_v", "similarity": 0.6}])
        # OS seam 가 요청 전 모달리티로, 주입 클라이언트로 1회 호출됨
        self.assertEqual(
            set(os_cap["modalities"]), {"text", "audio", "image", "video"}  # type: ignore[arg-type]
        )
        self.assertEqual(os_cap["client"], "FAKE_CLIENT")
        self.assertEqual(client_calls, [1])

    def test_image_only_comes_from_os(self) -> None:
        fake_os, _ = _recording_os({"image": [{"id": "os_i"}]})
        out = search_hybrid(
            "질의", modalities=["image"], _os_search_fn=fake_os, _os_client_fn=lambda: None
        )
        self.assertEqual(set(out["results"].keys()), {"image"})
        self.assertEqual(out["results"]["image"], [{"id": "os_i"}])  # OS 버킷

    def test_video_only_comes_from_os(self) -> None:
        fake_os, os_cap = _recording_os({"video": [{"id": "os_v"}]})
        out = search_hybrid(
            "질의", modalities=["video"], _os_search_fn=fake_os, _os_client_fn=lambda: None
        )
        self.assertEqual(set(os_cap["modalities"]), {"video"})  # type: ignore[arg-type]
        self.assertEqual(out["results"]["video"], [{"id": "os_v"}])  # OS 버킷

    def test_text_only_uses_os(self) -> None:
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})
        out = search_hybrid(
            "질의", modalities=["text"], _os_search_fn=fake_os, _os_client_fn=lambda: None
        )
        self.assertEqual(set(os_cap["modalities"]), {"text"})  # type: ignore[arg-type]
        self.assertEqual(set(out["results"].keys()), {"text_documents"})
        self.assertEqual(out["results"]["text_documents"], [{"id": "os_t"}])

    def test_meta_is_backend_opensearch_with_visual(self) -> None:
        # 시각(image/video) 동반 요청에도 meta 는 backend=opensearch + os_gate 관측키만.
        fake_os, _ = _recording_os(
            {"text": [{"id": "os_t"}], "audio": [], "image": [{"id": "os_i"}], "video": []}
        )
        out = search_hybrid("질의", _os_search_fn=fake_os, _os_client_fn=lambda: "C")
        self.assertEqual(out["meta"].get("backend"), "opensearch")
        self.assertIn("os_gate", out["meta"])  # 027 게이트 관측성 합류(F4)


class TestBackendOsMorphQueryNormWiring(unittest.TestCase):
    """072 — query-norm 토글 on 이면 검색 직전 질의를 **nori 형태소 명사(client _analyze)**로 정규화해
    ``os_search_fn`` 에 전달한다. off·단어 질의(어절<3)는 원문 그대로·``_analyze`` 미호출."""

    def _analyze_client(self, tokens: list[dict]) -> object:
        from unittest.mock import MagicMock

        c = MagicMock()
        c.indices.analyze.return_value = {"detail": {"tokenizer": {"tokens": tokens}}}
        return c

    def test_morph_norm_applied_when_enabled(self) -> None:
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(
            search_backend="opensearch", search_os_query_norm_enabled=True, opensearch_index="assets",
        )
        client = self._analyze_client([
            {"token": "김밥", "leftPOS": "NNG(General Noun)"},
            {"token": "만들", "leftPOS": "VV(Verb)"},        # 비명사 → 제거
            {"token": "법", "leftPOS": "NNG(General Noun)"},   # 스톱워드 → 제거
            {"token": "영상", "leftPOS": "NNG(General Noun)"},  # 스톱워드 → 제거
        ])
        fake_os, cap = _recording_os({"text": [{"id": "t"}]})
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            svc.search_hybrid(
                "김밥 만드는 법 영상", modalities=["text"],
                _os_search_fn=fake_os, _os_client_fn=lambda: client,
            )
        self.assertEqual(cap["query"], "김밥")  # 명사만·스톱워드 제거된 정규화 질의가 OS seam 에 감
        client.indices.analyze.assert_called()  # nori _analyze 경유(형태소 정규화)

    def test_word_query_passthrough_no_analyze(self) -> None:
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(
            search_backend="opensearch", search_os_query_norm_enabled=True, opensearch_index="assets",
        )
        client = self._analyze_client([])
        fake_os, cap = _recording_os({"text": [{"id": "t"}]})
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            svc.search_hybrid(
                "양궁", modalities=["text"],  # 1어절 단어 질의
                _os_search_fn=fake_os, _os_client_fn=lambda: client,
            )
        self.assertEqual(cap["query"], "양궁")  # 어절<3 → 원문 그대로
        client.indices.analyze.assert_not_called()  # 단어 질의는 _analyze IO 스킵

    def test_off_passthrough_no_analyze(self) -> None:
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(
            search_backend="opensearch", search_os_query_norm_enabled=False,
        )
        client = self._analyze_client([])
        fake_os, cap = _recording_os({"text": [{"id": "t"}]})
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            svc.search_hybrid(
                "김밥 만드는 법 영상", modalities=["text"],
                _os_search_fn=fake_os, _os_client_fn=lambda: client,
            )
        self.assertEqual(cap["query"], "김밥 만드는 법 영상")  # off → 원문 바이트 동일(회귀 0·SC-002)
        client.indices.analyze.assert_not_called()


class TestBackendOpenSearchCutoffWiring(unittest.TestCase):
    """027 — OS 경로가 cfg 의 게이트·컷 임계를 ``os_search_fn`` 에 전달한다.

    search_constants 단일 출처를 ``getattr`` 폴백으로 써 ``cutoff_enabled``/``cutoff_eps``/
    ``cutoff_floor``/``result_floor``/``bm25_operator`` 를 G1/G2 ``search_assets_os`` seam 에 넘긴다
    (cross-module private import 안 함). 027: 게이트 표본 수(probe_k)·정규화 스케일 임계 4종(min_scores)은
    제거되고, per-result 컷이 코사인 스케일 단일 임계(``result_floor``)로 통합됐다.
    """

    def test_cutoff_settings_forwarded_from_cfg(self) -> None:
        import src.search.search_service as svc

        # 컷오프 값을 search_constants 기본값과 다르게 둔 가짜 cfg — 전달이 폴백이 아니라 cfg 에서 옴을 증명.
        cfg = types.SimpleNamespace(
            search_backend="opensearch",
            search_os_cutoff_enabled=True,
            search_os_cutoff_eps=0.22,
            search_os_cutoff_floor=0.55,
            search_os_result_floor=0.33,
        )
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            svc.search_hybrid(
                "질의",
                modalities=["text"],
                _os_search_fn=fake_os,
                _os_client_fn=lambda: "C",
            )
        # os_search_fn 가 cfg 값 그대로 받음(search_constants 폴백 아님)
        self.assertIs(os_cap["cutoff_enabled"], True)
        self.assertEqual(os_cap["cutoff_eps"], 0.22)
        self.assertEqual(os_cap["cutoff_floor"], 0.55)
        self.assertEqual(os_cap["result_floor"], 0.33)
        # 027: 제거된 인자는 전달되지 않는다(probe_k·정규화 min_scores·pipeline_name 소멸).
        self.assertNotIn("cutoff_probe_k", os_cap)
        self.assertNotIn("pipeline_name", os_cap)

    def test_cutoff_falls_back_to_search_constants_when_cfg_missing(self) -> None:
        # settings 미초기화(순수 단위) → cfg=None → search_constants 단일 출처 폴백으로 전달.
        # 027: 미초기화 폴백 enabled 는 운영 기본과 동일(True) — 디버그 우회는 disable_os_cutoff 가 담당.
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})
        out = search_hybrid(
            "질의",
            modalities=["text"],
            _os_search_fn=fake_os,
            _os_client_fn=lambda: "C",
        )
        self.assertIs(os_cap["cutoff_enabled"], search_constants.OS_CUTOFF_ENABLED_DEFAULT)
        self.assertEqual(os_cap["cutoff_eps"], search_constants.OS_CUTOFF_EPS_DEFAULT)
        self.assertEqual(os_cap["cutoff_floor"], search_constants.OS_CUTOFF_FLOOR_DEFAULT)
        self.assertEqual(os_cap["result_floor"], search_constants.OS_RESULT_FLOOR_DEFAULT)
        self.assertEqual(out["results"]["text_documents"], [{"id": "os_t"}])

    def test_disable_os_cutoff_forces_cutoff_disabled(self) -> None:
        # disable_os_cutoff=True(no_cutoff 디버그 우회) → cfg 가 enabled=True 라도 cutoff_enabled=False
        # 로 강제 전달(게이트·per-result 컷 모두 off → 약한 후보까지 노출).
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(
            search_backend="opensearch",
            search_os_cutoff_enabled=True,
            search_os_cutoff_eps=0.22,
            search_os_cutoff_floor=0.55,
            search_os_result_floor=0.33,
        )
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            svc.search_hybrid(
                "질의",
                modalities=["text"],
                disable_os_cutoff=True,
                _os_search_fn=fake_os,
                _os_client_fn=lambda: "C",
            )
        self.assertIs(os_cap["cutoff_enabled"], False)  # cfg True 를 우회가 덮음

    def test_default_disable_os_cutoff_false_keeps_cfg_enabled(self) -> None:
        # disable_os_cutoff 기본 False → cfg 의 enabled 가 그대로 전달(우회 미적용).
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(
            search_backend="opensearch", search_os_cutoff_enabled=True,
        )
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            svc.search_hybrid(
                "질의", modalities=["text"],
                _os_search_fn=fake_os, _os_client_fn=lambda: "C",
            )
        self.assertIs(os_cap["cutoff_enabled"], True)


class TestBackendOsPerResultCutDelegation(unittest.TestCase):
    """027 — OS 는 per-result 컷을 ``search_assets_os`` 내부(코사인 스케일 cut_rows·result_floor)에
    위임하므로, search_hybrid 호출부는 전달 ``min_scores``(PG 코사인 스케일)를 OS 경로에 **적용하지
    않는다**(스케일 불일치 방지·F1·037 PG 제거로 no-op). OS 컷은 seam 내부에서 끝난다.
    """

    @staticmethod
    def _os_rows() -> dict[str, list[dict[str, object]]]:
        # OS 버킷: 이미 search_assets_os 내부 컷을 통과한 행들(여기선 가짜 seam 이 그대로 돌려줌).
        return {
            "image": [
                {"id": "hi", "similarity": 0.6},
                {"id": "lo", "similarity": 0.4},
            ]
        }

    def test_opensearch_ignores_passed_min_scores(self) -> None:
        # 전달 min_scores(PG 스케일)가 매우 높아도 OS 버킷 행이 잘리지 않는다 —
        # 호출부가 OS 경로에 _filter_by_min_score 를 적용하지 않음(컷은 seam 내부에서 이미 끝남).
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(search_backend="opensearch")
        fake_os, _cap = _recording_os(self._os_rows())
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            out = svc.search_hybrid(
                "질의",
                modalities=["image"],
                min_scores={"image": 0.99},  # 적용됐다면 둘 다 잘렸을 값 — OS 경로는 무시
                _os_search_fn=fake_os,
                _os_client_fn=lambda: "C",
            )
        self.assertEqual([r["id"] for r in out["results"]["image"]], ["hi", "lo"])

    def test_os_gate_meta_merged_into_response_meta(self) -> None:
        # (F4 관측성) search_assets_os 가 돌려준 gate_meta 가 응답 meta["os_gate"] 로 합류한다.
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(search_backend="opensearch")
        gate = {"text": {"top": 0.7, "baseline": 0.2, "gate_passed": True, "cut_count": 1}}
        fake_os, _cap = _recording_os({"text": [{"id": "os_t", "similarity": 0.9}]}, gate)
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            out = svc.search_hybrid(
                "질의",
                modalities=["text"],
                _os_search_fn=fake_os,
                _os_client_fn=lambda: "C",
            )
        self.assertEqual(out["meta"]["backend"], "opensearch")
        self.assertEqual(out["meta"]["os_gate"], gate)


class TestBackendOsRerankWiring(unittest.TestCase):
    """029 T011: OS 경로가 cfg 의 rerank_* 4종을 ``os_search_fn`` 에 전달한다(027 cutoff 동형).

    028 에서 rerank_enabled/top_r/tau/model 배선이 추가됐다 — 029 augment 전환 후에도 그 전달이
    유지됨을 봉인한다(cfg→os seam·getattr 폴백). 두 토글 off(기본)면 rerank_enabled=False 가 전달돼
    027 경로(게이트·컷) 그대로."""

    def test_rerank_settings_forwarded_from_cfg(self) -> None:
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(
            search_backend="opensearch",
            search_os_rerank_enabled=True,
            search_os_rerank_top_r=7,
            search_os_rerank_tau=0.2,
            search_os_rerank_model="가짜-reranker",
        )
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            svc.search_hybrid(
                "질의", modalities=["text"],
                _os_search_fn=fake_os, _os_client_fn=lambda: "C",
            )
        self.assertIs(os_cap["rerank_enabled"], True)
        self.assertEqual(os_cap["rerank_top_r"], 7)
        self.assertEqual(os_cap["rerank_tau"], 0.2)
        self.assertEqual(os_cap["rerank_model"], "가짜-reranker")

    def test_rerank_falls_back_to_constants_when_cfg_missing(self) -> None:
        # settings 미초기화(cfg=None) → search_constants 단일 출처 폴백(기본 off — 027 동치).
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})
        search_hybrid(
            "질의", modalities=["text"],
            _os_search_fn=fake_os, _os_client_fn=lambda: "C",
        )
        self.assertIs(os_cap["rerank_enabled"], search_constants.OS_RERANK_ENABLED_DEFAULT)
        self.assertEqual(os_cap["rerank_top_r"], search_constants.OS_RERANK_TOP_R_DEFAULT)
        self.assertEqual(os_cap["rerank_tau"], search_constants.OS_RERANK_TAU_DEFAULT)
        self.assertEqual(os_cap["rerank_model"], search_constants.OS_RERANK_MODEL_DEFAULT)


class TestBackendOsQueryNormWiring(unittest.TestCase):
    """029 T008/T011: query-norm 토글 배선 — service 가 cfg 토글(getattr 폴백)을 읽어 검색 직전 질의를
    **service 레벨에서 1회** 명사구 정규화하고(단일 LLM 호출), 정규화된 질의를 OS seam 에 넘긴다.
    관측성(FR-007)은 top-level ``meta["query_norm"]`` 로 노출해 모달리티 키 dict 인 ``os_gate``
    (gate_meta)를 오염시키지 않는다(골든 하니스 보호). off(기본)면 원문 passthrough(바이트 동일)·
    noun_phrase_query 미호출·meta 표식 없음(027 동일 — FR-008)."""

    def test_query_norm_on_passes_normalized_query_and_exposes_meta(self) -> None:
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(search_backend="opensearch", search_os_query_norm_enabled=True)
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})
        calls: list[str] = []

        def fake_norm(q: str) -> str:
            calls.append(q)
            return "천체 관측"

        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            out = svc.search_hybrid(
                "별 보는 방법", modalities=["text"],
                _os_search_fn=fake_os, _os_client_fn=lambda: "C",
                _query_norm_fn=fake_norm,
            )
        self.assertEqual(os_cap["query"], "천체 관측")   # OS seam 이 정규화된 질의를 받음
        self.assertEqual(calls, ["별 보는 방법"])         # 정규화 1회(중복 LLM 호출 0)
        qn = out["meta"]["query_norm"]
        self.assertIs(qn["enabled"], True)
        self.assertEqual(qn["original"], "별 보는 방법")
        self.assertEqual(qn["normalized"], "천체 관측")
        self.assertNotIn("query_norm", out["meta"]["os_gate"])  # gate_meta 미오염(F4 소비자 보호)

    def test_query_norm_off_is_byte_identical_passthrough(self) -> None:
        # off(기본): 원문 그대로 OS seam 에 전달·noun_phrase_query 미호출·meta 표식 없음(027 바이트 동일).
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(search_backend="opensearch", search_os_query_norm_enabled=False)
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})

        def boom(q: str) -> str:
            raise AssertionError("off 면 query-norm seam 을 호출하지 않아야 한다")

        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            out = svc.search_hybrid(
                "별 보는 방법", modalities=["text"],
                _os_search_fn=fake_os, _os_client_fn=lambda: "C",
                _query_norm_fn=boom,
            )
        self.assertEqual(os_cap["query"], "별 보는 방법")  # 원문 passthrough(바이트 동일)
        self.assertNotIn("query_norm", out["meta"])        # off 면 meta 표식 없음(027 동일)
        self.assertNotIn("query_norm_enabled", os_cap)     # search_assets_os 에 토글 무전달(원문 동치)

    def test_query_norm_falls_back_off_when_cfg_missing(self) -> None:
        # settings 미초기화(cfg=None) → getattr 폴백 OS_QUERY_NORM_ENABLED_DEFAULT(False) → 원문 passthrough.
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})

        def boom(q: str) -> str:
            raise AssertionError("기본 off 폴백이면 query-norm 호출 0")

        out = search_hybrid(
            "별 보는 방법", modalities=["text"],
            _os_search_fn=fake_os, _os_client_fn=lambda: "C",
            _query_norm_fn=boom,
        )
        self.assertEqual(os_cap["query"], "별 보는 방법")
        self.assertNotIn("query_norm", out["meta"])


class TestBackendOsBm25OperatorWiring(unittest.TestCase):
    """025 G1: OS 경로가 cfg 의 bm25 operator 를 os_search_fn 에 전달(023 cutoff 동형)."""

    def test_operator_forwarded_from_cfg(self) -> None:
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(search_backend="opensearch", search_os_bm25_operator="and")
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t", "similarity": 0.9}]})
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            svc.search_hybrid(
                "질의", modalities=["text"],
                _os_search_fn=fake_os, _os_client_fn=lambda: "C",
            )
        self.assertEqual(os_cap["bm25_operator"], "and")

    def test_operator_falls_back_to_constants_when_cfg_missing(self) -> None:
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t", "similarity": 0.9}]})
        search_hybrid(
            "질의", modalities=["text"],
            _os_search_fn=fake_os, _os_client_fn=lambda: "C",
        )
        # 폴백 = search_constants 단일 출처(027 리뷰 후속: 기본 'and' — 운영 검증값).
        self.assertEqual(os_cap["bm25_operator"], "and")


class TestBackendOsLexicalFilterWiring(unittest.TestCase):
    """057 FR-202: search_hybrid 가 must_include/must_exclude 를 os_search_fn(OS seam)에 배선한다."""

    def test_must_include_exclude_forwarded_to_os(self) -> None:
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t", "similarity": 0.9}]})
        search_hybrid(
            "질의", modalities=["text"],
            must_include=["충전"], must_exclude=["광고"],
            _os_search_fn=fake_os, _os_client_fn=lambda: "C",
        )
        self.assertEqual(os_cap["must_include"], ["충전"])
        self.assertEqual(os_cap["must_exclude"], ["광고"])

    def test_default_lexical_filters_forwarded_empty(self) -> None:
        # 미지정(기본)이면 빈 리스트로 전달 → OS seam 에서 body 무변경(하위호환).
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t", "similarity": 0.9}]})
        search_hybrid(
            "질의", modalities=["text"],
            _os_search_fn=fake_os, _os_client_fn=lambda: "C",
        )
        self.assertEqual(os_cap["must_include"], [])
        self.assertEqual(os_cap["must_exclude"], [])


class TestBackendOpenSearchUnreachable(unittest.TestCase):
    """(FR-007·SC-006) OS 미도달 → 예외 전파(silent 폴백 없음)."""

    def test_os_search_exception_propagates(self) -> None:
        def boom_os(*a: object, **k: object) -> tuple[dict[str, list[dict[str, object]]], dict]:
            raise ConnectionError("OS 미도달")

        with self.assertRaises(ConnectionError):
            search_hybrid(
                "질의",
                modalities=["text"],
                _os_search_fn=boom_os,
                _os_client_fn=lambda: None,
            )

    def test_os_search_exception_propagates_for_visual(self) -> None:
        # image/video 도 OS 경로이므로 OS 미도달 시 예외 전파(silent 폴백 금지).
        def boom_os(*a: object, **k: object) -> tuple[dict[str, list[dict[str, object]]], dict]:
            raise ConnectionError("OS 미도달")

        with self.assertRaises(ConnectionError):
            search_hybrid(
                "질의",
                modalities=["image"],
                _os_search_fn=boom_os,
                _os_client_fn=lambda: None,
            )

    def test_os_client_exception_propagates(self) -> None:
        def boom_client() -> object:
            raise ConnectionError("OS 클라이언트 생성 실패")

        fake_os, _ = _recording_os({"text": []})
        with self.assertRaises(ConnectionError):
            search_hybrid(
                "질의",
                modalities=["text"],
                _os_search_fn=fake_os,
                _os_client_fn=boom_client,
            )


if __name__ == "__main__":
    unittest.main()
