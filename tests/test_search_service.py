"""006 검색 서비스 진입점(search_service.search_hybrid) — 호출부 독립 함수의 단위 테스트.

실제 검색(``search_media_all_grouped``)은 DB·모델이 필요하므로 ``_grouped_fn`` 주입으로
서비스 계층의 모달리티 필터·응답 모양만 네트워크 없이 검증한다(실 동작은 T016 e2e).
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

from src.search.search_service import search_hybrid


def _fake_grouped(query: str, **_kw: object) -> dict[str, object]:
    return {
        "text_documents": [{"id": "t1"}],
        "audio": [{"id": "a1"}],
        "image": [{"id": "i1"}],
        "video": [{"id": "v1"}],
        "meta": {"structured": {"semantic_query": query}},
    }


class TestSearchHybridService(unittest.TestCase):
    def test_returns_all_buckets_by_default(self) -> None:
        out = search_hybrid("질의", _grouped_fn=_fake_grouped)
        self.assertEqual(
            set(out["results"].keys()),
            {"text_documents", "audio", "image", "video"},
        )
        self.assertEqual(out["query"], "질의")

    def test_filters_to_requested_modalities(self) -> None:
        out = search_hybrid("질의", modalities=["text", "image"], _grouped_fn=_fake_grouped)
        self.assertEqual(set(out["results"].keys()), {"text_documents", "image"})
        self.assertEqual(out["results"]["image"], [{"id": "i1"}])

    def test_unknown_modality_raises(self) -> None:
        with self.assertRaises(ValueError):
            search_hybrid("질의", modalities=["bogus"], _grouped_fn=_fake_grouped)

    def test_meta_passthrough(self) -> None:
        out = search_hybrid("질의", _grouped_fn=_fake_grouped)
        self.assertEqual(out["meta"]["structured"]["semantic_query"], "질의")

    def test_structured_forwarded_to_grouped(self) -> None:
        # structured 를 넘기면 그대로 grouped 검색에 전달돼 LLM 질의구조화를 건너뛸 수 있어야 한다.
        captured: dict[str, object] = {}

        def fake(query: str, **kw: object) -> dict[str, object]:
            captured.update(kw)
            return {"meta": {}}

        s = {"semantic_query": "재작성", "semantic_query_en": "rewritten"}
        search_hybrid("원질의", structured=s, _grouped_fn=fake)
        self.assertEqual(captured["structured"], s)


def _scored_grouped(query: str, **_kw: object) -> dict[str, object]:
    """각 버킷에 ``similarity`` 가 다른 두 행을 담은 grouped 결과(적합도 필터 검증용)."""
    return {
        "text_documents": [
            {"id": "t_hi", "similarity": 0.40},
            {"id": "t_lo", "similarity": 0.25},
        ],
        "image": [
            {"id": "i_hi", "similarity": 0.30},
            {"id": "i_lo", "similarity": 0.17},
        ],
        "audio": [{"id": "a1", "similarity": 0.10}],
        "video": [{"id": "v1", "similarity": 0.10}],
        "meta": {},
    }


class TestSearchHybridMinScore(unittest.TestCase):
    def test_filters_rows_below_threshold(self) -> None:
        out = search_hybrid(
            "질의", modalities=["text"], min_scores={"text": 0.3}, _grouped_fn=_scored_grouped
        )
        ids = [r["id"] for r in out["results"]["text_documents"]]
        self.assertEqual(ids, ["t_hi"])  # 0.25 행은 제외

    def test_none_and_zero_threshold_passthrough(self) -> None:
        out_none = search_hybrid("질의", modalities=["text"], _grouped_fn=_scored_grouped)
        out_zero = search_hybrid(
            "질의", modalities=["text"], min_scores={"text": 0.0}, _grouped_fn=_scored_grouped
        )
        self.assertEqual(len(out_none["results"]["text_documents"]), 2)
        self.assertEqual(len(out_zero["results"]["text_documents"]), 2)

    def test_threshold_is_independent_per_modality(self) -> None:
        # text 임계값이 image 버킷을 거르면 안 된다.
        out = search_hybrid(
            "질의",
            modalities=["text", "image"],
            min_scores={"text": 0.35},
            _grouped_fn=_scored_grouped,
        )
        self.assertEqual([r["id"] for r in out["results"]["text_documents"]], ["t_hi"])
        self.assertEqual(len(out["results"]["image"]), 2)  # image 임계값 미지정 → 그대로

    def test_all_below_yields_empty_bucket(self) -> None:
        out = search_hybrid(
            "질의", modalities=["image"], min_scores={"image": 0.99}, _grouped_fn=_scored_grouped
        )
        self.assertEqual(out["results"]["image"], [])

    def test_negative_threshold_disables_filter(self) -> None:
        # 음수 임계값(오타 등)은 0.0 과 동일하게 필터 비활성 — 조용히 전부 통과시킨다.
        out = search_hybrid(
            "질의", modalities=["text"], min_scores={"text": -0.3}, _grouped_fn=_scored_grouped
        )
        self.assertEqual(len(out["results"]["text_documents"]), 2)

    def test_nan_and_missing_similarity_treated_as_zero(self) -> None:
        def grouped(query: str, **_kw: object) -> dict[str, object]:
            return {
                "text_documents": [
                    {"id": "good", "similarity": 0.5},
                    {"id": "nan", "similarity": float("nan")},
                    {"id": "missing"},  # similarity 키 없음
                ],
                "meta": {},
            }

        out = search_hybrid(
            "질의", modalities=["text"], min_scores={"text": 0.1}, _grouped_fn=grouped
        )
        self.assertEqual([r["id"] for r in out["results"]["text_documents"]], ["good"])


class TestIncludeVisualWiring(unittest.TestCase):
    """요청 모달리티에 image/video 가 없으면 시각 2단계(CLIP)는 낭비이므로,
    search_hybrid 가 grouped 에 ``include_visual=False`` 를 넘겨 건너뛰게 한다(결과 동치·비용 절감).

    text/audio 만 요청 → include_visual=False. image·video 중 하나라도 포함하거나 전체(None) → True.
    """

    @staticmethod
    def _capturing_grouped():
        captured: dict[str, object] = {}

        def _g(query: str, **kw: object) -> dict[str, object]:
            captured.update(kw)
            return {"text_documents": [], "audio": [], "image": [], "video": [], "meta": {}}

        return _g, captured

    def test_text_only_skips_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["text"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], False)

    def test_text_audio_skips_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["text", "audio"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], False)

    def test_audio_only_skips_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["audio"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], False)

    def test_image_requested_keeps_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["image"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], True)

    def test_video_requested_keeps_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["video"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], True)

    def test_text_with_video_keeps_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["text", "video"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], True)

    def test_none_modalities_keeps_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", _grouped_fn=g)  # 전체 모달리티
        self.assertIs(cap["include_visual"], True)

    def test_video_requested_returns_video_bucket(self) -> None:
        # 동치성 역방향 봉인: video 요청 시 include_visual=True 로 채워진 video 버킷이
        # 응답에 그대로 보존된다(스킵 최적화가 요청 버킷을 누락시키지 않음).
        def grouped(query: str, **_kw: object) -> dict[str, object]:
            return {
                "text_documents": [],
                "audio": [],
                "image": [{"id": "img1", "similarity": 0.6}],
                "video": [{"id": "vid1", "similarity": 0.7}],
                "meta": {},
            }

        out = search_hybrid("질의", modalities=["text", "video"], _grouped_fn=grouped)
        self.assertEqual([r["id"] for r in out["results"]["video"]], ["vid1"])
        self.assertNotIn("image", out["results"])  # 미요청 버킷은 반환 안 함


# ──────────────────────────────────────────────────────────────────────────
# 검색 서비스 백엔드 분기(search_hybrid 의 SEARCH_BACKEND). pg(기본)=현 경로 동치(회귀 0·SC-001).
# 022: opensearch=text·audio·**image·video 모두 OS**(현 PG CLIP 대신, 021 대비 변경) — 요청 모달리티
# 전체를 한 번의 OS 호출로 검색해 4 버킷을 만든다(FR-003). PG grouped(시각 CLIP)·structure_user_query
# (LLM) 미접촉 → 멀티모달 LLM 0(FR-002·SC-004). meta={"backend":"opensearch"}. OS 미도달 명확 실패
# (FR-007·SC-006). 실제 OS 검색은 가짜 seam(_os_search_fn·_os_client_fn) 주입으로 네트워크 없이 검증한다.
# ──────────────────────────────────────────────────────────────────────────


def _recording_os(buckets: dict[str, list[dict[str, object]]]) -> tuple[object, dict[str, object]]:
    """주입용 가짜 OS 검색 seam. 호출 인자를 캡처하고 주어진 버킷을 그대로 돌려준다."""
    captured: dict[str, object] = {}

    def _os(client: object, query: str, **kw: object) -> dict[str, list[dict[str, object]]]:
        captured["client"] = client
        captured["query"] = query
        captured.update(kw)
        return buckets

    return _os, captured


class TestBackendPgRegression(unittest.TestCase):
    """(a·SC-001) backend 미지정/'pg' → 현 PG grouped 경로와 동치, OS seam 미접촉(회귀 0)."""

    def test_default_backend_uses_grouped_not_os(self) -> None:
        os_calls: list[object] = []
        client_calls: list[object] = []

        def fake_os(*a: object, **k: object) -> dict[str, list[dict[str, object]]]:
            os_calls.append((a, k))
            return {}

        def fake_client() -> object:
            client_calls.append(1)
            return None

        out = search_hybrid(
            "질의", _grouped_fn=_fake_grouped, _os_search_fn=fake_os, _os_client_fn=fake_client
        )
        # OS seam(검색·클라이언트) 미호출 — pg 기본 경로
        self.assertEqual(os_calls, [])
        self.assertEqual(client_calls, [])
        # 응답은 기존 동작과 동치(전체 버킷)
        self.assertEqual(
            set(out["results"].keys()), {"text_documents", "audio", "image", "video"}
        )
        self.assertEqual(out["results"]["text_documents"], [{"id": "t1"}])

    def test_explicit_pg_backend_uses_grouped_not_os(self) -> None:
        os_calls: list[object] = []

        def fake_os(*a: object, **k: object) -> dict[str, list[dict[str, object]]]:
            os_calls.append(1)
            return {}

        out = search_hybrid(
            "질의", backend="pg", _grouped_fn=_fake_grouped, _os_search_fn=fake_os
        )
        self.assertEqual(os_calls, [])
        self.assertEqual(out["results"]["image"], [{"id": "i1"}])


class TestBackendOpenSearchMerge(unittest.TestCase):
    """(b·FR-003·SC-005) backend='opensearch' → text·audio·image·video **모두 OS**(022, PG CLIP 대신).

    021 대비 변경: 종전엔 image·video 가 PG grouped(CLIP)에서 왔으나, 022 는 image/video 도 020
    assets 인덱스(캡션 nori + KoSimCSE 임베딩)에서 OS 하이브리드로 검색한다 → 한 번의 OS 호출이
    4 버킷을 만들고 PG grouped 는 전혀 호출되지 않는다(약화 아님 — 동작이 PG→OS 로 바뀐 것).
    """

    def test_all_modalities_come_from_os(self) -> None:
        # 요청 전체(text/audio/image/video)가 한 번의 OS 호출에서 오고, PG grouped 는 미호출.
        fake_os, os_cap = _recording_os(
            {
                "text": [{"id": "os_t", "similarity": 0.9}],
                "audio": [{"id": "os_a", "similarity": 0.8}],
                "image": [{"id": "os_i", "similarity": 0.7}],
                "video": [{"id": "os_v", "similarity": 0.6}],
            }
        )
        grouped_calls: list[object] = []

        def fake_grouped(query: str, **kw: object) -> dict[str, object]:
            grouped_calls.append(1)  # 022: OS 경로에서 PG grouped 는 절대 호출 안 됨
            return {"image": [{"id": "pg_i"}], "video": [{"id": "pg_v"}], "meta": {}}

        client_calls: list[object] = []

        def fake_client() -> str:
            client_calls.append(1)
            return "FAKE_CLIENT"

        out = search_hybrid(
            "질의",
            backend="opensearch",
            _grouped_fn=fake_grouped,
            _os_search_fn=fake_os,
            _os_client_fn=fake_client,
        )
        # 응답 키가 기존과 동일(SC-005)
        self.assertEqual(
            set(out["results"].keys()), {"text_documents", "audio", "image", "video"}
        )
        # 전 버킷이 OS 에서 옴(image·video 도 PG 가 아니라 OS, FR-003)
        self.assertEqual(out["results"]["text_documents"], [{"id": "os_t", "similarity": 0.9}])
        self.assertEqual(out["results"]["audio"], [{"id": "os_a", "similarity": 0.8}])
        self.assertEqual(out["results"]["image"], [{"id": "os_i", "similarity": 0.7}])
        self.assertEqual(out["results"]["video"], [{"id": "os_v", "similarity": 0.6}])
        # PG grouped(CLIP) 미호출(FR-002)
        self.assertEqual(grouped_calls, [])
        # OS seam 가 요청 전 모달리티로, 주입 클라이언트로 1회 호출됨
        self.assertEqual(
            set(os_cap["modalities"]), {"text", "audio", "image", "video"}  # type: ignore[arg-type]
        )
        self.assertEqual(os_cap["client"], "FAKE_CLIENT")
        self.assertEqual(client_calls, [1])

    def test_image_only_comes_from_os(self) -> None:
        # 022: image 단독 요청도 OS 에서 검색(021 의 'image→PG' 가정을 OS 동작으로 갱신).
        os_calls: list[object] = []

        def fake_os(*a: object, **k: object) -> dict[str, list[dict[str, object]]]:
            os_calls.append((a, k))
            return {"image": [{"id": "os_i"}]}

        grouped_calls: list[object] = []

        def fake_grouped(query: str, **kw: object) -> dict[str, object]:
            grouped_calls.append(1)
            return {"image": [{"id": "pg_i"}], "video": [], "meta": {}}

        out = search_hybrid(
            "질의",
            modalities=["image"],
            backend="opensearch",
            _grouped_fn=fake_grouped,
            _os_search_fn=fake_os,
            _os_client_fn=lambda: None,
        )
        self.assertEqual(len(os_calls), 1)  # OS 호출됨
        self.assertEqual(grouped_calls, [])  # PG grouped 미호출
        self.assertEqual(set(out["results"].keys()), {"image"})
        self.assertEqual(out["results"]["image"], [{"id": "os_i"}])  # OS 버킷

    def test_video_only_comes_from_os(self) -> None:
        # 022: video 단독 요청도 OS 에서 검색.
        fake_os, os_cap = _recording_os({"video": [{"id": "os_v"}]})

        grouped_calls: list[object] = []

        def fake_grouped(query: str, **kw: object) -> dict[str, object]:
            grouped_calls.append(1)
            return {"image": [], "video": [{"id": "pg_v"}], "meta": {}}

        out = search_hybrid(
            "질의",
            modalities=["video"],
            backend="opensearch",
            _grouped_fn=fake_grouped,
            _os_search_fn=fake_os,
            _os_client_fn=lambda: None,
        )
        self.assertEqual(grouped_calls, [])  # PG grouped 미호출
        self.assertEqual(set(os_cap["modalities"]), {"video"})  # type: ignore[arg-type]
        self.assertEqual(out["results"]["video"], [{"id": "os_v"}])  # OS 버킷

    def test_text_only_uses_os_not_pg(self) -> None:
        # text 단독: OS 1회 호출, PG grouped 미호출(기존 의도 유지 — image/video 무관하게 PG 미접촉).
        grouped_calls: list[object] = []

        def fake_grouped(query: str, **kw: object) -> dict[str, object]:
            grouped_calls.append(1)
            return {"image": [], "video": [], "meta": {}}

        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})
        out = search_hybrid(
            "질의",
            modalities=["text"],
            backend="opensearch",
            _grouped_fn=fake_grouped,
            _os_search_fn=fake_os,
            _os_client_fn=lambda: None,
        )
        self.assertEqual(grouped_calls, [])  # PG grouped 미호출
        self.assertEqual(set(os_cap["modalities"]), {"text"})  # type: ignore[arg-type]
        self.assertEqual(set(out["results"].keys()), {"text_documents"})
        self.assertEqual(out["results"]["text_documents"], [{"id": "os_t"}])

    def test_meta_is_backend_opensearch_with_visual(self) -> None:
        # 022: 시각(image/video) 동반 요청에도 meta 는 {"backend":"opensearch"} 단일 — PG grouped 가
        # 호출되지 않으므로 PG meta(structured 등) 통과 키는 없다(021 대비 변경).
        fake_os, _ = _recording_os(
            {"text": [{"id": "os_t"}], "audio": [], "image": [{"id": "os_i"}], "video": []}
        )

        def fake_grouped(query: str, **kw: object) -> dict[str, object]:
            return {"image": [{"id": "pg_i"}], "video": [], "meta": {"structured": {"q": 1}}}

        out = search_hybrid(
            "질의",
            backend="opensearch",
            _grouped_fn=fake_grouped,
            _os_search_fn=fake_os,
            _os_client_fn=lambda: "C",
        )
        self.assertEqual(out["meta"], {"backend": "opensearch"})
        self.assertNotIn("structured", out["meta"])  # PG meta 통과 없음(grouped 미호출)


class TestBackendOpenSearchCutoffWiring(unittest.TestCase):
    """T008(023 G3) — backend='opensearch' 경로가 cfg 의 컷오프 4종을 ``os_search_fn`` 에 전달한다.

    021 fusion_weights·index·pipeline 배선(``getattr(cfg, ...)``)과 동형으로, search_service 상단
    fallback 상수 + ``getattr`` 폴백으로 ``cutoff_enabled``/``cutoff_eps``/``cutoff_floor``/
    ``cutoff_probe_k`` 를 G1/G2 ``search_assets_os`` seam 에 넘긴다(cross-module private import 안 함).
    """

    def test_cutoff_settings_forwarded_from_cfg(self) -> None:
        import src.search.search_service as svc

        # 컷오프 값을 모듈 fallback 기본 상수와 다르게 둔 가짜 cfg — 전달이 폴백이 아니라 cfg 에서
        # 옴을 증명한다. search_backend='opensearch' 가 경로를 결정(진입점은 backend 인자 미전달).
        cfg = types.SimpleNamespace(
            search_backend="opensearch",
            search_os_cutoff_enabled=True,
            search_os_cutoff_eps=0.22,
            search_os_cutoff_floor=0.55,
            search_os_probe_k=33,
        )
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})

        def fake_grouped(query: str, **kw: object) -> dict[str, object]:
            raise AssertionError("opensearch 백엔드에서 pg grouped 가 호출되면 안 됨")

        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            svc.search_hybrid(
                "질의",
                modalities=["text"],
                _os_search_fn=fake_os,
                _os_client_fn=lambda: "C",
                _grouped_fn=fake_grouped,
            )
        # os_search_fn 가 cfg 값 그대로 받음(기본 상수 폴백 아님)
        self.assertIs(os_cap["cutoff_enabled"], True)
        self.assertEqual(os_cap["cutoff_eps"], 0.22)
        self.assertEqual(os_cap["cutoff_floor"], 0.55)
        self.assertEqual(os_cap["cutoff_probe_k"], 33)

    def test_cutoff_falls_back_to_module_defaults_when_cfg_missing(self) -> None:
        # settings 미초기화(순수 단위) → cfg=None → 모듈 fallback 상수(무게이트·기본값)로 전달.
        fake_os, os_cap = _recording_os({"text": [{"id": "os_t"}]})
        out = search_hybrid(
            "질의",
            modalities=["text"],
            backend="opensearch",  # cfg 없이 명시 라우팅
            _os_search_fn=fake_os,
            _os_client_fn=lambda: "C",
            _grouped_fn=_fake_grouped,
        )
        self.assertIs(os_cap["cutoff_enabled"], False)  # 미초기화 안전(무게이트)
        self.assertEqual(os_cap["cutoff_eps"], 0.15)  # G4 calibration 확정(모듈 fallback)
        self.assertEqual(os_cap["cutoff_floor"], 0.43)
        self.assertEqual(os_cap["cutoff_probe_k"], 50)
        self.assertEqual(out["results"]["text_documents"], [{"id": "os_t"}])

    def test_pg_default_never_touches_os_cutoff(self) -> None:
        # (SC-001) backend=pg(기본) → OS seam(컷오프 포함) 미접촉·동작 동치(회귀 0).
        os_calls: list[object] = []

        def boom_os(*a: object, **k: object) -> dict[str, list[dict[str, object]]]:
            os_calls.append(k)
            raise AssertionError("pg 기본 경로에서 OS seam 이 호출되면 안 됨")

        out = search_hybrid(
            "질의",
            _grouped_fn=_fake_grouped,
            _os_search_fn=boom_os,
            _os_client_fn=lambda: None,
        )
        self.assertEqual(os_calls, [])
        self.assertEqual(out["results"]["text_documents"], [{"id": "t1"}])


class TestBackendOsPerResultMinScore(unittest.TestCase):
    """024 G2 — backend='opensearch' 는 per-result 필터에 OS 정규화 스케일 임계
    (``cfg.search_os_min_scores``)를 쓰고, pg 는 전달 ``min_scores``(PG 코사인 스케일) 그대로.

    023 버킷 게이트는 버킷 통째 유지/비움만 하므로, 켜진 버킷 안의 노이즈 꼬리(F4)는 per-result
    임계가 거른다 — 단 PG 스케일 값(image 0.25)은 OS 정규화 점수(top≈1.0)에 무력이라 분리한다.
    """

    @staticmethod
    def _os_rows() -> dict[str, list[dict[str, object]]]:
        # OS 버킷: 관련(0.6)·노이즈 꼬리(0.4) — OS 임계 0.5 면 꼬리만 잘려야 한다.
        return {
            "image": [
                {"id": "hi", "similarity": 0.6},
                {"id": "lo", "similarity": 0.4},
            ]
        }

    def test_opensearch_uses_os_min_scores_not_passed(self) -> None:
        # (a) OS 백엔드: cfg.search_os_min_scores 적용·전달 min_scores(PG 스케일) 무시.
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(
            search_backend="opensearch",
            search_os_min_scores={"text": 0.45, "image": 0.5, "video": 0.45, "audio": 0.4},
        )
        fake_os, _cap = _recording_os(self._os_rows())
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            out = svc.search_hybrid(
                "질의",
                modalities=["image"],
                min_scores={"image": 0.25},  # PG 스케일 — OS 경로에선 무시돼야 함
                _os_search_fn=fake_os,
                _os_client_fn=lambda: "C",
                _grouped_fn=_fake_grouped,
            )
        # OS 임계 0.5 적용: 0.6 유지·0.4 제거. (전달 0.25 가 적용됐다면 둘 다 남았을 것.)
        self.assertEqual([r["id"] for r in out["results"]["image"]], ["hi"])

    def test_pg_uses_passed_min_scores_unchanged(self) -> None:
        # (b) pg 백엔드: cfg 에 search_os_min_scores 가 있어도 전달 min_scores 그대로(회귀 0).
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(
            search_backend="pg",
            search_os_min_scores={"text": 0.99},  # 적용되면 t1(기본 sim 없음=0.0)이 잘려버림
        )

        def grouped_with_sims(query: str, **_kw: object) -> dict[str, object]:
            return {"text_documents": [{"id": "t1", "similarity": 0.3}], "meta": {}}

        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            out = svc.search_hybrid(
                "질의",
                modalities=["text"],
                min_scores={"text": 0.1},  # PG 스케일 그대로 적용 — 0.3 유지
                _grouped_fn=grouped_with_sims,
            )
        self.assertEqual([r["id"] for r in out["results"]["text_documents"]], ["t1"])

    def test_opensearch_falls_back_to_passed_when_cfg_missing(self) -> None:
        # (c) settings 미초기화(cfg=None)·backend 명시 → 전달 min_scores 폴백(필터 동작 보존).
        # 폴백 값 0.7 은 OS 기본 임계(image 0.45)와 결과가 갈리는 값 — 실 cfg 가 새어들면 ['hi'],
        # 진짜 폴백이면 [](둘 다 제거)로 구분돼 폴백을 강하게 격리한다(리뷰 후속).
        fake_os, _cap = _recording_os(self._os_rows())
        out = search_hybrid(
            "질의",
            modalities=["image"],
            backend="opensearch",
            min_scores={"image": 0.7},  # 폴백으로 이 값이 적용 → 0.6·0.4 둘 다 제거
            _os_search_fn=fake_os,
            _os_client_fn=lambda: "C",
            _grouped_fn=_fake_grouped,
        )
        self.assertEqual(out["results"]["image"], [])

    def test_opensearch_empty_dict_sentinel_disables_filter(self) -> None:
        # (d) 명시 빈 dict({}) = "필터 비활성" 센티넬(no_cutoff 디버그 경로) — OS 임계로 대체하지
        # 않고 무필터로 약한 후보까지 노출한다(리뷰 후속: or 가 {} 를 삼키던 회귀 교정).
        import src.search.search_service as svc

        cfg = types.SimpleNamespace(
            search_backend="opensearch",
            search_os_min_scores={"text": 0.45, "image": 0.5, "video": 0.45, "audio": 0.4},
        )
        fake_os, _cap = _recording_os(self._os_rows())
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            out = svc.search_hybrid(
                "질의",
                modalities=["image"],
                min_scores={},  # 명시 비활성 — OS 임계(0.5)가 적용되면 'lo' 가 잘려버림
                _os_search_fn=fake_os,
                _os_client_fn=lambda: "C",
                _grouped_fn=_fake_grouped,
            )
        self.assertEqual([r["id"] for r in out["results"]["image"]], ["hi", "lo"])


class TestBackendOpenSearchNoLLM(unittest.TestCase):
    """(c·FR-002·SC-004) backend='opensearch' 전 모달리티 경로는 structure_user_query(LLM) 미호출."""

    def test_opensearch_path_does_not_call_structure_user_query(self) -> None:
        # 022: image/video 포함 전 모달리티(None)에서도 LLM 질의 구조화 0(멀티모달 LLM 0).
        import src.search.media_search as ms

        fake_os, _ = _recording_os(
            {
                "text": [{"id": "os_t"}],
                "audio": [{"id": "os_a"}],
                "image": [{"id": "os_i"}],
                "video": [{"id": "os_v"}],
            }
        )
        grouped_calls: list[object] = []

        def fake_grouped(query: str, **kw: object) -> dict[str, object]:
            grouped_calls.append(1)
            return {"image": [], "video": [], "meta": {}}

        with mock.patch.object(ms, "structure_user_query") as m:
            search_hybrid(
                "질의",
                backend="opensearch",  # modalities=None → text/audio/image/video 전체
                _os_search_fn=fake_os,
                _os_client_fn=lambda: None,
                _grouped_fn=fake_grouped,
            )
        m.assert_not_called()  # SC-004: LLM 질의 구조화 0(전 모달리티)
        self.assertEqual(grouped_calls, [])  # FR-002: PG CLIP 경로(grouped) 미접촉


class TestBackendOpenSearchUnreachable(unittest.TestCase):
    """(d·FR-007·SC-006) backend='opensearch' OS 미도달 → 예외 전파(silent pg 폴백 없음)."""

    def test_os_search_exception_propagates(self) -> None:
        def boom_os(*a: object, **k: object) -> dict[str, list[dict[str, object]]]:
            raise ConnectionError("OS 미도달")

        with self.assertRaises(ConnectionError):
            search_hybrid(
                "질의",
                modalities=["text"],
                backend="opensearch",
                _os_search_fn=boom_os,
                _os_client_fn=lambda: None,
                _grouped_fn=_fake_grouped,
            )

    def test_os_search_exception_propagates_for_visual(self) -> None:
        # 022: image/video 도 OS 경로이므로 OS 미도달 시 예외 전파(silent PG CLIP 폴백 금지).
        def boom_os(*a: object, **k: object) -> dict[str, list[dict[str, object]]]:
            raise ConnectionError("OS 미도달")

        grouped_calls: list[object] = []

        def fake_grouped(query: str, **kw: object) -> dict[str, object]:
            grouped_calls.append(1)
            return {"image": [{"id": "pg_i"}], "video": [], "meta": {}}

        with self.assertRaises(ConnectionError):
            search_hybrid(
                "질의",
                modalities=["image"],
                backend="opensearch",
                _os_search_fn=boom_os,
                _os_client_fn=lambda: None,
                _grouped_fn=fake_grouped,
            )
        self.assertEqual(grouped_calls, [])  # PG 폴백 없음

    def test_os_client_exception_propagates(self) -> None:
        def boom_client() -> object:
            raise ConnectionError("OS 클라이언트 생성 실패")

        fake_os, _ = _recording_os({"text": []})
        with self.assertRaises(ConnectionError):
            search_hybrid(
                "질의",
                modalities=["text"],
                backend="opensearch",
                _os_search_fn=fake_os,
                _os_client_fn=boom_client,
                _grouped_fn=_fake_grouped,
            )

    def test_pg_backend_unaffected_by_os_failure(self) -> None:
        # 같은 입력이라도 backend='pg' 면 OS seam 미접촉 → 정상 응답(SC-006 격리).
        def boom_os(*a: object, **k: object) -> dict[str, list[dict[str, object]]]:
            raise ConnectionError("OS 미도달")

        def boom_client() -> object:
            raise ConnectionError("OS 클라이언트 생성 실패")

        out = search_hybrid(
            "질의",
            modalities=["text"],
            backend="pg",
            _os_search_fn=boom_os,
            _os_client_fn=boom_client,
            _grouped_fn=_fake_grouped,
        )
        self.assertEqual(out["results"]["text_documents"], [{"id": "t1"}])


if __name__ == "__main__":
    unittest.main()
