"""021 G1·G2 — OpenSearch 검색 쿼리 빌더·결과 매핑·검색 seam 단위 테스트.

G1(순수): OS·DB·opensearch-py 불필요. ``build_search_body``(하이브리드 쿼리 본문)와
``os_hit_to_row``(결과 행 매핑)가 020 인덱스 매핑(opensearch_sync.build_index_body)과
일치하고 media_search 버킷 행과 동형(SC-005)인지 충실히 검증한다.

G2(IO seam): ``ensure_search_pipeline``(정규화 검색 파이프라인 멱등 등록)·``search_assets_os``
(질의 1회 임베딩 → modality 별 하이브리드 검색 → 버킷)을 **가짜 OS 클라이언트 주입**으로 OS 없이
액션 조립을 검증한다(docs/테스트_가이드.md §2 seam 주입). OS 미도달은 예외 전파로 검증(FR-008).

022 G1(image/video): image·video 도 020 assets 인덱스에 한국어 VLM 캡션(nori) + KoSimCSE 캡션
임베딩(``embedding``)으로 색인돼 있어 text/audio 와 **동일 OS 하이브리드**로 검색한다(CLIP 아님).
``_MODALITY_VALUES`` 의 image·video 명시 등재(저장값=라벨)와 그 하이브리드 본문·의료배제·결정 정렬을
고정한다(spec 022 FR-001·004).

테스트 위조·약화 금지(docs/테스트_가이드.md). 실제 OS 검색 실효·결정성·의료배제는 G5(실OS e2e).
"""

from __future__ import annotations

import unittest

from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind
from src.search.opensearch_search import (
    _MODALITY_VALUES,
    build_probe_body,
    build_search_body,
    ensure_search_pipeline,
    knn_score_to_cosine,
    os_hit_to_row,
    passes_cutoff,
    probe_relevance,
    search_assets_os,
)

# 020 인덱스의 nori 텍스트 필드(BM25 multi_match 대상). 필드명 정본 = opensearch_sync.build_index_body.
# 026 T008(FR-003①): boost 차등 표기 — summary^3·keywords^2·labels^1·file_name^0.5. search_text 는
# boost 1(이제 file_name 비포함이라 안전). 파일명 노이즈가 토픽 신호를 압도하지 못하게 차등을 둔다.
_NORI_TEXT_FIELDS = {"summary^3", "keywords^2", "labels^1", "file_name^0.5", "search_text"}


def _subqueries(body: dict) -> list:
    """hybrid 쿼리의 서브쿼리 목록을 꺼낸다."""
    return body["query"]["hybrid"]["queries"]


def _find_with_clause(subqueries: list, clause_key: str):
    """주어진 절(clause_key)을 가진 서브쿼리·절을 찾는다(없으면 (None, None)).

    텍스트 서브쿼리는 ``bool.must`` 안에 절이 있고(multi_match), knn 서브쿼리는 native filter 구조라
    **top-level**(``{"knn": ...}``)이다 — 둘 다 찾는다.
    """
    for sub in subqueries:
        if clause_key in sub:  # top-level 서브쿼리(예: knn native filter)
            return sub, sub
        for clause in sub.get("bool", {}).get("must", []):
            if clause_key in clause:
                return sub, clause
    return None, None


def _sub_bool(sub: dict) -> dict:
    """서브쿼리의 modality/의료배제 bool 절 — text 는 ``bool``, knn 은 native ``filter.bool``."""
    if "knn" in sub:
        return sub["knn"]["embedding"]["filter"]["bool"]
    return sub["bool"]


class BuildSearchBodyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query = "한국어 검색 질의"
        self.vector = [0.1, 0.2, 0.3, 0.4]
        self.body = build_search_body(self.query, self.vector, modality_values=["txt"], k=50)

    def test_hybrid_with_two_subqueries(self) -> None:
        # (1) OpenSearch hybrid 쿼리 + 서브쿼리 2개(텍스트·knn).
        subs = _subqueries(self.body)
        self.assertEqual(len(subs), 2)

    def test_text_subquery_multi_match_nori_fields(self) -> None:
        # (1-①) nori 텍스트 multi_match — query 를 020 nori 필드 5개 대상으로(boost 차등 표기).
        _, clause = _find_with_clause(_subqueries(self.body), "multi_match")
        self.assertIsNotNone(clause, "nori multi_match 서브쿼리가 있어야 한다")
        mm = clause["multi_match"]
        self.assertEqual(mm["query"], self.query)
        self.assertEqual(set(mm["fields"]), _NORI_TEXT_FIELDS)

    def test_text_fields_boost_differentiation(self) -> None:
        # 026 T008(FR-003①): boost 차등이 multi_match fields 에 그대로 실린다 — summary 가 가장 높고
        # (^3), file_name 이 가장 낮다(^0.5). search_text 는 boost 1(표기 없음). 순서·표기 고정.
        _, clause = _find_with_clause(_subqueries(self.body), "multi_match")
        self.assertEqual(
            clause["multi_match"]["fields"],
            ["summary^3", "keywords^2", "labels^1", "file_name^0.5", "search_text"],
        )

    def test_knn_subquery_embedding_vector_k(self) -> None:
        # (1-②) knn 서브쿼리 — embedding 필드·query_vector·k.
        _, clause = _find_with_clause(_subqueries(self.body), "knn")
        self.assertIsNotNone(clause, "knn 서브쿼리가 있어야 한다")
        knn = clause["knn"]["embedding"]
        self.assertEqual(knn["vector"], self.vector)
        self.assertEqual(knn["k"], 50)

    def test_modality_filter_in_each_subquery(self) -> None:
        # (2) modality terms 필터(집합)가 각 서브쿼리에 포함 — text 는 bool.filter, knn 은 native filter.
        for sub in _subqueries(self.body):
            self.assertIn({"terms": {"modality": ["txt"]}}, _sub_bool(sub)["filter"])

    def test_knn_uses_native_prefilter(self) -> None:
        # (2') G3 교정: knn 은 modality 를 **native filter(pre-filter)**로 좁힌다(bool 사후필터 아님) —
        # 작은 k(=버킷 한도)에서 비우세 모달리티(image 등)가 0 건이 되는 것을 막는다(실OS 발견).
        _, knn_sub = _find_with_clause(_subqueries(self.body), "knn")
        self.assertIn("filter", knn_sub["knn"]["embedding"], "knn 에 native filter 가 있어야 한다")
        self.assertIn(
            {"terms": {"modality": ["txt"]}},
            knn_sub["knn"]["embedding"]["filter"]["bool"]["filter"],
        )

    def test_exclude_medical_must_not_default(self) -> None:
        # (3) exclude_medical=True(기본): domain_label='medical' must_not(FR-011). text=bool, knn=native filter.
        for sub in _subqueries(self.body):
            self.assertIn(
                {"term": {"domain_label": "medical"}}, _sub_bool(sub)["must_not"]
            )

    def test_include_medical_when_disabled(self) -> None:
        # (3) exclude_medical=False: 의료 배제 미포함.
        body = build_search_body(
            self.query, self.vector, modality_values=["txt"], exclude_medical=False
        )
        for sub in _subqueries(body):
            self.assertNotIn(
                {"term": {"domain_label": "medical"}},
                _sub_bool(sub).get("must_not", []),
            )

    def test_no_score_field_combined_sort(self) -> None:
        # (4) FR-009 결정적 tiebreaker 는 OS sort 가 아니라 클라이언트(search_assets_os)에서 적용한다.
        # OpenSearch hybrid 쿼리는 _score 와 필드 정렬 조합을 금지(실OS 400: "_score sort criteria
        # cannot be applied with any other criteria")하므로, build_search_body 본문에는 sort 를 두지
        # 않고 정규화 융합 점수(_score)로만 정렬한다(G5 실OS 검증으로 교정).
        self.assertNotIn("sort", self.body)

    def test_size_equals_k(self) -> None:
        # (5) size=k.
        self.assertEqual(self.body["size"], 50)

    def test_audio_modality_filter(self) -> None:
        # 모달리티 분담(text·audio→OS): audio 도 terms 필터(['audio'])로 들어간다.
        body = build_search_body(self.query, self.vector, modality_values=["audio"])
        for sub in _subqueries(body):
            self.assertIn({"terms": {"modality": ["audio"]}}, _sub_bool(sub)["filter"])

    def test_operator_and_requires_all_tokens(self) -> None:
        # 025 FR-001: operator='and' 면 multi_match 가 모든 nori 토큰 매칭을 요구한다 —
        # 복합어 질의('스마트폰 생산량')의 부분 토큰 가짜매칭(F2) 차단. knn 서브쿼리는 무영향.
        body = build_search_body(
            self.query, self.vector, modality_values=["txt"], operator="and"
        )
        _, clause = _find_with_clause(_subqueries(body), "multi_match")
        self.assertEqual(clause["multi_match"]["operator"], "and")
        _, knn = _find_with_clause(_subqueries(body), "knn")
        self.assertIn("knn", knn)  # knn 서브쿼리 구조 불변

    def test_operator_default_or_keeps_body_unchanged(self) -> None:
        # 회귀 0(SC-001): 기본(or)·미지정은 현행 본문과 동일 — multi_match 에 operator 키 자체가 없다.
        default_body = build_search_body(self.query, self.vector, modality_values=["txt"])
        or_body = build_search_body(
            self.query, self.vector, modality_values=["txt"], operator="or"
        )
        self.assertEqual(default_body, or_body)
        _, clause = _find_with_clause(_subqueries(default_body), "multi_match")
        self.assertNotIn("operator", clause["multi_match"])

    def test_text_modality_values_use_terms_set(self) -> None:
        # text 버킷은 다중 modality 값(txt·json·pdf·office)을 terms 집합으로 정렬해 거른다 —
        # 실데이터의 'txt' 가 'text' 라벨로 색인되지 않는 불일치를 흡수(실OS A/B 에서 발견·교정).
        from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS

        body = build_search_body(
            self.query, self.vector, modality_values=ALLOWED_TEXT_META_FILE_KINDS
        )
        expected = {"terms": {"modality": sorted(ALLOWED_TEXT_META_FILE_KINDS)}}
        for sub in _subqueries(body):
            self.assertIn(expected, _sub_bool(sub)["filter"])


class OsHitToRowTest(unittest.TestCase):
    def test_maps_to_media_search_bucket_shape(self) -> None:
        # OS hit → media_search 버킷 행 핵심 키(id·modality·similarity·summary·file_uri).
        hit = {
            "_id": "asset-uuid-1",
            "_score": 0.87,
            "_source": {
                "asset_id": "asset-uuid-1",
                "modality": "text",
                "domain_label": "general",
                "file_name": "보고서.txt",
                "fs_uri": "/data/보고서.txt",
                "summary": "요약 텍스트",
            },
        }
        row = os_hit_to_row(hit)
        self.assertEqual(row["id"], "asset-uuid-1")
        self.assertIsInstance(row["id"], str)
        self.assertEqual(row["modality"], "text")
        # similarity = _score(검색 파이프라인 정규화·융합 점수).
        self.assertEqual(row["similarity"], 0.87)
        self.assertEqual(row["summary"], "요약 텍스트")
        self.assertEqual(row["file_uri"], "/data/보고서.txt")

    def test_row_keys_homogeneous_with_media_search_bucket(self) -> None:
        # SC-005 응답 동형: media_search 버킷 행 핵심 키 집합과 동일(id·file_uri·modality·summary·similarity).
        hit = {
            "_id": "x",
            "_score": 1.0,
            "_source": {"asset_id": "x", "modality": "audio"},
        }
        row = os_hit_to_row(hit)
        self.assertEqual(
            set(row), {"id", "file_uri", "modality", "summary", "similarity"}
        )

    def test_missing_source_safe(self) -> None:
        # 엣지: _source 누락·_score None 안전 처리.
        row = os_hit_to_row({"_id": "y", "_score": None})
        self.assertEqual(row["id"], "y")
        self.assertEqual(row["similarity"], 0.0)
        self.assertEqual(row["summary"], "")
        self.assertEqual(row["file_uri"], "")
        self.assertIsNone(row["modality"])

    def test_none_meta_safe(self) -> None:
        # 엣지: 메타 필드 None 안전 처리(빈 문자열/None).
        hit = {
            "_id": "z",
            "_score": 0.5,
            "_source": {
                "asset_id": "z",
                "modality": None,
                "summary": None,
                "fs_uri": None,
            },
        }
        row = os_hit_to_row(hit)
        self.assertEqual(row["summary"], "")
        self.assertEqual(row["file_uri"], "")
        self.assertIsNone(row["modality"])

    def test_score_nan_safe(self) -> None:
        # 비유한 점수(NaN/inf) 방어 → 0.0.
        row = os_hit_to_row({"_id": "q", "_score": float("nan"), "_source": {"asset_id": "q"}})
        self.assertEqual(row["similarity"], 0.0)

    def test_asset_id_falls_back_to_underscore_id(self) -> None:
        # _source.asset_id 부재 시 _id 로 폴백(OS 색인이 _id=asset_id 라 동일).
        row = os_hit_to_row({"_id": "fallback-id", "_score": 0.3, "_source": {"modality": "text"}})
        self.assertEqual(row["id"], "fallback-id")


# ──────────────────────────────────────────────────────────────────────────
# G2 — IO seam(검색 파이프라인·질의 임베딩·search_assets_os). 가짜 OS 클라이언트 주입.
# 실제 opensearch-py 검색 파이프라인 API(client.search_pipeline.get/put·client.search(search_pipeline=))
# 시그니처는 G5 실OS 에서 확정 검증한다 — 여기선 가짜 클라이언트가 그 인터페이스를 흉내내 액션 조립만 본다.
# ──────────────────────────────────────────────────────────────────────────


class _FakeSearchPipelineNS:
    """opensearch-py ``client.search_pipeline`` 네임스페이스 대역(get/put).

    ``get()``(id 미지정)은 전체 검색 파이프라인 dict 를 돌려준다(실 OS: 없으면 ``{}``) — 020
    ``indices.exists`` 의 boolean-멱등 패턴과 동형으로, 멤버십 검사만으로 존재 여부를 판단한다.
    ``put`` 호출은 (id, body) 로 기록해 멱등(존재 시 PUT 0회)을 단언한다.
    """

    def __init__(self, existing: dict | None = None) -> None:
        self._existing: dict = dict(existing or {})
        self.put_calls: list[tuple[str, dict]] = []

    def get(self, *, id=None, **_kw):  # noqa: A002 (opensearch-py 시그니처 미러)
        if id is None:
            return dict(self._existing)
        return {id: self._existing[id]}  # 미러용(미사용 경로) — 부재 시 KeyError

    def put(self, *, id, body, **_kw):  # noqa: A002
        self.put_calls.append((id, body))
        self._existing[id] = body
        return {"acknowledged": True}


class _FakeSearchClient:
    """opensearch-py OpenSearch 클라이언트 대역 — search_pipeline NS + search() 기록.

    ``search`` 호출 인자를 기록하고 modality 별 canned hits 를 돌려준다. ``raise_on_search`` 가
    주어지면 그 예외를 던져 OS 미도달(FR-008)을 흉내낸다.
    """

    def __init__(
        self,
        *,
        existing_pipelines: dict | None = None,
        hits_by_modality: dict | None = None,
        raise_on_search: Exception | None = None,
    ) -> None:
        self.search_pipeline = _FakeSearchPipelineNS(existing_pipelines)
        self._hits_by_modality = hits_by_modality or {}
        self._raise_on_search = raise_on_search
        self.search_calls: list[dict] = []

    def search(self, *, index=None, body=None, search_pipeline=None, **_kw):
        self.search_calls.append(
            {"index": index, "body": body, "search_pipeline": search_pipeline}
        )
        if self._raise_on_search is not None:
            raise self._raise_on_search
        terms = body["query"]["hybrid"]["queries"][0]["bool"]["filter"][0]["terms"][
            "modality"
        ]
        # 저장값 집합(terms) → canned hits 버킷 라벨. audio/video/image 는 단일값, 그 외(txt·json…)는 text.
        if "audio" in terms:
            label = "audio"
        elif "video" in terms:
            label = "video"
        elif "image" in terms:
            label = "image"
        else:
            label = "text"
        hits = self._hits_by_modality.get(label, [])
        return {"hits": {"hits": hits}}


def _os_hit(asset_id: str, score: float, modality: str) -> dict:
    return {
        "_id": asset_id,
        "_score": score,
        "_source": {"asset_id": asset_id, "modality": modality, "summary": f"요약 {asset_id}"},
    }


class EnsureSearchPipelineTest(unittest.TestCase):
    """T003/FR-006: normalization-processor 검색 파이프라인을 멱등 등록(020 ensure_index 동형)."""

    def test_puts_when_absent(self) -> None:
        # 부재 → PUT 1회, 'created'. 본문에 normalization-processor + 가중치.
        client = _FakeSearchClient(existing_pipelines={})
        result = ensure_search_pipeline(client, "assets-hybrid", weights=(0.3, 0.7))
        self.assertEqual(result, "created")
        self.assertEqual(len(client.search_pipeline.put_calls), 1)
        put_id, body = client.search_pipeline.put_calls[0]
        self.assertEqual(put_id, "assets-hybrid")
        procs = body["phase_results_processors"]
        norm = procs[0]["normalization-processor"]
        self.assertEqual(norm["normalization"]["technique"], "min_max")
        self.assertEqual(
            norm["combination"]["parameters"]["weights"], [0.3, 0.7]
        )

    def test_noop_when_exists(self) -> None:
        # 존재 → PUT 0회(멱등), 'exists'.
        client = _FakeSearchClient(existing_pipelines={"assets-hybrid": {"x": 1}})
        result = ensure_search_pipeline(client, "assets-hybrid", weights=(0.5, 0.5))
        self.assertEqual(result, "exists")
        self.assertEqual(client.search_pipeline.put_calls, [])


class SearchAssetsOsTest(unittest.TestCase):
    """T003/FR-002: 질의 1회 임베딩 → modality 별 build_search_body → client.search → 버킷."""

    def setUp(self) -> None:
        self.query = "한국어 검색 질의"
        self.vector = [0.11, 0.22, 0.33]
        self.embed_calls: list[tuple[str, str]] = []

        def fake_embed(q: str, *, channel: str) -> list[float]:
            self.embed_calls.append((q, channel))
            return list(self.vector)

        self.fake_embed = fake_embed
        self.client = _FakeSearchClient(
            hits_by_modality={
                "text": [_os_hit("t1", 0.9, "text"), _os_hit("t2", 0.7, "text")],
                "audio": [_os_hit("a1", 0.8, "audio")],
            }
        )

    def _run(self, **overrides):
        kwargs = {
            "modalities": ("text", "audio"),
            "k": 20,
            "channel": "st",
            "weights": (0.4, 0.6),
            "index": "assets",
            "pipeline_name": "assets-hybrid",
            "embed_fn": self.fake_embed,
        }
        kwargs.update(overrides)
        return search_assets_os(self.client, self.query, **kwargs)

    def test_embeds_query_once_and_reuses_vector(self) -> None:
        # (a) embed_fn 으로 질의 1회만 임베딩(modality 마다 재임베딩 금지) — 활성 채널 전달.
        self._run()
        self.assertEqual(len(self.embed_calls), 1)
        self.assertEqual(self.embed_calls[0], (self.query, "st"))

    def test_searches_each_modality_with_pipeline(self) -> None:
        # (b) modality 마다 client.search(index=, body=, search_pipeline=) 1회.
        self._run()
        self.assertEqual(len(self.client.search_calls), 2)
        for call in self.client.search_calls:
            self.assertEqual(call["index"], "assets")
            self.assertEqual(call["search_pipeline"], "assets-hybrid")

    def test_search_body_uses_builder_with_vector_and_modality(self) -> None:
        # build_search_body 의 산출(하이브리드·knn 벡터·modality 필터·size=k)을 그대로 search body 로.
        self._run()
        modalities_searched = []
        for call in self.client.search_calls:
            body = call["body"]
            self.assertEqual(body["size"], 20)
            subs = body["query"]["hybrid"]["queries"]
            _, knn = _find_with_clause(subs, "knn")
            self.assertEqual(knn["knn"]["embedding"]["vector"], self.vector)
            terms = subs[0]["bool"]["filter"][0]["terms"]["modality"]
            modalities_searched.append("audio" if "audio" in terms else "text")
        self.assertEqual(set(modalities_searched), {"text", "audio"})

    def test_results_deterministic_tiebreaker(self) -> None:
        # FR-009(헌법 3조): OS sort 불가(hybrid 제약) → 클라이언트에서 (-similarity, id) 결정적 정렬.
        # OS 가 동점·뒤섞인 순서로 hit 을 줘도 출력은 점수 desc·동점 id asc 로 고정된다.
        client = _FakeSearchClient(
            hits_by_modality={
                "text": [
                    _os_hit("b", 0.5, "text"),
                    _os_hit("c", 0.9, "text"),
                    _os_hit("a", 0.5, "text"),
                ]
            }
        )
        out = search_assets_os(
            client,
            self.query,
            modalities=("text",),
            index="assets",
            pipeline_name="assets-hybrid",
            embed_fn=self.fake_embed,
        )
        self.assertEqual([r["id"] for r in out["text"]], ["c", "a", "b"])

    def test_returns_bucket_per_modality_mapped_rows(self) -> None:
        # (c) {modality: [os_hit_to_row 행]} 버킷 — text 2건·audio 1건.
        buckets = self._run()
        self.assertEqual(set(buckets), {"text", "audio"})
        self.assertEqual(len(buckets["text"]), 2)
        self.assertEqual(len(buckets["audio"]), 1)
        # os_hit_to_row 와 동형(id·similarity·modality·summary·file_uri).
        row = buckets["text"][0]
        self.assertEqual(set(row), {"id", "file_uri", "modality", "summary", "similarity"})
        self.assertEqual(row["id"], "t1")
        self.assertEqual(row["similarity"], 0.9)
        self.assertEqual(row["modality"], "text")

    def test_empty_bucket_when_no_hits(self) -> None:
        # hits 없는 modality 는 빈 버킷(키는 존재).
        buckets = self._run(modalities=("text", "video"))
        self.assertEqual(buckets["video"], [])

    def test_propagates_os_unreachable_error(self) -> None:
        # FR-008: client.search 예외 → search_assets_os 가 전파(silent 폴백 없음).
        client = _FakeSearchClient(raise_on_search=ConnectionError("OS 미도달"))
        with self.assertRaises(ConnectionError):
            search_assets_os(
                client,
                self.query,
                modalities=("text",),
                k=10,
                channel="st",
                weights=(0.5, 0.5),
                index="assets",
                pipeline_name="assets-hybrid",
                embed_fn=self.fake_embed,
            )


# ──────────────────────────────────────────────────────────────────────────
# 022 G1 — image/video 모달리티 값 매핑(순수·seam). image/video 는 020 assets 인덱스에 캡션 nori +
# KoSimCSE 캡션 임베딩으로 이미 색인돼 있어 text/audio 와 동일 OS 하이브리드로 검색한다(CLIP 아님).
# ──────────────────────────────────────────────────────────────────────────


class ModalityValuesMappingTest(unittest.TestCase):
    """T001/FR-001: image·video 라벨이 저장값 집합으로 _MODALITY_VALUES 에 명시 해소되는지."""

    def test_image_video_explicit_in_modality_values(self) -> None:
        # image/video 는 020 인덱스에 라벨과 동일한 modality 값('image'/'video')으로 색인된다.
        # search_assets_os 의 fallback frozenset({label}) 로도 동작하지만, 022 는 지원 모달리티를
        # 문서화·가독화하려 _MODALITY_VALUES 에 명시 등재한다 — 그 등재를 직접 단언(명시 전 RED: KeyError).
        self.assertEqual(_MODALITY_VALUES[MediaKind.IMAGE.value], frozenset({"image"}))
        self.assertEqual(_MODALITY_VALUES[MediaKind.VIDEO.value], frozenset({"video"}))

    def test_text_audio_mapping_unchanged(self) -> None:
        # 021 매핑 무손상(회귀 0): text=ALLOWED_TEXT_META_FILE_KINDS, audio={'audio'}.
        self.assertEqual(
            _MODALITY_VALUES["text"], frozenset(ALLOWED_TEXT_META_FILE_KINDS)
        )
        self.assertEqual(_MODALITY_VALUES[MediaKind.AUDIO.value], frozenset({"audio"}))


class ImageVideoHybridBodyTest(unittest.TestCase):
    """T001/FR-001·004: image·video 도 text/audio 와 동일 OS 하이브리드(캡션 nori + 캡션 임베딩 knn).

    image/video 자산의 ``embedding`` 은 KoSimCSE 캡션 임베딩(텍스트 의미)이고 질의도 같은 채널로
    임베딩하므로 같은 벡터 공간(plan §R3). CLIP·새 필드 없이 build_search_body 로 동형 본문을 만든다.
    """

    def setUp(self) -> None:
        self.query = "아이폰으로 찍은 사진"
        self.vector = [0.5, 0.6, 0.7]

    def _check_modality(self, modality_value: str) -> None:
        body = build_search_body(
            self.query, self.vector, modality_values=[modality_value], k=30
        )
        subs = _subqueries(body)
        # (1) 하이브리드 서브쿼리 2개(nori multi_match + embedding knn) — text/audio 와 동일.
        self.assertEqual(len(subs), 2)
        _, mm = _find_with_clause(subs, "multi_match")
        self.assertIsNotNone(mm, "nori multi_match 서브쿼리가 있어야 한다")
        self.assertEqual(mm["multi_match"]["query"], self.query)
        self.assertEqual(set(mm["multi_match"]["fields"]), _NORI_TEXT_FIELDS)
        _, knn = _find_with_clause(subs, "knn")
        self.assertIsNotNone(knn, "knn 서브쿼리가 있어야 한다")
        self.assertEqual(knn["knn"]["embedding"]["vector"], self.vector)
        # (2) terms 필터(라벨→값 집합) + (3) 의료배제 must_not(FR-004)이 각 서브쿼리에 적용 —
        # text 는 bool, knn 은 native filter(pre-filter, G3 교정). _sub_bool 로 양쪽 다 꺼낸다.
        for sub in subs:
            self.assertIn(
                {"terms": {"modality": [modality_value]}}, _sub_bool(sub)["filter"]
            )
            self.assertIn(
                {"term": {"domain_label": "medical"}}, _sub_bool(sub)["must_not"]
            )
        self.assertEqual(body["size"], 30)

    def test_image_hybrid_body(self) -> None:
        self._check_modality("image")

    def test_video_hybrid_body(self) -> None:
        self._check_modality("video")


class SearchAssetsOsImageVideoTest(unittest.TestCase):
    """T001/FR-001·006: search_assets_os 가 image·video 라벨을 terms 필터로 해소·하이브리드 검색·결정 정렬."""

    def setUp(self) -> None:
        self.query = "강아지 영상"
        self.vector = [0.1, 0.9]

        def fake_embed(q: str, *, channel: str) -> list[float]:
            return list(self.vector)

        self.fake_embed = fake_embed

    def test_image_video_resolve_to_terms_filter(self) -> None:
        # 라벨('image'/'video') → 저장값 집합 terms 필터로 해소돼 그 버킷에서 회수된다(FR-001).
        client = _FakeSearchClient(
            hits_by_modality={
                "image": [_os_hit("i1", 0.9, "image")],
                "video": [_os_hit("v1", 0.8, "video"), _os_hit("v2", 0.6, "video")],
            }
        )
        buckets = search_assets_os(
            client,
            self.query,
            modalities=("image", "video"),
            index="assets",
            pipeline_name="assets-hybrid",
            embed_fn=self.fake_embed,
        )
        self.assertEqual(set(buckets), {"image", "video"})
        self.assertEqual([r["id"] for r in buckets["image"]], ["i1"])
        self.assertEqual([r["id"] for r in buckets["video"]], ["v1", "v2"])
        # 각 search 본문의 terms 필터가 라벨→값 집합으로 해소됐는지(image→['image'], video→['video']).
        searched = []
        for call in client.search_calls:
            terms = call["body"]["query"]["hybrid"]["queries"][0]["bool"]["filter"][0][
                "terms"
            ]["modality"]
            searched.append(tuple(terms))
        self.assertIn(("image",), searched)
        self.assertIn(("video",), searched)

    def test_image_deterministic_tiebreaker(self) -> None:
        # FR-006(021 동형): 동점에서 (-similarity, id) 결정 정렬이 image 버킷에도 그대로 적용.
        client = _FakeSearchClient(
            hits_by_modality={
                "image": [
                    _os_hit("b", 0.5, "image"),
                    _os_hit("c", 0.9, "image"),
                    _os_hit("a", 0.5, "image"),
                ]
            }
        )
        out = search_assets_os(
            client,
            self.query,
            modalities=("image",),
            index="assets",
            pipeline_name="assets-hybrid",
            embed_fn=self.fake_embed,
        )
        self.assertEqual([r["id"] for r in out["image"]], ["c", "a", "b"])


# ──────────────────────────────────────────────────────────────────────────
# 023 G1 — 적합도 게이트 순수 함수(probe 본문·코사인 환산·게이트). OS·DB·opensearch-py 불필요.
# 정규화 파이프라인 없는 plain kNN probe 로 원시 코사인을 얻어 buckets 유지/비움을 결정한다.
# 실 OS 의 정확한 스케일·calibration 은 G4(실OS e2e)에서 확정한다 — 여기선 결정적 로직만 고정.
# ──────────────────────────────────────────────────────────────────────────


class BuildProbeBodyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vector = [0.1, 0.2, 0.3]

    def test_plain_knn_no_pipeline_fields(self) -> None:
        # probe 는 정규화 파이프라인 없는 plain knn 1개(하이브리드 아님) — size=k, embedding 벡터·k.
        body = build_probe_body(self.vector, modality_values=["image"], k=50)
        self.assertNotIn("hybrid", body["query"])
        knn = body["query"]["knn"]["embedding"]
        self.assertEqual(knn["vector"], self.vector)
        self.assertEqual(knn["k"], 50)
        self.assertEqual(body["size"], 50)

    def test_modality_terms_prefilter(self) -> None:
        # modality terms 필터(정렬·집합)가 knn native filter(pre-filter)에 들어간다 — 같은 후보 집단.
        body = build_probe_body(self.vector, modality_values=["video", "image"], k=10)
        flt = body["query"]["knn"]["embedding"]["filter"]["bool"]["filter"]
        self.assertIn({"terms": {"modality": ["image", "video"]}}, flt)

    def test_exclude_medical_must_not_default(self) -> None:
        # FR-008: 의료배제 기본 — domain_label='medical' must_not(정규화 경로와 동일 후보).
        body = build_probe_body(self.vector, modality_values=["text"], k=10)
        self.assertIn(
            {"term": {"domain_label": "medical"}},
            body["query"]["knn"]["embedding"]["filter"]["bool"]["must_not"],
        )

    def test_include_medical_when_disabled(self) -> None:
        body = build_probe_body(self.vector, modality_values=["text"], k=10, exclude_medical=False)
        self.assertNotIn("must_not", body["query"]["knn"]["embedding"]["filter"]["bool"])


class KnnScoreToCosineTest(unittest.TestCase):
    def test_lucene_cosinesimil_inverse(self) -> None:
        # 020 인덱스: space_type=cosinesimil·engine=lucene → _score=(1+cos)/2 → cos=2*score-1.
        self.assertAlmostEqual(knn_score_to_cosine(1.0), 1.0)    # cos=1
        self.assertAlmostEqual(knn_score_to_cosine(0.5), 0.0)    # cos=0
        self.assertAlmostEqual(knn_score_to_cosine(0.85), 0.7)   # 진단 영역

    def test_clamps_and_safe(self) -> None:
        # 비유한·범위 밖 방어([-1,1] clamp) — 결정적.
        self.assertEqual(knn_score_to_cosine(float("nan")), 0.0)
        self.assertEqual(knn_score_to_cosine(2.0), 1.0)
        self.assertEqual(knn_score_to_cosine(-1.0), -1.0)


class PassesCutoffTest(unittest.TestCase):
    # 진단: no-match max-mean ≤0.088·top ≤0.707 / match max-mean ≥0.116·top ≥0.745.
    def test_keep_strong_relative_signal(self) -> None:  # match
        self.assertTrue(passes_cutoff(0.78, 0.66, eps=0.10, floor=0.65))   # top-mean=0.12, top≥floor

    def test_cut_flat_no_match(self) -> None:  # no-match: 상대신호 부족
        self.assertFalse(passes_cutoff(0.70, 0.65, eps=0.10, floor=0.65))  # top-mean=0.05 < 0.10

    def test_cut_below_floor_backstop(self) -> None:  # 전체 평평·낮음 → floor 가 차단
        self.assertFalse(passes_cutoff(0.60, 0.30, eps=0.10, floor=0.65))  # top-mean ok, top<floor

    def test_boundary_inclusive(self) -> None:  # 경계 포함(>=)
        self.assertTrue(passes_cutoff(0.75, 0.65, eps=0.10, floor=0.65))   # =0.10, =0.65

    def test_empty_probe_zero_cut(self) -> None:  # probe 빈(top=mean=0) → cut
        self.assertFalse(passes_cutoff(0.0, 0.0, eps=0.10, floor=0.65))


# ──────────────────────────────────────────────────────────────────────────
# 023 G2 — probe IO seam(probe_relevance) + search_assets_os 게이트 통합. 가짜 클라이언트·주입 probe_fn.
# probe 는 정규화 검색과 분리된 plain kNN 1회다 — search_pipeline 인자 없이 호출됨을 단언한다.
# 실 opensearch-py knn 응답·스케일은 G4(실OS) 확정(021 ensure_search_pipeline 동형).
# ──────────────────────────────────────────────────────────────────────────


class _ProbeFakeClient:
    """plain knn probe 용 가짜 — search(search_pipeline 미지정) 호출을 기록하고 canned hits 반환."""
    def __init__(self, hits):
        self._hits = hits
        self.calls = []
    def search(self, *, index=None, body=None, search_pipeline=None, **_kw):
        self.calls.append({"index": index, "body": body, "search_pipeline": search_pipeline})
        return {"hits": {"hits": self._hits}}


class ProbeRelevanceTest(unittest.TestCase):
    def test_returns_top_and_mean_cosine(self) -> None:
        # _score (1+cos)/2: 0.9→cos0.8, 0.8→0.6, 0.75→0.5 ⇒ top=0.8, mean=(0.8+0.6+0.5)/3≈0.6333.
        client = _ProbeFakeClient([{"_score": 0.9}, {"_score": 0.8}, {"_score": 0.75}])
        top, mean = probe_relevance(client, [0.1, 0.2], modality_values=["image"], k=50, index="assets")
        self.assertAlmostEqual(top, 0.8, places=4)
        self.assertAlmostEqual(mean, (0.8 + 0.6 + 0.5) / 3, places=4)

    def test_no_pipeline_and_probe_body(self) -> None:
        # 정규화 파이프라인 미적용(search_pipeline=None) + build_probe_body 본문(plain knn).
        client = _ProbeFakeClient([{"_score": 0.9}])
        probe_relevance(client, [0.1], modality_values=["text"], k=5, index="assets")
        call = client.calls[0]
        self.assertIsNone(call["search_pipeline"])
        self.assertIn("knn", call["body"]["query"])

    def test_empty_hits_zero(self) -> None:
        top, mean = probe_relevance(_ProbeFakeClient([]), [0.1], modality_values=["video"], k=5, index="assets")
        self.assertEqual((top, mean), (0.0, 0.0))


class SearchAssetsOsCutoffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vector = [0.1, 0.2]
        def fake_embed(q, *, channel): return list(self.vector)
        self.fake_embed = fake_embed
        self.client = _FakeSearchClient(hits_by_modality={
            "text": [_os_hit("t1", 0.9, "text"), _os_hit("t2", 0.7, "text")],
            "image": [_os_hit("i1", 0.8, "image")],
        })

    def _run(self, probe_fn, **ov):
        kw = {
            "modalities": ("text", "image"), "index": "assets", "pipeline_name": "assets-hybrid",
            "embed_fn": self.fake_embed, "cutoff_enabled": True, "cutoff_eps": 0.10,
            "cutoff_floor": 0.65, "probe_fn": probe_fn,
        }
        kw.update(ov)
        return search_assets_os(self.client, "질의", **kw)

    def test_disabled_no_probe_unchanged(self) -> None:
        # 회귀 0: cutoff_enabled=False(기본) → probe 미호출·버킷 그대로(021/022 동작).
        calls = []
        def probe_fn(*a, **k):
            calls.append(1)
            return (0.0, 0.0)
        out = search_assets_os(self.client, "질의", modalities=("text",), index="assets",
                               pipeline_name="assets-hybrid", embed_fn=self.fake_embed, probe_fn=probe_fn)
        self.assertEqual(calls, [])
        self.assertEqual(len(out["text"]), 2)

    def test_keep_bucket_when_signal_passes(self) -> None:
        # 강한 신호(top-mean=0.12·top≥floor) → 버킷 유지(정규화 랭킹 그대로, FR-003).
        def probe_fn(client, vec, *, modality_values, k, index, exclude_medical=True):
            return (0.78, 0.66)
        out = self._run(probe_fn)
        self.assertEqual([r["id"] for r in out["text"]], ["t1", "t2"])
        self.assertEqual(len(out["image"]), 1)

    def test_empty_bucket_when_signal_fails(self) -> None:
        # 약한 신호(top-mean=0.05<eps) → 빈 버킷(no-match 차단, SC-003·SC-006 동형).
        def probe_fn(client, vec, *, modality_values, k, index, exclude_medical=True):
            return (0.70, 0.65)
        out = self._run(probe_fn)
        self.assertEqual(out["text"], [])
        self.assertEqual(out["image"], [])

    def test_per_modality_gate_independent(self) -> None:
        # modality 별 독립 판정 — text 통과·image 차단.
        def probe_fn(client, vec, *, modality_values, k, index, exclude_medical=True):
            return (0.80, 0.66) if "image" not in set(modality_values) else (0.66, 0.64)
        out = self._run(probe_fn)
        self.assertEqual(len(out["text"]), 2)
        self.assertEqual(out["image"], [])

    def test_probe_receives_same_vector_and_modality(self) -> None:
        # probe 는 질의 벡터 재사용(중복 임베딩 0)·같은 modality 값 집합을 받는다(FR-002).
        seen = []
        def probe_fn(client, vec, *, modality_values, k, index, exclude_medical=True):
            seen.append((tuple(vec), tuple(sorted(modality_values)), exclude_medical))
            return (0.9, 0.6)
        self._run(probe_fn, modalities=("image",))
        self.assertEqual(seen[0][0], tuple(self.vector))
        self.assertEqual(seen[0][1], ("image",))
        self.assertTrue(seen[0][2])  # 의료배제 전달


if __name__ == "__main__":
    unittest.main()
