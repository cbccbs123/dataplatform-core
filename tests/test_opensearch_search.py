"""021 G1·G2 — OpenSearch 검색 쿼리 빌더·결과 매핑·검색 seam 단위 테스트.

G1(순수): OS·DB·opensearch-py 불필요. ``build_search_body``(하이브리드 쿼리 본문)와
``os_hit_to_row``(결과 행 매핑)가 020 인덱스 매핑(opensearch_sync.build_index_body)과
일치하고 media_search 버킷 행과 동형(SC-005)인지 충실히 검증한다.

G2(IO seam): ``ensure_search_pipeline``(정규화 검색 파이프라인 멱등 등록)·``search_assets_os``
(질의 1회 임베딩 → modality 별 하이브리드 검색 → 버킷)을 **가짜 OS 클라이언트 주입**으로 OS 없이
액션 조립을 검증한다(docs/테스트_가이드.md §2 seam 주입). OS 미도달은 예외 전파로 검증(FR-008).

테스트 위조·약화 금지(docs/테스트_가이드.md). 실제 OS 검색 실효·결정성·의료배제는 G5(실OS e2e).
"""

from __future__ import annotations

import unittest

from src.search.opensearch_search import (
    build_search_body,
    ensure_search_pipeline,
    os_hit_to_row,
    search_assets_os,
)

# 020 인덱스의 nori 텍스트 필드(BM25 multi_match 대상). 필드명 정본 = opensearch_sync.build_index_body.
_NORI_TEXT_FIELDS = {"summary", "keywords", "labels", "file_name", "search_text"}


def _subqueries(body: dict) -> list:
    """hybrid 쿼리의 서브쿼리 목록을 꺼낸다."""
    return body["query"]["hybrid"]["queries"]


def _find_with_clause(subqueries: list, clause_key: str):
    """bool.must 에 주어진 절(clause_key)을 가진 서브쿼리·절을 찾는다(없으면 (None, None))."""
    for sub in subqueries:
        for clause in sub.get("bool", {}).get("must", []):
            if clause_key in clause:
                return sub, clause
    return None, None


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
        # (1-①) nori 텍스트 multi_match — query 를 020 nori 필드 5개 대상으로.
        _, clause = _find_with_clause(_subqueries(self.body), "multi_match")
        self.assertIsNotNone(clause, "nori multi_match 서브쿼리가 있어야 한다")
        mm = clause["multi_match"]
        self.assertEqual(mm["query"], self.query)
        self.assertEqual(set(mm["fields"]), _NORI_TEXT_FIELDS)

    def test_knn_subquery_embedding_vector_k(self) -> None:
        # (1-②) knn 서브쿼리 — embedding 필드·query_vector·k.
        _, clause = _find_with_clause(_subqueries(self.body), "knn")
        self.assertIsNotNone(clause, "knn 서브쿼리가 있어야 한다")
        knn = clause["knn"]["embedding"]
        self.assertEqual(knn["vector"], self.vector)
        self.assertEqual(knn["k"], 50)

    def test_modality_filter_in_each_subquery(self) -> None:
        # (2) modality keyword terms 필터(집합)가 각 서브쿼리 bool filter 에 포함 — 단일 term 아님.
        for sub in _subqueries(self.body):
            self.assertIn({"terms": {"modality": ["txt"]}}, sub["bool"]["filter"])

    def test_exclude_medical_must_not_default(self) -> None:
        # (3) exclude_medical=True(기본): domain_label='medical' must_not(FR-011).
        for sub in _subqueries(self.body):
            self.assertIn(
                {"term": {"domain_label": "medical"}}, sub["bool"]["must_not"]
            )

    def test_include_medical_when_disabled(self) -> None:
        # (3) exclude_medical=False: 의료 배제 미포함.
        body = build_search_body(
            self.query, self.vector, modality_values=["txt"], exclude_medical=False
        )
        for sub in _subqueries(body):
            self.assertNotIn(
                {"term": {"domain_label": "medical"}},
                sub["bool"].get("must_not", []),
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
            self.assertIn({"terms": {"modality": ["audio"]}}, sub["bool"]["filter"])

    def test_text_modality_values_use_terms_set(self) -> None:
        # text 버킷은 다중 modality 값(txt·json·pdf·office)을 terms 집합으로 정렬해 거른다 —
        # 실데이터의 'txt' 가 'text' 라벨로 색인되지 않는 불일치를 흡수(실OS A/B 에서 발견·교정).
        from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS

        body = build_search_body(
            self.query, self.vector, modality_values=ALLOWED_TEXT_META_FILE_KINDS
        )
        expected = {"terms": {"modality": sorted(ALLOWED_TEXT_META_FILE_KINDS)}}
        for sub in _subqueries(body):
            self.assertIn(expected, sub["bool"]["filter"])


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
        # 저장값 집합(terms) → canned hits 버킷 라벨. audio/video 는 단일값, 그 외(txt·json…)는 text.
        if "audio" in terms:
            label = "audio"
        elif "video" in terms:
            label = "video"
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


if __name__ == "__main__":
    unittest.main()
