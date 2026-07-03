"""021·027 — OpenSearch 검색 서브검색 빌더·클라이언트 융합·게이트·컷·검색 seam 단위 테스트.

순수(OS·DB·opensearch-py 불필요): 두 서브검색 본문(``build_bm25_body``·``build_knn_body``)·결과 행
매핑(``os_hit_to_row``)이 020 인덱스 매핑(opensearch_sync.build_index_body)과 일치하고 media_search
버킷 행과 동형(SC-005)인지 검증한다. 027 클라이언트 융합 순수 함수(``minmax_normalize``·``fuse_hybrid``·
``gate_signal``·``cut_rows``·``passes_cutoff``)의 융합 수학·robust 게이트·per-result 컷을 결정적으로
고정한다(헌법 3조·FR-002·003·004).

IO seam: ``search_assets_os``(질의 1회 임베딩 → 전 모달리티 [knn+bm25] **_msearch 1회** → 클라이언트
융합 → 게이트 → 컷 → (buckets, gate_meta))을 **가짜 msearch 클라이언트 주입**으로 OS 없이 액션 조립을
검증한다(docs/테스트_가이드.md §2 seam 주입). OS 미도달은 예외 전파로 검증(FR-007).

022(image/video): image·video 도 020 assets 인덱스에 한국어 VLM 캡션(nori) + KoSimCSE 캡션 임베딩
(``embedding``)으로 색인돼 text/audio 와 **동일 OS 서브검색**으로 검색한다(CLIP 아님). 라벨 명시 등재와
그 서브검색 본문·의료배제를 고정한다.

테스트 위조·약화 금지(docs/테스트_가이드.md). 실제 OS 검색 실효·결정성·의료배제·재보정은 G4(실OS e2e).
"""

from __future__ import annotations

import json
import unittest
from datetime import date

from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind
from src.search.opensearch_search import (
    _MODALITY_VALUES,
    BM25_NAMED_QUERY_NAMES,
    build_bm25_body,
    build_knn_body,
    cut_rows,
    fuse_hybrid,
    gate_signal,
    knn_score_to_cosine,
    minmax_normalize,
    normalize_query,
    os_hit_to_row,
    passes_cutoff,
    rerank_reorder,
    search_assets_os,
)
from src.search.search_filters import SearchFilters

# 044: named query should 절 — boost는 clause boost 로 계승. 047: cross_meta 는 multi_match cross_fields.
_BM25_FIELD_BOOSTS = {
    "summary": 3.0,
    "keywords": 2.0,
    "labels": 1.0,
    "file_name": 0.5,
}
_CROSS_META_BOOST = 1.0


def _bm25_named_names(body: dict) -> set[str]:
    names: set[str] = set()
    for clause in body["query"]["bool"]["should"]:
        if "match" in clause:
            for inner in clause["match"].values():
                names.add(inner["_name"])
        elif "term" in clause:
            for inner in clause["term"].values():
                names.add(inner["_name"])
        elif "multi_match" in clause:
            names.add(clause["multi_match"]["_name"])
    return names


def _bm25_first_match_query(body: dict) -> str:
    for clause in body["query"]["bool"]["should"]:
        if "match" in clause:
            for inner in clause["match"].values():
                return str(inner["query"])
    raise AssertionError("match 절 없음")


# 027 T006: hybrid 서브쿼리 헬퍼(_subqueries·_find_with_clause·_sub_bool)와 BuildSearchBodyTest 는
# 제거됐다 — 서버 hybrid 쿼리가 두 독립 서브검색(build_bm25_body·build_knn_body)으로 분리되면서
# 그 본문 검증 의도는 BuildBm25BodyTest·BuildKnnBodyTest·ImageVideoSubqueryBodyTest 로 이전됐다.


class OsHitToRowTest(unittest.TestCase):
    def test_maps_to_media_search_bucket_shape(self) -> None:
        # OS hit → media_search 버킷 행 핵심 키(id·modality·domain_label·similarity·summary·file_uri).
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
                "topics": ["요리", "베이킹"],
                "subtopics": ["제빵"],
            },
        }
        row = os_hit_to_row(hit)
        self.assertEqual(row["id"], "asset-uuid-1")
        self.assertIsInstance(row["id"], str)
        self.assertEqual(row["modality"], "text")
        self.assertEqual(row["domain_label"], "general")
        # similarity = _score(검색 파이프라인 정규화·융합 점수).
        self.assertEqual(row["similarity"], 0.87)
        self.assertEqual(row["summary"], "요약 텍스트")
        self.assertEqual(row["file_uri"], "/data/보고서.txt")
        # 057-후속: 색인 topics/subtopics 통과(결과-좁히기 패싯·클라 필터·필터와 동일 소스).
        self.assertEqual(row["topics"], ["요리", "베이킹"])
        self.assertEqual(row["subtopics"], ["제빵"])

    def test_row_keys_homogeneous_with_media_search_bucket(self) -> None:
        # SC-005 응답 동형: media_search 버킷 행 핵심 키 집합과 동일(042 domain_label 포함).
        hit = {
            "_id": "x",
            "_score": 1.0,
            "_source": {"asset_id": "x", "modality": "audio", "domain_label": "review"},
        }
        row = os_hit_to_row(hit)
        self.assertEqual(
            set(row),
            {"id", "file_uri", "modality", "domain_label", "summary", "similarity",
             "topics", "subtopics"},  # 057-후속: 주제 패싯·클라 좁히기용
        )
        self.assertEqual(row["domain_label"], "review")
        self.assertEqual(row["topics"], [])  # topics 없는 hit → 빈 리스트(안전)

    def test_missing_source_safe(self) -> None:
        # 엣지: _source 누락·_score None 안전 처리.
        row = os_hit_to_row({"_id": "y", "_score": None})
        self.assertEqual(row["id"], "y")
        self.assertEqual(row["similarity"], 0.0)
        self.assertEqual(row["summary"], "")
        self.assertEqual(row["file_uri"], "")
        self.assertIsNone(row["modality"])
        self.assertIsNone(row["domain_label"])

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
        self.assertIsNone(row["domain_label"])

    def test_score_nan_safe(self) -> None:
        # 비유한 점수(NaN/inf) 방어 → 0.0.
        row = os_hit_to_row({"_id": "q", "_score": float("nan"), "_source": {"asset_id": "q"}})
        self.assertEqual(row["similarity"], 0.0)

    def test_asset_id_falls_back_to_underscore_id(self) -> None:
        # _source.asset_id 부재 시 _id 로 폴백(OS 색인이 _id=asset_id 라 동일).
        row = os_hit_to_row({"_id": "fallback-id", "_score": 0.3, "_source": {"modality": "text"}})
        self.assertEqual(row["id"], "fallback-id")


# 027 T006: 서버 파이프라인 등록(EnsureSearchPipelineTest)·구 hybrid search_assets_os(SearchAssetsOsTest)
# 와 그 가짜(_FakeSearchPipelineNS·_FakeSearchClient·_os_hit)는 제거됐다 — 파이프라인 경로(FR-005)와
# 구 client.search 융합 경로가 msearch 클라이언트 융합으로 대체됐기 때문. 그 검증 의도(질의 1회 임베딩·
# modality 별 본문·결정 정렬·버킷·OS 미도달 전파)는 SearchAssetsOsMsearchTest 로 이전됐다.


# ──────────────────────────────────────────────────────────────────────────
# 022 G1 — image/video 모달리티 값 매핑(순수·seam). image/video 는 020 assets 인덱스에 캡션 nori +
# KoSimCSE 캡션 임베딩으로 이미 색인돼 있어 text/audio 와 동일 OS 서브검색으로 검색한다(CLIP 아님).
# ──────────────────────────────────────────────────────────────────────────


class ModalityValuesMappingTest(unittest.TestCase):
    """T001/FR-001: image·video 라벨이 저장값 집합으로 _MODALITY_VALUES 에 명시 해소되는지."""

    def test_image_video_explicit_in_modality_values(self) -> None:
        # image/video 는 020 인덱스에 라벨과 동일한 modality 값('image'/'video')으로 색인된다.
        # search_assets_os 의 fallback frozenset({label}) 로도 동작하지만, 022 는 지원 모달리티를
        # 문서화·가독화하려 _MODALITY_VALUES 에 명시 등재한다 — 그 등재를 직접 단언(명시 전 RED: KeyError).
        self.assertEqual(_MODALITY_VALUES[MediaKind.IMAGE.value], frozenset({"image"}))
        self.assertEqual(_MODALITY_VALUES[MediaKind.VIDEO.value], frozenset({"video"}))

    def test_text_mapping_union_for_canonical_transition(self) -> None:
        # 053 canonical 전환: 저장 modality 는 canonical 'text'. 재색인 전 구 OS 문서엔 file_kind
        # 값(txt/json/pdf/office)이 남아 있어 합집합({text}∪ALLOWED_TEXT_META_FILE_KINDS)으로
        # 구·신 문서를 동시 매칭한다(무중단·C5). 재색인 안정 후 {"text"}로 정리(FR-403).
        self.assertEqual(
            _MODALITY_VALUES["text"], frozenset({"text", *ALLOWED_TEXT_META_FILE_KINDS})
        )
        self.assertIn("text", _MODALITY_VALUES["text"])  # canonical 저장값 회수
        self.assertIn("txt", _MODALITY_VALUES["text"])   # 재색인 전 구값 회수
        self.assertEqual(_MODALITY_VALUES[MediaKind.AUDIO.value], frozenset({"audio"}))


class ImageVideoSubqueryBodyTest(unittest.TestCase):
    """T006(022 의도 이전): image·video 도 text/audio 와 동일 두 서브검색(BM25 + 캡션 임베딩 kNN).

    image/video 자산의 ``embedding`` 은 KoSimCSE 캡션 임베딩(텍스트 의미)이고 질의도 같은 채널로
    임베딩하므로 같은 벡터 공간(plan §R3). CLIP·새 필드 없이 build_bm25_body·build_knn_body 로 동형
    서브검색 본문을 만든다 — 구 ImageVideoHybridBodyTest(하이브리드 단일 본문)의 검증 의도 이전.
    """

    def setUp(self) -> None:
        self.query = "아이폰으로 찍은 사진"
        self.vector = [0.5, 0.6, 0.7]

    def _check_modality(self, modality_value: str) -> None:
        bm25 = build_bm25_body(self.query, modality_values=[modality_value], k=30)
        knn = build_knn_body(self.vector, modality_values=[modality_value], k=30)
        # (1) BM25 서브검색: named query should + terms 필터 + 의료 must_not + size.
        self.assertEqual(_bm25_first_match_query(bm25), self.query)
        self.assertEqual(_bm25_named_names(bm25), set(BM25_NAMED_QUERY_NAMES))
        self.assertIn({"terms": {"modality": [modality_value]}}, bm25["query"]["bool"]["filter"])
        self.assertIn(
            {"term": {"domain_label": "medical"}}, bm25["query"]["bool"]["must_not"]
        )
        self.assertEqual(bm25["size"], 30)
        # (2) kNN 서브검색: 캡션 임베딩 벡터 + native pre-filter terms + 의료 must_not + size.
        self.assertEqual(knn["query"]["knn"]["embedding"]["vector"], self.vector)
        knn_bool = knn["query"]["knn"]["embedding"]["filter"]["bool"]
        self.assertIn({"terms": {"modality": [modality_value]}}, knn_bool["filter"])
        self.assertIn({"term": {"domain_label": "medical"}}, knn_bool["must_not"])
        self.assertEqual(knn["size"], 30)

    def test_image_subquery_bodies(self) -> None:
        self._check_modality("image")

    def test_video_subquery_bodies(self) -> None:
        self._check_modality("video")


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
    # 진단: no-match top−baseline ≤0.088·top ≤0.707 / match top−baseline ≥0.116·top ≥0.745.
    # 027: 두 번째 인자는 robust baseline(하위 절반 평균)이다 — 판정식·경계는 023 동일(의미만 이전).
    def test_keep_strong_relative_signal(self) -> None:  # match
        self.assertTrue(passes_cutoff(0.78, 0.66, eps=0.10, floor=0.65))   # top-baseline=0.12, top≥floor

    def test_cut_flat_no_match(self) -> None:  # no-match: 상대신호 부족
        self.assertFalse(passes_cutoff(0.70, 0.65, eps=0.10, floor=0.65))  # top-baseline=0.05 < 0.10

    def test_cut_below_floor_backstop(self) -> None:  # 전체 평평·낮음 → floor 가 차단
        self.assertFalse(passes_cutoff(0.60, 0.30, eps=0.10, floor=0.65))  # top-baseline ok, top<floor

    def test_boundary_inclusive(self) -> None:  # 경계 포함(>=)
        self.assertTrue(passes_cutoff(0.75, 0.65, eps=0.10, floor=0.65))   # =0.10, =0.65

    def test_empty_probe_zero_cut(self) -> None:  # 표본 빈(top=baseline=0) → cut
        self.assertFalse(passes_cutoff(0.0, 0.0, eps=0.10, floor=0.65))


# 027 T006: probe IO seam(_ProbeFakeClient·ProbeRelevanceTest)·probe_fn 주입 게이트 통합 테스트
# (SearchAssetsOsCutoffTest)는 제거됐다 — probe 경로가 소멸(융합이 클라이언트로 와 kNN 응답에 원시
# 코사인 보존)했기 때문. 그 의도(원시 코사인 top·baseline 산출 → 게이트 keep/empty → modality 별 독립)는
# GateSignalTest(top·robust baseline)와 SearchAssetsOsMsearchTest(게이트 keep/cut·meta·결정성)로 이전됐다.


# ──────────────────────────────────────────────────────────────────────────
# 027 G1 — 클라이언트 융합 순수 함수(min-max 정규화·robust 게이트 신호·per-result 컷).
# OS·DB·opensearch-py 불필요. 결정적 단위로 융합 수학을 검증한다(헌법 3조·FR-002·003·004).
# ──────────────────────────────────────────────────────────────────────────


class MinmaxNormalizeTest(unittest.TestCase):
    """027 T002/FR-002: min-max 정규화(순수·결정적). 빈→[], max==min→전원 1.0."""

    def test_empty_returns_empty(self) -> None:
        # 빈 입력 → 빈 출력(결정적 단락 — 0 나눗셈 없음).
        self.assertEqual(minmax_normalize([]), [])

    def test_single_element_is_one(self) -> None:
        # 단일 원소: max==min 퇴화 → 1.0(전원 동일 취급). 0 나눗셈 회피.
        self.assertEqual(minmax_normalize([0.42]), [1.0])

    def test_all_equal_all_ones(self) -> None:
        # 동점(max==min): 전원 1.0 — min-max 의 결정적 퇴화 정의(랭킹 보존·결정성, FR-002).
        self.assertEqual(minmax_normalize([0.3, 0.3, 0.3]), [1.0, 1.0, 1.0])

    def test_scales_to_unit_interval(self) -> None:
        # 일반: (x−min)/(max−min) — min→0.0, max→1.0, 중간은 비례.
        self.assertEqual(minmax_normalize([0.0, 5.0, 10.0]), [0.0, 0.5, 1.0])

    def test_preserves_order_and_length(self) -> None:
        # 정규화는 입력 순서·길이를 보존한다(자리표시 정합 — fuse 가 hit 과 zip).
        out = minmax_normalize([2.0, 8.0, 4.0, 8.0])
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0], 0.0)   # 2 = min
        self.assertEqual(out[1], 1.0)   # 8 = max
        self.assertEqual(out[3], 1.0)   # 8 = max(동점도 1.0)
        self.assertAlmostEqual(out[2], (4.0 - 2.0) / (8.0 - 2.0))


class GateSignalTest(unittest.TestCase):
    """027 T002/FR-003: kNN 코사인 리스트 → (top, robust baseline=하위 절반 평균). <2개면 baseline 0."""

    def test_empty_zero(self) -> None:
        # 빈 표본 → (0.0, 0.0): 게이트가 cut(no-match) 하도록 중립 신호.
        self.assertEqual(gate_signal([]), (0.0, 0.0))

    def test_single_element_baseline_zero(self) -> None:
        # 원소 1개: top 은 그 값, baseline=0.0(하위 절반을 평균낼 표본 부족 — 정의상 0).
        top, baseline = gate_signal([0.8])
        self.assertEqual(top, 0.8)
        self.assertEqual(baseline, 0.0)

    def test_top_is_max(self) -> None:
        # top 은 표본 최댓값(순서 무관).
        top, _ = gate_signal([0.2, 0.9, 0.5, 0.7])
        self.assertEqual(top, 0.9)

    def test_baseline_is_lower_half_mean(self) -> None:
        # robust baseline = 코사인 하위 절반 평균(밀집 토픽에서도 background 해석 유지 — '충전' 해결).
        # 정렬 [0.1,0.2,0.8,0.9] 하위 절반(2개)=[0.1,0.2] → 평균 0.15. 전체 평균(0.5)보다 낮다.
        top, baseline = gate_signal([0.9, 0.1, 0.8, 0.2])
        self.assertEqual(top, 0.9)
        self.assertAlmostEqual(baseline, (0.1 + 0.2) / 2)

    def test_baseline_uses_floor_half_for_odd(self) -> None:
        # 홀수 표본: 하위 절반은 floor(n/2) 개(정렬 하위). [0.1,0.3,0.5,0.7,0.9] → 하위 2개=[0.1,0.3] 평균 0.2.
        _, baseline = gate_signal([0.5, 0.9, 0.1, 0.7, 0.3])
        self.assertAlmostEqual(baseline, (0.1 + 0.3) / 2)

    def test_dense_topic_baseline_below_full_mean(self) -> None:
        # '충전' 시나리오(밀집 토픽): 다수 고코사인 + 소수 저코사인. 하위 절반 baseline 이 전체 평균보다
        # 낮아 top−baseline 이 충분히 벌어진다 → 게이트 통과(전체 평균이면 baseline 이 높아 오컷됐다).
        cosines = [0.85, 0.84, 0.83, 0.82, 0.30, 0.28]
        top, baseline = gate_signal(cosines)
        full_mean = sum(cosines) / len(cosines)
        self.assertEqual(top, 0.85)
        self.assertLess(baseline, full_mean)  # robust baseline < 전체 평균(밀집 편향 흡수)


class CutRowsTest(unittest.TestCase):
    """027 T002/FR-004: per-result 컷 — 행 유지 = _bm25 매칭 OR _cos≥result_floor. _cos None 은 _bm25 필수."""

    def test_keep_bm25_matched_even_without_cos(self) -> None:
        # BM25 매칭(어휘 증거)이면 코사인 없어도(_cos None) 유지 — BM25-only 행 안전(plan R2).
        rows = [{"id": "a", "_bm25": True, "_cos": None}]
        self.assertEqual([r["id"] for r in cut_rows(rows, result_floor=0.4)], ["a"])

    def test_keep_when_cos_at_or_above_floor(self) -> None:
        # BM25 미매칭이어도 원시 코사인 ≥ floor 면 유지(의미 증거).
        rows = [{"id": "a", "_bm25": False, "_cos": 0.5}]
        self.assertEqual([r["id"] for r in cut_rows(rows, result_floor=0.4)], ["a"])

    def test_drop_when_cos_below_floor_and_no_bm25(self) -> None:
        # BM25 미매칭 + 코사인 < floor → 컷(노이즈 꼬리 제거).
        rows = [{"id": "a", "_bm25": False, "_cos": 0.3}]
        self.assertEqual(cut_rows(rows, result_floor=0.4), [])

    def test_drop_cos_none_without_bm25(self) -> None:
        # _cos None + BM25 미매칭 → 코사인 비교 불가이므로 컷(_cos None 은 _bm25 가 유일 근거).
        rows = [{"id": "a", "_bm25": False, "_cos": None}]
        self.assertEqual(cut_rows(rows, result_floor=0.4), [])

    def test_boundary_inclusive(self) -> None:
        # 경계 포용(≥): _cos == floor 면 유지(부동소수 표현오차 허용).
        rows = [{"id": "a", "_bm25": False, "_cos": 0.4}]
        self.assertEqual([r["id"] for r in cut_rows(rows, result_floor=0.4)], ["a"])

    def test_mixed_rows_preserve_input_order(self) -> None:
        # 혼합: 유지 행만 남기되 입력(이미 정렬된) 순서를 보존한다(랭킹 불변).
        rows = [
            {"id": "keep_bm25", "_bm25": True, "_cos": None},
            {"id": "drop", "_bm25": False, "_cos": 0.1},
            {"id": "keep_cos", "_bm25": False, "_cos": 0.9},
        ]
        self.assertEqual([r["id"] for r in cut_rows(rows, result_floor=0.4)], ["keep_bm25", "keep_cos"])


# ──────────────────────────────────────────────────────────────────────────
# 027 G1/T003 — 모달리티당 [BM25 + plain kNN] 서브검색 본문(순수). msearch 의 두 서브검색 body.
# 종전 서버 hybrid 융합 쿼리를 대체한다: 서버 융합(파이프라인)을 쓰지 않고 두 본문을 따로 만들어
# msearch 로 받은 뒤 클라이언트(fuse_hybrid)가 융합한다. build_knn_body 는 게이트 신호용 kNN 본문이다.
# ──────────────────────────────────────────────────────────────────────────


class BuildBm25BodyTest(unittest.TestCase):
    """T003/FR-001 · 044 FR-101: BM25 named query 서브검색 — boost·operator·filter."""

    def setUp(self) -> None:
        self.query = "한국어 검색 질의"
        self.body = build_bm25_body(self.query, modality_values=["txt"], k=20)

    def test_named_query_clauses_and_boosts(self) -> None:
        self.assertEqual(_bm25_named_names(self.body), set(BM25_NAMED_QUERY_NAMES))
        for clause in self.body["query"]["bool"]["should"]:
            if "match" in clause:
                field, inner = next(iter(clause["match"].items()))
                self.assertEqual(inner["query"], self.query)
                self.assertAlmostEqual(inner.get("boost", 1.0), _BM25_FIELD_BOOSTS[field])
            elif "multi_match" in clause:
                inner = clause["multi_match"]
                self.assertEqual(inner["_name"], "hit_cross_meta")
                self.assertEqual(inner["type"], "cross_fields")
                self.assertEqual(inner["fields"], ["summary^3", "keywords^2"])
                self.assertAlmostEqual(inner.get("boost", 1.0), _CROSS_META_BOOST)

    def test_size_equals_k(self) -> None:
        self.assertEqual(self.body["size"], 20)

    def test_modality_terms_filter(self) -> None:
        # modality terms(집합·정렬) 필터 — 라벨↔저장값 불일치 흡수.
        self.assertIn({"terms": {"modality": ["txt"]}}, self.body["query"]["bool"]["filter"])

    def test_text_terms_set_sorted(self) -> None:
        body = build_bm25_body(self.query, modality_values=ALLOWED_TEXT_META_FILE_KINDS, k=10)
        expected = {"terms": {"modality": sorted(ALLOWED_TEXT_META_FILE_KINDS)}}
        self.assertIn(expected, body["query"]["bool"]["filter"])

    def test_exclude_medical_must_not_default(self) -> None:
        # FR-011(헌법 10조): 의료 자산 배제 must_not(bool 래핑).
        self.assertIn(
            {"term": {"domain_label": "medical"}}, self.body["query"]["bool"]["must_not"]
        )

    def test_include_medical_when_disabled(self) -> None:
        body = build_bm25_body(self.query, modality_values=["txt"], k=10, exclude_medical=False)
        self.assertNotIn("must_not", body["query"]["bool"])

    def test_operator_and_requires_all_tokens(self) -> None:
        body = build_bm25_body(self.query, modality_values=["txt"], k=10, operator="and")
        for clause in body["query"]["bool"]["should"]:
            if "match" in clause:
                for inner in clause["match"].values():
                    self.assertEqual(inner.get("operator"), "and")
            elif "multi_match" in clause:
                self.assertEqual(clause["multi_match"].get("operator"), "and")

    def test_operator_default_or_omits_key(self) -> None:
        default_body = build_bm25_body(self.query, modality_values=["txt"], k=10)
        or_body = build_bm25_body(self.query, modality_values=["txt"], k=10, operator="or")
        self.assertEqual(default_body, or_body)
        for clause in default_body["query"]["bool"]["should"]:
            if "match" in clause:
                for inner in clause["match"].values():
                    self.assertNotIn("operator", inner)

    def test_audio_modality_filter(self) -> None:
        body = build_bm25_body(self.query, modality_values=["audio"], k=10)
        self.assertIn({"terms": {"modality": ["audio"]}}, body["query"]["bool"]["filter"])

    def test_search_filters_in_bool_filter(self) -> None:
        filters = SearchFilters(file_exts=("txt",), source_datasets=("wikipedia",))
        body = build_bm25_body(
            self.query,
            modality_values=["txt"],
            k=10,
            search_filters=filters,
        )
        flt = body["query"]["bool"]["filter"]
        self.assertIn({"terms": {"filter_kw.file_ext": ["txt"]}}, flt)
        self.assertIn({"terms": {"filter_kw.source_dataset": ["wikipedia"]}}, flt)


class Bm25MustIncludeExcludeTest(unittest.TestCase):
    """057 FR-202: BM25 서버 lexical 필터 — must_include→bool.must·must_exclude→bool.must_not.

    각 텀은 **기존 cross_fields 텍스트 대상**(summary^3·keywords^2)에 **전토큰(operator=and)** 매칭
    절로 삽입된다. 필터 절(must/must_not)만 추가하고 should(스코어)·filter·minimum_should_match 는
    무변경이라야 랭킹 융합·컷오프·정렬이 불변이다. 미지정(빈)시 body 바이트 동일(하위호환·회귀 0).
    """

    def setUp(self) -> None:
        self.query = "한국어 검색 질의"

    def _cross_and_clause(self, term: str) -> dict:
        # must/must_not 에 들어갈 lexical 절의 정본 형상 — cross_fields·operator=and·기존 필드 대상.
        return {
            "multi_match": {
                "query": term,
                "type": "cross_fields",
                "fields": ["summary^3", "keywords^2"],
                "operator": "and",
            }
        }

    def test_must_include_adds_cross_fields_and_clause_to_must(self) -> None:
        body = build_bm25_body(
            self.query, modality_values=["txt"], k=10, must_include=["충전"]
        )
        must = body["query"]["bool"]["must"]
        self.assertIn(self._cross_and_clause("충전"), must)

    def test_each_include_term_is_its_own_must_clause(self) -> None:
        # 텀 간 AND — 각 텀이 개별 must 절(전 텀이 모두 매칭돼야 결과 유지).
        body = build_bm25_body(
            self.query, modality_values=["txt"], k=10, must_include=["충전", "배터리"]
        )
        must = body["query"]["bool"]["must"]
        self.assertEqual(len(must), 2)
        self.assertIn(self._cross_and_clause("충전"), must)
        self.assertIn(self._cross_and_clause("배터리"), must)

    def test_must_exclude_adds_clause_to_must_not_beside_medical(self) -> None:
        # 의료 must_not 은 유지되고(헌법 10조), 제외 텀 절이 함께 must_not 에 추가된다.
        body = build_bm25_body(
            self.query, modality_values=["txt"], k=10, must_exclude=["광고"]
        )
        must_not = body["query"]["bool"]["must_not"]
        self.assertIn({"term": {"domain_label": "medical"}}, must_not)
        self.assertIn(self._cross_and_clause("광고"), must_not)

    def test_must_exclude_without_medical_when_disabled(self) -> None:
        # exclude_medical=False 여도 제외 텀은 must_not 로 들어간다(의료 절만 빠짐).
        body = build_bm25_body(
            self.query,
            modality_values=["txt"],
            k=10,
            exclude_medical=False,
            must_exclude=["광고"],
        )
        must_not = body["query"]["bool"]["must_not"]
        self.assertNotIn({"term": {"domain_label": "medical"}}, must_not)
        self.assertEqual(must_not, [self._cross_and_clause("광고")])

    def test_empty_filters_body_byte_identical(self) -> None:
        # 하위호환 봉인: must_include/exclude 미지정(빈)이면 질의 body 가 **바이트 동일**해야 한다.
        default_body = build_bm25_body(self.query, modality_values=["txt"], k=20)
        empty_body = build_bm25_body(
            self.query, modality_values=["txt"], k=20, must_include=[], must_exclude=[]
        )
        self.assertEqual(
            json.dumps(default_body, ensure_ascii=False, sort_keys=False),
            json.dumps(empty_body, ensure_ascii=False, sort_keys=False),
        )
        # 빈 필터면 must 키 자체가 없고 must_not 은 의료 절만(기존 형상 봉인).
        self.assertNotIn("must", empty_body["query"]["bool"])
        self.assertEqual(
            empty_body["query"]["bool"]["must_not"],
            [{"term": {"domain_label": "medical"}}],
        )

    def test_blank_terms_ignored_byte_identical(self) -> None:
        # 공백·빈 문자열 텀은 무시(빈 절 생성 금지) → 미지정과 바이트 동일(방어적 정규화).
        default_body = build_bm25_body(self.query, modality_values=["txt"], k=20)
        blank_body = build_bm25_body(
            self.query,
            modality_values=["txt"],
            k=20,
            must_include=["", "   "],
            must_exclude=["  "],
        )
        self.assertEqual(
            json.dumps(default_body, ensure_ascii=False, sort_keys=False),
            json.dumps(blank_body, ensure_ascii=False, sort_keys=False),
        )

    def test_should_and_filter_clauses_unchanged_with_filters(self) -> None:
        # 랭킹 불변: 필터 텀을 넣어도 should(스코어) 절·filter·minimum_should_match 는 무변경.
        default_body = build_bm25_body(self.query, modality_values=["txt"], k=20)
        filtered_body = build_bm25_body(
            self.query,
            modality_values=["txt"],
            k=20,
            must_include=["충전"],
            must_exclude=["광고"],
        )
        self.assertEqual(
            filtered_body["query"]["bool"]["should"],
            default_body["query"]["bool"]["should"],
        )
        self.assertEqual(
            filtered_body["query"]["bool"]["filter"],
            default_body["query"]["bool"]["filter"],
        )
        self.assertEqual(filtered_body["query"]["bool"]["minimum_should_match"], 1)


class BuildKnnBodyTest(unittest.TestCase):
    """T003/FR-001: plain kNN 서브검색 본문(게이트 신호용) — native pre-filter·의료 must_not."""

    def setUp(self) -> None:
        self.vector = [0.1, 0.2, 0.3]

    def test_plain_knn_no_hybrid(self) -> None:
        # plain knn 1개(하이브리드 아님) — embedding 벡터·k·size=k.
        body = build_knn_body(self.vector, modality_values=["image"], k=50)
        self.assertNotIn("hybrid", body["query"])
        knn = body["query"]["knn"]["embedding"]
        self.assertEqual(knn["vector"], self.vector)
        self.assertEqual(knn["k"], 50)
        self.assertEqual(body["size"], 50)

    def test_modality_native_prefilter(self) -> None:
        # G3 교훈 계승: modality 를 **native filter(pre-filter)**로 좁힌다(작은 k 비우세 모달리티 0건 방지).
        body = build_knn_body(self.vector, modality_values=["video", "image"], k=10)
        flt = body["query"]["knn"]["embedding"]["filter"]["bool"]["filter"]
        self.assertIn({"terms": {"modality": ["image", "video"]}}, flt)

    def test_exclude_medical_must_not_default(self) -> None:
        body = build_knn_body(self.vector, modality_values=["text"], k=10)
        self.assertIn(
            {"term": {"domain_label": "medical"}},
            body["query"]["knn"]["embedding"]["filter"]["bool"]["must_not"],
        )

    def test_include_medical_when_disabled(self) -> None:
        body = build_knn_body(self.vector, modality_values=["text"], k=10, exclude_medical=False)
        self.assertNotIn("must_not", body["query"]["knn"]["embedding"]["filter"]["bool"])

    def test_must_include_in_knn_prefilter(self) -> None:
        # 057 FR-202: must_include 를 kNN pre-filter(bool.filter)에도 적용해야 융합에서 kNN 회수분이
        # 필터를 우회하지 않는다(T213 골든에서 실효 없음 발견). BM25 와 동일 _lexical_clause 재사용.
        from src.search.opensearch_search import _lexical_clause

        body = build_knn_body(self.vector, modality_values=["text"], k=10, must_include=["핸드드립"])
        self.assertIn(
            _lexical_clause("핸드드립"),
            body["query"]["knn"]["embedding"]["filter"]["bool"]["filter"],
        )

    def test_must_exclude_in_knn_must_not(self) -> None:
        from src.search.opensearch_search import _lexical_clause

        body = build_knn_body(self.vector, modality_values=["text"], k=10, must_exclude=["에스프레소"])
        self.assertIn(
            _lexical_clause("에스프레소"),
            body["query"]["knn"]["embedding"]["filter"]["bool"]["must_not"],
        )

    def test_knn_body_byte_identical_when_no_lexical(self) -> None:
        # 미지정/빈 must_include·must_exclude 는 kNN body 를 바이트 동일하게 유지(하위호환·회귀 0).
        base = build_knn_body(self.vector, modality_values=["text"], k=10)
        empty = build_knn_body(
            self.vector, modality_values=["text"], k=10, must_include=[], must_exclude=["  "]
        )
        self.assertEqual(json.dumps(base, sort_keys=True), json.dumps(empty, sort_keys=True))

    def test_search_filters_in_knn_prefilter(self) -> None:
        filters = SearchFilters(created_from=date(2026, 1, 1))
        body = build_knn_body(
            self.vector,
            modality_values=["text"],
            k=10,
            search_filters=filters,
        )
        flt = body["query"]["knn"]["embedding"]["filter"]["bool"]["filter"]
        self.assertIn({"range": {"filter_date.created_at": {"gte": "2026-01-01"}}}, flt)


# ──────────────────────────────────────────────────────────────────────────
# 027 G1/T004 — fuse_hybrid: 두 서브검색(BM25·kNN) hit 을 클라이언트에서 융합(합집합·min-max·가중평균).
# 서버 normalization-processor 가 하던 융합을 순수 함수로 이관(헌법 3조). 누락측 정규화 기여 0.
# 행에 _cos(원시 코사인|None)·_bm25(매칭 여부) 동반 → 게이트·per-result 컷의 단일 코사인 스케일 신호.
# ──────────────────────────────────────────────────────────────────────────


class FuseHybridTest(unittest.TestCase):
    """T004/FR-002: 합집합·min-max+가중평균·(-sim,id) 정렬·_cos/_bm25 동반·같은 자산 한 행."""

    def _bm25_hit(self, asset_id: str, score: float) -> dict:
        # BM25 서브검색 hit — _score 는 BM25 원시 점수(스케일 임의).
        return {"_id": asset_id, "_score": score,
                "_source": {"asset_id": asset_id, "modality": "text", "summary": f"요약 {asset_id}"}}

    def _knn_hit(self, asset_id: str, cos: float) -> dict:
        # kNN 서브검색 hit — _score 는 lucene cosinesimil=(1+cos)/2 라 원시 코사인 cos 로 역산된다.
        return {"_id": asset_id, "_score": (1.0 + cos) / 2.0,
                "_source": {"asset_id": asset_id, "modality": "text", "summary": f"요약 {asset_id}"}}

    def test_union_of_both_sides(self) -> None:
        # 합집합: BM25-only·kNN-only·양쪽 자산이 모두 한 번씩 등장(양쪽=한 행).
        bm25 = [self._bm25_hit("both", 10.0), self._bm25_hit("bm_only", 5.0)]
        knn = [self._knn_hit("both", 0.8), self._knn_hit("knn_only", 0.4)]
        out = fuse_hybrid(bm25, knn, weights=(0.5, 0.5))
        self.assertEqual({r["id"] for r in out}, {"both", "bm_only", "knn_only"})
        # 같은 자산('both')은 한 행으로 결합(중복 0).
        self.assertEqual(sum(1 for r in out if r["id"] == "both"), 1)

    def test_weighted_average_with_missing_side_zero(self) -> None:
        # similarity = w_b*norm_bm25 + w_k*norm_knn. 누락측 정규화 기여 0(서버 hybrid 와 동일 시맨틱).
        # bm25 norm: both=1.0(max), bm_only=0.0(min). knn norm: both=1.0(max), knn_only=0.0(min).
        bm25 = [self._bm25_hit("both", 10.0), self._bm25_hit("bm_only", 5.0)]
        knn = [self._knn_hit("both", 0.8), self._knn_hit("knn_only", 0.4)]
        out = {r["id"]: r for r in fuse_hybrid(bm25, knn, weights=(0.5, 0.5))}
        self.assertAlmostEqual(out["both"]["similarity"], 0.5 * 1.0 + 0.5 * 1.0)
        self.assertAlmostEqual(out["bm_only"]["similarity"], 0.5 * 0.0 + 0.5 * 0.0)   # knn 누락측 0
        self.assertAlmostEqual(out["knn_only"]["similarity"], 0.5 * 0.0 + 0.5 * 0.0)  # bm25 누락측 0

    def test_weights_applied_asymmetrically(self) -> None:
        # 가중치 (w_bm25, w_knn) 비대칭 적용 확인 — knn 쪽 가중을 키우면 knn-only 가 bm25-only 보다 높다.
        bm25 = [self._bm25_hit("bm_top", 10.0), self._bm25_hit("bm_only", 0.0)]
        knn = [self._knn_hit("knn_top", 0.9), self._knn_hit("knn_only2", -1.0)]
        out = {r["id"]: r for r in fuse_hybrid(bm25, knn, weights=(0.2, 0.8))}
        self.assertAlmostEqual(out["bm_top"]["similarity"], 0.2 * 1.0)   # bm25 max·knn 누락
        self.assertAlmostEqual(out["knn_top"]["similarity"], 0.8 * 1.0)  # knn max·bm25 누락

    def test_cos_and_bm25_flags_on_rows(self) -> None:
        # 각 행에 _cos(원시 코사인|None)·_bm25(매칭 여부) 동반 — 게이트·per-result 컷 신호.
        bm25 = [self._bm25_hit("both", 10.0), self._bm25_hit("bm_only", 5.0)]
        knn = [self._knn_hit("both", 0.8), self._knn_hit("knn_only", 0.4)]
        out = {r["id"]: r for r in fuse_hybrid(bm25, knn, weights=(0.5, 0.5))}
        # 양쪽 행: _bm25 True + _cos = 원시 코사인.
        self.assertTrue(out["both"]["_bm25"])
        self.assertAlmostEqual(out["both"]["_cos"], 0.8)
        # BM25-only 행: _bm25 True, _cos None(kNN 부재 → 코사인 없음, plan R2).
        self.assertTrue(out["bm_only"]["_bm25"])
        self.assertIsNone(out["bm_only"]["_cos"])
        # kNN-only 행: _bm25 False, _cos = 원시 코사인.
        self.assertFalse(out["knn_only"]["_bm25"])
        self.assertAlmostEqual(out["knn_only"]["_cos"], 0.4)

    def test_row_shape_homogeneous_with_media_search(self) -> None:
        # SC-005: 행은 os_hit_to_row 동형 키 + 내부키(_cos·_bm25·_rrtext — 응답 전 전부 제거).
        out = fuse_hybrid([self._bm25_hit("a", 1.0)], [], weights=(0.5, 0.5))
        row = out[0]
        self.assertEqual(
            set(row),
            {"id", "file_uri", "modality", "domain_label", "summary", "similarity",
             "topics", "subtopics", "_cos", "_bm25", "_rrtext"},  # 057-후속: topics/subtopics 통과
        )
        self.assertEqual(row["modality"], "text")

    def test_sorted_by_similarity_desc_then_id_asc(self) -> None:
        # (-similarity, id) 결정 정렬: 점수 desc, 동점은 id asc(FR-002 tiebreaker).
        bm25 = [self._bm25_hit("high", 10.0), self._bm25_hit("b", 5.0), self._bm25_hit("a", 5.0)]
        out = fuse_hybrid(bm25, [], weights=(1.0, 0.0))
        # bm25 norm: high=1.0, b=0.0, a=0.0 → high 먼저, 동점(b,a)은 id asc → a, b.
        self.assertEqual([r["id"] for r in out], ["high", "a", "b"])

    def test_all_tie_deterministic_id_order(self) -> None:
        # 전부 동점(전 점수 동일 → min-max 전원 1.0): similarity 동일 → id asc 결정.
        bm25 = [self._bm25_hit("c", 7.0), self._bm25_hit("a", 7.0), self._bm25_hit("b", 7.0)]
        knn = [self._knn_hit("c", 0.5), self._knn_hit("a", 0.5), self._knn_hit("b", 0.5)]
        out = fuse_hybrid(bm25, knn, weights=(0.5, 0.5))
        self.assertEqual([r["id"] for r in out], ["a", "b", "c"])
        for r in out:
            self.assertAlmostEqual(r["similarity"], 1.0)  # 0.5*1.0 + 0.5*1.0

    def test_empty_inputs_empty_output(self) -> None:
        self.assertEqual(fuse_hybrid([], [], weights=(0.5, 0.5)), [])

    def test_knn_only_side(self) -> None:
        # BM25 측이 비어도 kNN-only 융합이 동작(누락 bm25 기여 0, _bm25 False).
        knn = [self._knn_hit("k1", 0.9), self._knn_hit("k2", 0.1)]
        out = fuse_hybrid([], knn, weights=(0.5, 0.5))
        ids = [r["id"] for r in out]
        self.assertEqual(set(ids), {"k1", "k2"})
        top = next(r for r in out if r["id"] == "k1")
        self.assertFalse(top["_bm25"])
        self.assertAlmostEqual(top["similarity"], 0.5 * 1.0)  # knn norm max=1.0, bm25 누락 0


class RerankReorderTest(unittest.TestCase):
    """T001/FR-002(G3 정정): cut_rows 생존자를 cross-encoder 로 **재정렬**(드롭 0·융합점수 집합 보존).

    초기 029 는 리랭크가 선별까지(τ 드롭·R 절삭) 맡았으나 G3 실OS 측정에서 recall 0.94→0.90·차단
    23→22 회귀 → 선별은 cut_rows 가 맡고 리랭크는 **순서만** 바꾸도록 정정(rerank_reorder).
    """

    @staticmethod
    def _row(asset_id: str, sim: float, *, rrtext: str | None = None) -> dict:
        # 융합 행 동형(id·similarity·_rrtext). _rrtext 없으면 summary 폴백 경로 확인용.
        r = {"id": asset_id, "similarity": sim, "summary": f"요약 {asset_id}", "_cos": 0.5, "_bm25": False}
        if rrtext is not None:
            r["_rrtext"] = rrtext
        return r

    def test_scores_only_head_tail_kept_after(self) -> None:
        # head=rows[:top_r] 만 채점, tail 은 융합 순서 유지하며 뒤에 붙는다(드롭 0 — R 절삭 안 함).
        rows = [self._row(c, s) for c, s in (("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6))]
        seen = []
        def fake(query, texts, *, model_name):
            seen.extend(texts)
            return [0.5] * len(texts)
        kept, scores = rerank_reorder(rows, "질의", fake, top_r=2, model_name="가짜")
        self.assertEqual(seen, ["요약 a", "요약 b"])              # head 2건만 채점
        self.assertEqual(len(scores), 2)
        self.assertEqual([r["id"] for r in kept], ["a", "b", "c", "d"])  # 드롭 0(head 동점→id·tail 보존)

    def test_reorders_head_preserving_fusion_sim_set(self) -> None:
        # 리랭크 점수 순으로 head 재정렬하되 융합 similarity 값 집합은 보존(best-reranked=최고 융합점수).
        # → union 기여 분포 불변(recall) + 순서만 리랭크(p@3). 029 정정의 핵심 불변식.
        rows = [self._row("a", 0.9), self._row("b", 0.7), self._row("c", 0.5)]
        def fake(query, texts, *, model_name):
            table = {"요약 a": 0.1, "요약 b": 0.9, "요약 c": 0.5}
            return [table[t] for t in texts]
        kept, _ = rerank_reorder(rows, "질의", fake, top_r=10, model_name="가짜")
        self.assertEqual([r["id"] for r in kept], ["b", "c", "a"])              # 리랭크 순서
        self.assertEqual(sorted(r["similarity"] for r in kept), [0.5, 0.7, 0.9])  # 융합점수 집합 보존
        self.assertAlmostEqual(kept[0]["similarity"], 0.9)                     # best-reranked=최고 융합점수

    def test_tail_keeps_fusion_order(self) -> None:
        # head 만 재정렬·tail(R 밖)은 융합 순서 그대로 뒤에(recall@20 의 11~20위 보존).
        rows = [self._row(c, s) for c, s in (("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5))]
        def fake(query, texts, *, model_name):
            return [{"요약 a": 0.1, "요약 b": 0.9}[t] for t in texts]
        kept, _ = rerank_reorder(rows, "질의", fake, top_r=2, model_name="가짜")
        self.assertEqual([r["id"] for r in kept], ["b", "a", "c", "d", "e"])

    def test_default_tau_zero_no_drop(self) -> None:
        # 기본 τ=0 — 낮은 rerank 점수도 드롭하지 않는다(recall 보존, G3 정정 핵심).
        rows = [self._row(c, 0.5) for c in ("a", "b")]
        def fake(query, texts, *, model_name):
            return [{"요약 a": 0.9, "요약 b": 0.0001}[t] for t in texts]
        kept, _ = rerank_reorder(rows, "질의", fake, top_r=10, model_name="가짜")
        self.assertEqual({r["id"] for r in kept}, {"a", "b"})     # 둘 다 유지

    def test_optional_tau_drops_below_threshold(self) -> None:
        # τ>0 은 **선택적** 정밀 드롭(운영 옵트인). 기본은 재정렬만이나 더 엄격한 정밀이 필요할 때.
        rows = [self._row(c, 0.5) for c in ("a", "b", "c")]
        def fake(query, texts, *, model_name):
            return [{"요약 a": 0.9, "요약 b": 0.01, "요약 c": 0.6}[t] for t in texts]
        kept, _ = rerank_reorder(rows, "질의", fake, top_r=10, tau=0.05, model_name="가짜")
        self.assertEqual([r["id"] for r in kept], ["a", "c"])     # b(0.01<τ) 드롭·점수 내림차순

    def test_prefers_rrtext_over_summary(self) -> None:
        # 채점 문서측은 _rrtext("요약:…\n키워드:…") 우선, 없으면 summary 폴백(028 구조화 입력 계승).
        rows = [self._row("a", 0.5, rrtext="요약: A\n키워드: 키1"), self._row("b", 0.5)]
        seen = []
        def fake(query, texts, *, model_name):
            seen.extend(texts)
            return [0.9] * len(texts)
        rerank_reorder(rows, "질의", fake, top_r=10, model_name="가짜")
        self.assertEqual(seen, ["요약: A\n키워드: 키1", "요약 b"])

    def test_empty_input_returns_empty(self) -> None:
        # 빈 입력 → ([], []) (rerank_fn 미호출 안전).
        def boom(query, texts, *, model_name):
            raise AssertionError("빈 입력엔 채점하지 않아야 한다")
        self.assertEqual(rerank_reorder([], "질의", boom, top_r=10, model_name="가짜"), ([], []))

    def test_tie_break_by_id_ascending(self) -> None:
        # 동점 rerank 점수는 id asc 로 결정적 정렬(헌법 3조).
        rows = [self._row(c, 0.5) for c in ("c", "a", "b")]
        def fake(query, texts, *, model_name):
            return [0.7] * len(texts)  # 전 동점
        kept, _ = rerank_reorder(rows, "질의", fake, top_r=10, model_name="가짜")
        self.assertEqual([r["id"] for r in kept], ["a", "b", "c"])


class NormalizeQueryTest(unittest.TestCase):
    """T003/FR-004: LLM 질의 명사구 정규화 순수 토글 함수(off=바이트 동일 passthrough·llm_fn 주입 seam)."""

    def test_disabled_is_byte_identical_passthrough(self) -> None:
        # off(기본): 원문 그대로 — llm_fn 미호출(바이트 동일·027 회귀 0).
        def boom(q):
            raise AssertionError("off 면 llm_fn 을 호출하지 않아야 한다")
        self.assertEqual(normalize_query("별 보는 방법", enabled=False, llm_fn=boom), "별 보는 방법")

    def test_enabled_calls_llm_fn_for_noun_phrase(self) -> None:
        # on: 검색 직전 질의를 llm_fn 으로 명사구 정규화(가짜 주입 — 네트워크 0).
        def fake(q):
            return {"별 보는 방법": "천체 관측"}[q]
        self.assertEqual(normalize_query("별 보는 방법", enabled=True, llm_fn=fake), "천체 관측")

    def test_deterministic_same_query_same_phrase(self) -> None:
        # SC-003 결정성: 같은 질의 → 같은 명사구(순수 — llm_fn 이 결정적이면 함수도 결정적).
        def fake(q):
            return "천체 관측"
        out = [normalize_query("별 보는 방법", enabled=True, llm_fn=fake) for _ in range(3)]
        self.assertEqual(out, ["천체 관측"] * 3)

    def test_empty_and_none_safe(self) -> None:
        # 빈/None 질의 안전: 정규화할 내용 없음 → 원문 그대로(llm_fn 미호출).
        def boom(q):
            raise AssertionError("빈/None 질의엔 llm_fn 을 호출하지 않아야 한다")
        self.assertEqual(normalize_query("", enabled=True, llm_fn=boom), "")
        self.assertIsNone(normalize_query(None, enabled=True, llm_fn=boom))


# ──────────────────────────────────────────────────────────────────────────
# 027 G2/T005 — search_assets_os msearch 재구성. 가짜 msearch 클라이언트 주입(OS 불필요).
# 모달리티당 [knn, bm25] 서브검색을 전 모달리티 _msearch 1회로 → 클라이언트 융합 → 게이트 → per-result
# 컷 → (buckets, gate_meta) 튜플. 실 OS 의 msearch 응답·스케일은 G4(실OS e2e)에서 확정.
# ──────────────────────────────────────────────────────────────────────────


def _knn_hit_os(asset_id: str, cos: float, modality: str = "text") -> dict:
    # kNN 서브검색 hit — _score = lucene cosinesimil = (1+cos)/2 (원시 코사인 cos 로 역산됨).
    return {"_id": asset_id, "_score": (1.0 + cos) / 2.0,
            "_source": {"asset_id": asset_id, "modality": modality, "summary": f"요약 {asset_id}"}}


def _bm25_hit_os(asset_id: str, score: float, modality: str = "text") -> dict:
    return {"_id": asset_id, "_score": score,
            "_source": {"asset_id": asset_id, "modality": modality, "summary": f"요약 {asset_id}"}}


def _sub_terms_label(sub: dict) -> tuple[str, str]:
    """서브검색 본문에서 (kind, modality 라벨)을 뽑는다 — knn=native filter, bm25=bool.filter."""
    q = sub["query"]
    if "knn" in q:
        terms = q["knn"]["embedding"]["filter"]["bool"]["filter"][0]["terms"]["modality"]
        kind = "knn"
    else:
        terms = q["bool"]["filter"][0]["terms"]["modality"]
        kind = "bm25"
    if "audio" in terms:
        label = "audio"
    elif "video" in terms:
        label = "video"
    elif "image" in terms:
        label = "image"
    else:
        label = "text"
    return kind, label


class _FakeMsearchClient:
    """opensearch-py ``client.msearch(body=[header, sub, ...])`` 대역.

    body 는 [헤더, 서브검색본문, 헤더, 서브검색본문, …] 교대 리스트다(서브검색은 홀수 인덱스).
    각 서브검색을 (kind=knn|bm25, modality) 로 분류해 canned hits 를 ``{"responses": [...]}`` 로
    돌려준다. ``raise_on_msearch`` 면 그 예외를 던져 OS 미도달(FR-007)을 흉내낸다.
    """

    def __init__(self, *, knn_by_label=None, bm25_by_label=None, raise_on_msearch=None) -> None:
        self.knn_by_label = knn_by_label or {}
        self.bm25_by_label = bm25_by_label or {}
        self._raise = raise_on_msearch
        self.msearch_calls: list[list] = []

    def msearch(self, *, body=None, **_kw):
        self.msearch_calls.append(body)
        if self._raise is not None:
            raise self._raise
        responses = []
        for sub in body[1::2]:  # 서브검색 본문(헤더 다음 홀수 인덱스)
            kind, label = _sub_terms_label(sub)
            src = self.knn_by_label if kind == "knn" else self.bm25_by_label
            responses.append({"hits": {"hits": list(src.get(label, []))}})
        return {"responses": responses}


class SearchAssetsOsMsearchTest(unittest.TestCase):
    """T005/FR-001·002·003·004·007: msearch 1회 → 융합 → 게이트 → 컷 → (buckets, gate_meta)."""

    def setUp(self) -> None:
        self.query = "한국어 검색 질의"
        self.vector = [0.11, 0.22, 0.33]
        self.embed_calls: list[tuple[str, str]] = []

        def fake_embed(q: str, *, channel: str) -> list[float]:
            self.embed_calls.append((q, channel))
            return list(self.vector)

        self.fake_embed = fake_embed
        # text: 강한 신호(top 0.85·하위 절반 낮음 → 게이트 통과). audio: 단일 강신호.
        self.client = _FakeMsearchClient(
            knn_by_label={
                "text": [_knn_hit_os("t1", 0.85), _knn_hit_os("t2", 0.80),
                         _knn_hit_os("t3", 0.30), _knn_hit_os("t4", 0.28)],
                "audio": [_knn_hit_os("a1", 0.82, "audio"), _knn_hit_os("a2", 0.25, "audio")],
            },
            bm25_by_label={
                "text": [_bm25_hit_os("t1", 12.0), _bm25_hit_os("t2", 6.0)],
                "audio": [_bm25_hit_os("a1", 9.0, "audio")],
            },
        )

    def _run(self, **ov):
        kw = {
            "modalities": ("text", "audio"), "k": 20, "channel": "st",
            "weights": (0.5, 0.5), "index": "assets", "embed_fn": self.fake_embed,
            "cutoff_enabled": True, "cutoff_eps": 0.17, "cutoff_floor": 0.50,
            "result_floor": 0.40, "bm25_operator": "or",
        }
        kw.update(ov)
        return search_assets_os(self.client, self.query, **kw)

    def test_embeds_query_once(self) -> None:
        # (a) 질의 1회만 임베딩해 전 모달리티 재사용(중복 임베딩 0) — 활성 채널 전달.
        self._run()
        self.assertEqual(self.embed_calls, [(self.query, "st")])

    def test_single_msearch_call(self) -> None:
        # FR-001: 전 모달리티 서브검색을 _msearch **1회**로(HTTP 1회). client.search 미사용.
        self._run()
        self.assertEqual(len(self.client.msearch_calls), 1)
        self.assertFalse(hasattr(self.client, "search_calls"))

    def test_msearch_body_order_knn_then_bm25_per_modality(self) -> None:
        # 본문 순서 = [m1-헤더, m1-knn, m1-헤더, m1-bm25, m2-헤더, m2-knn, …] 결정적.
        self._run(modalities=("text", "audio"))
        body = self.client.msearch_calls[0]
        # 헤더는 짝수 인덱스(index 지정), 서브검색은 홀수 인덱스.
        self.assertEqual(len(body), 8)  # 2 모달리티 × (헤더+knn+헤더+bm25)
        for header in body[0::2]:
            self.assertEqual(header, {"index": "assets"})
        kinds = [_sub_terms_label(sub) for sub in body[1::2]]
        self.assertEqual(kinds, [("knn", "text"), ("bm25", "text"), ("knn", "audio"), ("bm25", "audio")])

    def test_must_include_exclude_applied_to_bm25_body(self) -> None:
        # 057 FR-202: search_assets_os 가 must_include/exclude 를 BM25 서브검색 본문에 배선한다.
        self._run(modalities=("text",), must_include=["충전"], must_exclude=["광고"])
        body = self.client.msearch_calls[0]
        bm25_sub = body[3]  # [헤더, knn, 헤더, bm25] → bm25 는 인덱스 3.
        clause = lambda t: {  # noqa: E731 — 테스트 내 짧은 지역 헬퍼
            "multi_match": {
                "query": t, "type": "cross_fields",
                "fields": ["summary^3", "keywords^2"], "operator": "and",
            }
        }
        self.assertIn(clause("충전"), bm25_sub["query"]["bool"]["must"])
        self.assertIn(clause("광고"), bm25_sub["query"]["bool"]["must_not"])

    def test_must_include_exclude_applied_to_knn_prefilter(self) -> None:
        # 057 FR-202: kNN 서브검색 pre-filter 에도 배선한다 — 융합(BM25∪kNN) 우회 방지(T213 골든에서
        # BM25-only 필터 실효 없음 발견). must_include→knn filter, must_exclude→knn must_not.
        self._run(modalities=("text",), must_include=["충전"], must_exclude=["광고"])
        knn_sub = self.client.msearch_calls[0][1]  # [헤더, knn, 헤더, bm25] → knn 은 인덱스 1.
        clause = lambda t: {  # noqa: E731
            "multi_match": {
                "query": t, "type": "cross_fields",
                "fields": ["summary^3", "keywords^2"], "operator": "and",
            }
        }
        knn_bool = knn_sub["query"]["knn"]["embedding"]["filter"]["bool"]
        self.assertIn(clause("충전"), knn_bool["filter"])
        self.assertIn(clause("광고"), knn_bool["must_not"])

    def test_no_lexical_filters_bm25_body_has_no_must(self) -> None:
        # 미지정(기본)이면 BM25 본문에 must 키가 없다(하위호환·회귀 0).
        self._run(modalities=("text",))
        bm25_sub = self.client.msearch_calls[0][3]
        self.assertNotIn("must", bm25_sub["query"]["bool"])

    def test_knn_k_is_max_request_and_sample(self) -> None:
        # kNN k = max(요청 k, OS_KNN_SAMPLE_K) — robust baseline 표본 하한. bm25 k = 요청 k.
        from src.config.search_constants import OS_KNN_SAMPLE_K
        self._run(k=5)  # 요청 5 < 표본하한
        body = self.client.msearch_calls[0]
        knn_sub, bm25_sub = body[1], body[3]
        self.assertEqual(knn_sub["query"]["knn"]["embedding"]["k"], OS_KNN_SAMPLE_K)
        self.assertEqual(knn_sub["size"], OS_KNN_SAMPLE_K)
        self.assertEqual(bm25_sub["size"], 5)  # bm25 는 요청 k

    def test_knn_k_uses_request_when_larger(self) -> None:
        from src.config.search_constants import OS_KNN_SAMPLE_K
        self._run(k=OS_KNN_SAMPLE_K + 30)
        knn_sub = self.client.msearch_calls[0][1]
        self.assertEqual(knn_sub["query"]["knn"]["embedding"]["k"], OS_KNN_SAMPLE_K + 30)

    def test_returns_buckets_and_gate_meta_tuple(self) -> None:
        # 반환은 (buckets, gate_meta) 튜플. buckets[modality]=행 리스트, gate_meta[modality]=신호 dict.
        result = self._run()
        self.assertIsInstance(result, tuple)
        buckets, gate_meta = result
        self.assertEqual(set(buckets), {"text", "audio"})
        self.assertEqual(set(gate_meta), {"text", "audio"})

    def test_buckets_fused_and_stripped_internal_keys(self) -> None:
        # 융합 결과가 버킷에 담기되 내부키(_cos·_bm25)는 제거된다(응답 동형 SC-005).
        buckets, _ = self._run()
        text = buckets["text"]
        self.assertEqual([r["id"] for r in text][:2], ["t1", "t2"])  # 융합 점수 desc
        for r in text:
            self.assertNotIn("_cos", r)
            self.assertNotIn("_bm25", r)
            self.assertEqual(
                set(r),
                {"id", "file_uri", "modality", "domain_label", "summary", "similarity",
                 "topics", "subtopics"},  # 057-후속: topics/subtopics 통과(내부키만 strip)
            )

    def test_gate_pass_keeps_bucket(self) -> None:
        # 강한 신호(top 0.85·robust baseline 낮음) → 게이트 통과·버킷 유지, gate_passed True.
        buckets, gate_meta = self._run()
        self.assertTrue(gate_meta["text"]["gate_passed"])
        self.assertGreater(len(buckets["text"]), 0)
        self.assertAlmostEqual(gate_meta["text"]["top"], 0.85)
        # robust baseline = 하위 절반 평균([0.28,0.30]) = 0.29.
        self.assertAlmostEqual(gate_meta["text"]["baseline"], (0.28 + 0.30) / 2)

    def test_gate_fail_empties_bucket_but_records_meta(self) -> None:
        # 약한 신호(평탄·낮음)·어휘 증거 없음 → 빈 버킷 + meta 기록(F4 관측성: no-match 사유 보존).
        # (어휘 증거가 있으면 구제 규칙 적용 — test_lexical_rescue_* 가 그 경로를 검증.)
        client = _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os("x1", 0.40), _knn_hit_os("x2", 0.39),
                                   _knn_hit_os("x3", 0.38), _knn_hit_os("x4", 0.37)]},
            bm25_by_label={"text": []},
        )
        buckets, gate_meta = search_assets_os(
            client, self.query, modalities=("text",), index="assets", embed_fn=self.fake_embed,
            cutoff_enabled=True, cutoff_eps=0.17, cutoff_floor=0.50, result_floor=0.40,
        )
        self.assertEqual(buckets["text"], [])
        self.assertFalse(gate_meta["text"]["gate_passed"])
        self.assertIn("top", gate_meta["text"])
        self.assertIn("baseline", gate_meta["text"])
        self.assertGreater(gate_meta["text"]["cut_count"], 0)  # 컷된 행 수 관측

    def test_per_result_cut_keeps_bm25_or_high_cos(self) -> None:
        # per-result 컷: 게이트 통과 버킷 안에서도 행 유지 = BM25 매칭 OR cos≥result_floor.
        # n1: knn-only cos 0.30(<floor 0.40)·BM25 미매칭 → 컷. n2: knn-only cos 0.80 → 유지.
        # n3: BM25 매칭(코사인 무관) → 유지.
        client = _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os("n2", 0.80), _knn_hit_os("n1", 0.30),
                                   _knn_hit_os("seed", 0.85), _knn_hit_os("low", 0.20)]},
            bm25_by_label={"text": [_bm25_hit_os("n3", 8.0), _bm25_hit_os("seed", 4.0)]},
        )
        buckets, gate_meta = search_assets_os(
            client, self.query, modalities=("text",), index="assets", embed_fn=self.fake_embed,
            cutoff_enabled=True, cutoff_eps=0.17, cutoff_floor=0.50, result_floor=0.40,
        )
        kept_ids = {r["id"] for r in buckets["text"]}
        self.assertIn("n2", kept_ids)     # cos 0.80 ≥ floor
        self.assertIn("n3", kept_ids)     # BM25 매칭
        self.assertIn("seed", kept_ids)   # 양쪽
        self.assertNotIn("n1", kept_ids)  # cos<floor·BM25 미매칭 → 컷
        self.assertNotIn("low", kept_ids)

    def test_rerank_scores_search_text_not_summary(self) -> None:
        # 028 입력 검증(사용자 지적 후속): 채점 문서측은 summary 단독이 아니라 **구조화
        # 텍스트("요약: …\n키워드: …")** — 입력 변형 4종 실측에서 최선(회식 0.003→0.071).
        # nori 토큰 나열은 패러프레이즈 붕괴로 기각(자연어 문맥 파괴 — ADR 기록).
        hit={"_id":"r1","_score":0.9,"_source":{"asset_id":"r1","modality":"txt",
             "summary":"요약문","keywords":["키워드","라벨"]}}
        client = _FakeMsearchClient(knn_by_label={"text":[hit]}, bm25_by_label={"text":[]})
        seen=[]
        def spy(query, texts, *, model_name):
            seen.extend(texts)
            return [0.9]*len(texts)
        buckets,_=search_assets_os(
            client, "질의", modalities=("text",), index="assets", embed_fn=self.fake_embed,
            rerank_enabled=True, rerank_tau=0.0, rerank_fn=spy,
        )
        self.assertEqual(seen, ["요약: 요약문\n키워드: 키워드 라벨"])  # 구조화 입력(실측 최선)
        self.assertNotIn("_rrtext", buckets["text"][0])  # 내부키 제거

    def test_rerank_augments_gate(self) -> None:
        # 029 행동 변경의 핵심 증인(replace→augment·G3 정정): rerank 는 게이트·컷을 **대체하지 않는다**.
        # (a) 게이트 실패면 rerank_enabled=True 여도 빈 버킷이고 rerank_fn 미호출(게이트가 '없음' 버킷을
        #     먼저 비운다 — 차단 23/24 보존), (b) 게이트 통과면 cut_rows 가 행을 선별(027·recall 보존)한
        #     **뒤** 통과 버킷 생존자를 rerank 가 **재정렬만** 한다(드롭 0 — 기본 τ=0).
        calls: list[list[str]] = []
        def fake_rerank(query, texts, *, model_name):
            calls.append(list(texts))
            table = {"요약 a": 0.9, "요약 b": 0.01, "요약 c": 0.6, "요약 d": 0.02}
            return [table[t] for t in texts]

        # (a) 약신호 + 게이트 확실 실패(eps=0.9/floor=0.9)·어휘 증거 없음 → rerank 미실행·빈 버킷.
        weak = _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os("a", 0.40), _knn_hit_os("b", 0.39),
                                   _knn_hit_os("c", 0.38)]},
            bm25_by_label={"text": []},
        )
        buckets, meta = search_assets_os(
            weak, "질의", modalities=("text",), index="assets", embed_fn=self.fake_embed,
            cutoff_enabled=True, cutoff_eps=0.9, cutoff_floor=0.9,
            rerank_enabled=True, rerank_top_r=10, rerank_tau=0.0,
            rerank_model="가짜", rerank_fn=fake_rerank,
        )
        self.assertEqual(buckets["text"], [])             # 게이트가 먼저 비움(rerank 무관)
        self.assertEqual(calls, [])                        # rerank_fn 미호출(augment — 대체 아님)
        self.assertFalse(meta["text"]["gate_passed"])
        self.assertNotIn("rerank", meta["text"])           # rerank 잎 미부착(게이트 실패)

        # (b) 강신호(top cos 0.90·나머지 cos≥0.55) → 게이트 통과(top−baseline 0.33≥0.17) → cut_rows 가
        #     4행 모두 선별(cos≥0.55) → rerank 가 그 생존자를 재정렬만(드롭 0). _knn_hit_os 2번째 인자=코사인.
        strong = _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os("a", 0.90), _knn_hit_os("b", 0.60),
                                   _knn_hit_os("c", 0.58), _knn_hit_os("d", 0.56)]},
            bm25_by_label={"text": []},
        )
        buckets, meta = search_assets_os(
            strong, "질의", modalities=("text",), index="assets", embed_fn=self.fake_embed,
            cutoff_enabled=True, cutoff_eps=0.17, cutoff_floor=0.50, result_floor=0.55,
            rerank_enabled=True, rerank_top_r=10, rerank_tau=0.0,
            rerank_model="가짜", rerank_fn=fake_rerank,
        )
        rows = buckets["text"]
        # 드롭 0 — cut_rows 생존 4행 전부 유지(τ=0), rerank 점수 순서로 재정렬(a>c>d>b).
        self.assertEqual([r["id"] for r in rows], ["a", "c", "d", "b"])
        self.assertEqual(len(rows), 4)                         # ★ G3 정정: 약-정답 행을 드롭하지 않는다
        self.assertEqual(len(calls), 1)                        # 통과 버킷에서만 rerank 1회 실행
        self.assertTrue(meta["text"]["gate_passed"])
        self.assertTrue(meta["text"]["rerank"]["enabled"])
        self.assertEqual(meta["text"]["rerank"]["scored"], 4)  # cut_rows 생존 4행 채점
        self.assertEqual(meta["text"]["rerank"]["kept"], 4)    # 드롭 0

    def test_rerank_disabled_is_027_identical(self) -> None:
        # 회귀 0: rerank 기본 off 면 027 경로(게이트·컷) 그대로 — rerank_fn 미호출.
        calls=[]
        def boom(query, texts, *, model_name):
            calls.append(1)
            return [1.0]*len(texts)
        client = _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os("t1", 0.85)]},
            bm25_by_label={"text": [_bm25_hit_os("t1", 9.0)]},
        )
        buckets, meta = search_assets_os(
            client, "질의", modalities=("text",), index="assets", embed_fn=self.fake_embed,
            rerank_fn=boom,
        )
        self.assertEqual(calls, [])
        self.assertNotIn("rerank", meta["text"])

    def test_query_norm_on_normalizes_embed_and_bm25(self) -> None:
        # 029 T008: query_norm on 이면 검색 직전 질의를 1회 정규화 → 임베딩·BM25 **양쪽**이 정규화된
        # 질의를 받는다(단일 query 지역변수 공유). 가짜 query_norm_fn 주입(네트워크 0).
        norm_calls: list[str] = []
        def fake_norm(q: str) -> str:
            norm_calls.append(q)
            return "천체 관측"
        buckets, _ = search_assets_os(
            self.client, "별 보는 방법", modalities=("text",), index="assets",
            embed_fn=self.fake_embed, cutoff_enabled=False,
            query_norm_enabled=True, query_norm_fn=fake_norm,
        )
        self.assertEqual(norm_calls, ["별 보는 방법"])               # 정규화 1회
        self.assertEqual(self.embed_calls, [("천체 관측", "st")])    # 임베딩이 정규화된 질의
        bm25_sub = self.client.msearch_calls[0][3]                    # [헤더,knn,헤더,bm25] → bm25
        self.assertEqual(_bm25_first_match_query(bm25_sub), "천체 관측")

    def test_query_norm_off_is_original_passthrough(self) -> None:
        # off(기본): 원문 그대로 — query_norm_fn 미호출(바이트 동일·027 동치).
        def boom(q: str) -> str:
            raise AssertionError("off 면 query_norm_fn 을 호출하지 않아야 한다")
        search_assets_os(
            self.client, "별 보는 방법", modalities=("text",), index="assets",
            embed_fn=self.fake_embed, cutoff_enabled=False,
            query_norm_enabled=False, query_norm_fn=boom,
        )
        self.assertEqual(self.embed_calls, [("별 보는 방법", "st")])  # 원문 임베딩
        bm25_sub = self.client.msearch_calls[0][3]
        self.assertEqual(_bm25_first_match_query(bm25_sub), "별 보는 방법")

    def test_query_norm_default_is_off(self) -> None:
        # 기본값 off(OS_QUERY_NORM_ENABLED_DEFAULT) — query_norm 인자 미전달 시 원문 임베딩(회귀 0).
        self._run(modalities=("text",))
        self.assertEqual(self.embed_calls, [(self.query, "st")])

    def test_query_norm_lazy_imports_noun_phrase_when_fn_none(self) -> None:
        # query_norm_fn 미주입 + enabled → noun_phrase_query 를 지연 import 해 사용(seam 기본·플래그 off
        # 환경에 LLM seam 을 당기지 않음). 패치로 네트워크 없이 호출 경유만 검증.
        from unittest import mock

        import src.search.query_preprocess as qp
        with mock.patch.object(qp, "noun_phrase_query", return_value="천체 관측") as m:
            search_assets_os(
                self.client, "별 보는 방법", modalities=("text",), index="assets",
                embed_fn=self.fake_embed, cutoff_enabled=False, query_norm_enabled=True,
            )
        m.assert_called_once_with("별 보는 방법")
        self.assertEqual(self.embed_calls, [("천체 관측", "st")])

    def test_fusion_uses_topk_knn_rows_not_gate_sample(self) -> None:
        # 융합 입력은 kNN **상위 k행만**(서버 융합 시절의 결과셋 범위와 동치) — 게이트 표본(50)을
        # 그대로 정규화에 쓰면 분모가 넓어져 상위 경계 순위가 흔들린다(골든 실측 -0.008 회귀 교정).
        # k=2 요청: kNN 표본엔 4행이 와도 융합·정규화는 상위 2행(n1·n2) 기준이어야 한다 —
        # n2 의 정규화 코사인이 (표본 최솟값이 아니라) 상위 2행 내 최솟값으로 0.0 이 되는지로 식별.
        client = _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os("n1", 0.90), _knn_hit_os("n2", 0.80),
                                   _knn_hit_os("n3", 0.40), _knn_hit_os("n4", 0.30)]},
            bm25_by_label={"text": []},
        )
        buckets, _meta = search_assets_os(
            client, self.query, modalities=("text",), k=2, index="assets",
            embed_fn=self.fake_embed, cutoff_enabled=False,
        )
        rows = buckets["text"]
        self.assertEqual([r["id"] for r in rows], ["n1", "n2"])
        # 상위 2행 기준 min-max: n1=1.0, n2=0.0 → 가중 0.5 적용 시 similarity 0.5·0.0.
        self.assertAlmostEqual(rows[0]["similarity"], 0.5)
        self.assertAlmostEqual(rows[1]["similarity"], 0.0)

    def test_lexical_rescue_requires_and_operator(self) -> None:
        # 구제 전제의 코드 강제(리뷰 후속): operator='or' 매칭은 단일 토큰 우연 일치도 _bm25=True 라
        # 증거 강도가 없다 — 'and' 가 아니면 게이트 실패 버킷은 구제 없이 빈 버킷이어야 한다.
        client = _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os("n1", 0.70)]},
            bm25_by_label={"text": [_bm25_hit_os("b1", 3.2)]},
        )
        buckets, meta = search_assets_os(
            client, "질의", modalities=("text",), index="assets",
            embed_fn=self.fake_embed, cutoff_enabled=True,
            cutoff_eps=0.9, cutoff_floor=0.9, result_floor=0.99,
            bm25_operator="or",  # 전제 불충족 → 구제 미적용
        )
        self.assertEqual(buckets["text"], [])
        self.assertTrue(meta["text"]["lexical_evidence"])  # 증거는 있었으나 전제 미충족

    def test_partial_failure_marked_in_meta(self) -> None:
        # msearch 부분 실패(서브검색 error 객체)는 빈 버킷(no-match)과 구분돼야 한다(FR-007) —
        # gate_meta["error"]=True 로 표식(운영자가 '오류로 빈 것'을 식별).
        class _PartialFailClient:
            def msearch(self, *, body=None, **_kw):
                # [knn 응답=error, bm25 응답=정상 빈] 1개 모달리티 분량
                return {"responses": [{"error": {"type": "search_phase_execution_exception"}},
                                      {"hits": {"hits": []}}]}
        buckets, meta = search_assets_os(
            _PartialFailClient(), "질의", modalities=("text",), index="assets",
            embed_fn=self.fake_embed, cutoff_enabled=True,
        )
        self.assertEqual(buckets["text"], [])
        self.assertTrue(meta["text"]["error"])

    def test_lexical_rescue_keeps_only_bm25_rows_when_gate_fails(self) -> None:
        # 어휘 구제(027 정밀화): 코사인 게이트가 실패해도 BM25(전 토큰 어휘) 증거 행이 있으면
        # 버킷을 통째로 죽이지 않는다 — 단 구제 시엔 **어휘 증거 행만** 남긴다(의미-노이즈 배제).
        # 근거: 약한-있음 토픽(예: '회식' — 자산 4개·코사인 약함·어휘 정확 매칭)이 버킷 통계로는
        # 인접-없음(자동차보험)보다 약해 어떤 임계로도 못 가르는 역전을 행 단위 증거로 해소.
        client = _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os("n1", 0.70), _knn_hit_os("n2", 0.69)]},  # cos 0.40·0.38 — 약함
            bm25_by_label={"text": [_bm25_hit_os("b1", 3.2)]},  # 어휘 증거
        )
        buckets, meta = search_assets_os(
            client, "회식 술자리", modalities=("text",), index="assets",
            embed_fn=self.fake_embed, cutoff_enabled=True,
            cutoff_eps=0.9, cutoff_floor=0.9,  # 게이트 확실 실패
            result_floor=0.99,  # cos 행은 전부 컷 — 어휘 행만 생존해야 함
        )
        self.assertEqual([r["id"] for r in buckets["text"]], ["b1"])  # 어휘 증거 행만
        self.assertFalse(meta["text"]["gate_passed"])
        self.assertTrue(meta["text"]["lexical_evidence"])

    def test_no_lexical_no_gate_means_empty(self) -> None:
        # 어휘 증거도 없고 게이트도 실패 → 빈 버킷(no-match 차단 불변).
        client = _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os("n1", 0.70)]}, bm25_by_label={"text": []}
        )
        buckets, meta = search_assets_os(
            client, "아이패드", modalities=("text",), index="assets",
            embed_fn=self.fake_embed, cutoff_enabled=True,
            cutoff_eps=0.9, cutoff_floor=0.9, result_floor=0.0,
        )
        self.assertEqual(buckets["text"], [])
        self.assertFalse(meta["text"]["lexical_evidence"])

    def test_cutoff_disabled_no_gate_no_cut(self) -> None:
        # 디버그 우회(cutoff_enabled=False): 게이트·per-result 컷 모두 off → 융합 전체 노출, gate_passed True.
        buckets, gate_meta = self._run(cutoff_enabled=False)
        # text 융합 합집합: t1·t2(양쪽) + t3·t4(knn-only, 저코사인)도 컷 없이 유지.
        ids = {r["id"] for r in buckets["text"]}
        self.assertEqual(ids, {"t1", "t2", "t3", "t4"})
        self.assertEqual(gate_meta["text"]["cut_count"], 0)
        self.assertTrue(gate_meta["text"]["gate_passed"])

    def test_cutoff_disabled_bypasses_rerank(self) -> None:
        # T004 디버그 우회 × rerank: cutoff_enabled=False 면 rerank_enabled=True 여도 게이트·컷·
        # rerank 를 **모두 off** — 융합 전체 노출·rerank_fn 미호출. 우회를 rerank 보다 우선 두는
        # 우선순위를 코드로 강제한다(현 replace 경로가 rerank 를 cutoff 검사보다 먼저 둬서 숨겼던 모호성 해소).
        calls: list[int] = []
        def boom(query, texts, *, model_name):
            calls.append(1)
            return [1.0] * len(texts)
        buckets, gate_meta = self._run(cutoff_enabled=False, rerank_enabled=True, rerank_fn=boom)
        ids = {r["id"] for r in buckets["text"]}
        self.assertEqual(ids, {"t1", "t2", "t3", "t4"})    # 융합 전체(약 후보 포함)
        self.assertEqual(calls, [])                          # rerank 미실행(우회가 먼저)
        self.assertEqual(gate_meta["text"]["cut_count"], 0)
        self.assertTrue(gate_meta["text"]["gate_passed"])
        self.assertNotIn("rerank", gate_meta["text"])        # rerank 잎 미부착

    def test_rerank_meta_cut_count_is_027_cut(self) -> None:
        # T005 관측성(G3 정정): rerank{enabled,scored,kept,top} 는 gate_passed ∧ rerank 잎에서만 부착.
        # cut_count 는 **cut_rows 의 027 컷**(BM25 OR cos≥floor 미달 제거)으로 센다 — 리랭크는 재정렬만
        # 이라 행을 추가로 컷하지 않는다(기본 τ=0). cos<0.55 인 c·d·e 컷(cut_count=3)·생존 a·b 재정렬(드롭 0).
        def fake_rerank(query, texts, *, model_name):
            return [{"요약 a": 0.9, "요약 b": 0.01}[t] for t in texts]  # 생존 a·b 만 채점
        client = _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os("a", 0.85), _knn_hit_os("b", 0.84),
                                   _knn_hit_os("c", 0.30), _knn_hit_os("d", 0.28),
                                   _knn_hit_os("e", 0.25)]},  # cos: a0.70 b0.68 c-0.40 d-0.44 e-0.50
            bm25_by_label={"text": []},
        )
        buckets, meta = search_assets_os(
            client, "질의", modalities=("text",), index="assets", embed_fn=self.fake_embed,
            cutoff_enabled=True, cutoff_eps=0.17, cutoff_floor=0.50, result_floor=0.55,
            rerank_enabled=True, rerank_top_r=2, rerank_tau=0.0,
            rerank_model="가짜", rerank_fn=fake_rerank,
        )
        self.assertTrue(meta["text"]["gate_passed"])
        self.assertEqual(meta["text"]["rerank"], {"enabled": True, "scored": 2, "kept": 2, "top": 0.9})
        # cut_count 는 cut_rows 가 제거한 c·d·e(cos<0.55) 3행 — rerank 는 행을 컷하지 않는다(드롭 0).
        self.assertEqual(meta["text"]["cut_count"], 3)
        self.assertEqual([r["id"] for r in buckets["text"]], ["a", "b"])

    def test_bucket_capped_to_k(self) -> None:
        # 버킷은 요청 k 로 상한(knn 표본 50 으로 더 받아도 응답은 ≤k — 구 size=k 계약 보존).
        client = _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os(f"a{i}", 0.85 - i * 0.001) for i in range(40)]},
            bm25_by_label={"text": [_bm25_hit_os("a0", 9.0)]},
        )
        buckets, _ = search_assets_os(
            client, self.query, modalities=("text",), k=3, index="assets", embed_fn=self.fake_embed,
            cutoff_enabled=True, cutoff_eps=0.17, cutoff_floor=0.50, result_floor=0.40,
        )
        self.assertLessEqual(len(buckets["text"]), 3)

    def test_deterministic_two_runs_identical(self) -> None:
        # SC-003: 같은 입력 → 같은 출력(2회 동일) — 융합·게이트·컷·정렬이 순수 결정적.
        out1 = self._run()
        out2 = self._run()
        self.assertEqual(out1, out2)

    def test_propagates_os_unreachable(self) -> None:
        # FR-007: client.msearch 예외 → 그대로 전파(silent pg 폴백 금지).
        client = _FakeMsearchClient(raise_on_msearch=ConnectionError("OS 미도달"))
        with self.assertRaises(ConnectionError):
            search_assets_os(client, self.query, modalities=("text",), index="assets",
                             embed_fn=self.fake_embed)

    def test_image_video_label_resolution_and_tiebreaker(self) -> None:
        # 022 계승: image·video 라벨이 terms 로 해소되고, 동점은 (-sim,id) 결정 정렬. cutoff off 로 컷 무관.
        client = _FakeMsearchClient(
            knn_by_label={"image": [_knn_hit_os("b", 0.5, "image"), _knn_hit_os("c", 0.5, "image"),
                                    _knn_hit_os("a", 0.5, "image")]},
            bm25_by_label={},
        )
        buckets, _ = search_assets_os(
            client, self.query, modalities=("image",), index="assets", embed_fn=self.fake_embed,
            cutoff_enabled=False,
        )
        # 전 동점(코사인 동일 → min-max 전원 1.0) → similarity 동일 → id asc.
        self.assertEqual([r["id"] for r in buckets["image"]], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
