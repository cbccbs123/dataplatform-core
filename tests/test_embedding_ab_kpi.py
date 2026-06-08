"""BGE-M3 임베딩 채널 A/B KPI 하니스 (017 G4 — T012).

KoSimCSE(``channel='st'``)와 BGE-M3(``channel='st_bge'``)를 **같은 한국어 골든셋**으로
retrieval 품질(recall@20·MRR·nDCG@20 + latency p95)을 비교 측정한다. 이 하니스의 목적은
"어느 모델이 이긴다"는 단정이 아니라 **두 채널 수치를 같은 골든셋으로 산출**하는 것이다(SC-003).

테스트 전략(docs/테스트_가이드.md §2~3)
  - **골든셋 로더 + 스키마 검증(순수 단위, DB 무관)**: ``load_golden`` 이
    ``[{"query": str, "relevant_asset_ids": [str]}]`` 형식을 엄격히 검증한다. 잘못된 골든셋이
    조용히 0 을 산출하지 않도록 빈/형식오류는 명확한 ``ValueError`` 로 차단한다.
  - **A/B 하니스 e2e(``RUN_DB_E2E`` 게이트)**: ``tests/fixtures/search/golden_ko.json`` 이
    **존재할 때만** 실행한다(부재면 skip — 사람 검수 확정 필요). 각 질의를 ``('st', KoSimCSE)``·
    ``('st_bge', BGE)`` 두 채널로 **text+audio+video 멀티모달** ``search_hybrid`` 검색해 세 버킷을
    단일 랭킹으로 합치고(``_merge_ranked_ids``) ``metrics.py`` 로 지표를 산출, 채널별 평균 비교표를
    로그로 남긴다. text 만 평가하면 BGE-M3 핵심 강점(긴 STT=audio·영상 자막)을 못 재기 때문에
    멀티모달로 확장한다. 평가 풀은 **두 채널 모두 백필 + text/audio/video 모달리티 자산**으로
    한정한다(FR-005 Edge — 'st' 만 있는 비백필 자산 + image 자산을 빼 공정 비교; image 후보는
    CLIP 채널이라 텍스트 A/B 무관). 2회 실행 동일(SC-004).

결정성(헌법 3조): 질의 구조화 LLM 을 건너뛰고(``structured`` 주입) 모달리티를 text+audio+video 로
고정, 합산 정렬 tiebreak(asset_id)·임베딩 normalize 가 결정적이라 2회 실행이 동일 수치를 낸다.
읽기 전용(검색·조회만, 쓰기·스키마 0). 학습 0(두 모델 inference only).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType

from src.file.file_type_defs import MEDIA_TYPES_ST_CHUNK_SEARCH, MediaKind
from src.search.search_service import search_hybrid
from tests.fixtures.search.metrics import mrr, ndcg_at_k, recall_at_k

_RUN = os.getenv("RUN_DB_E2E") == "1"
_REPO = Path(__file__).resolve().parents[1]
_ENV = _REPO / ".env.dev"
_GOLDEN_PATH = _REPO / "tests" / "fixtures" / "search" / "golden_ko.json"
_DRAFT_MOD_PATH = _REPO / "scripts" / "build_golden_ko_draft.py"


def _load_draft_module() -> ModuleType:
    """``scripts/build_golden_ko_draft.py`` 를 모듈로 적재(scripts 는 패키지 아님)."""
    spec = importlib.util.spec_from_file_location("build_golden_ko_draft", _DRAFT_MOD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = 20  # recall@K·nDCG@K (plan D-5)
# 평가 풀 한정 후에도 top-K 가 충분하도록 버킷을 넉넉히 가져온다(K 보다 크게).
_FETCH = 100


def load_golden(path: str | Path) -> list[dict]:
    """골든셋 JSON 을 로드하고 스키마를 검증한다 → ``[{"query": str, "relevant_asset_ids": [str]}]``.

    하니스가 잘못된 입력으로 조용히 0 점을 산출하지 않도록, 형식 위반·빈 골든셋은 어떤
    위치에서 무엇이 틀렸는지 담은 ``ValueError`` 로 즉시 차단한다. 파일 부재는 ``FileNotFoundError``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"골든셋 파일이 없습니다: {p}")
    # 깨진 JSON 은 json.JSONDecodeError(=ValueError 하위)로 자연 전파된다.
    raw = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise ValueError(f"골든셋 최상위는 list 여야 합니다(받음: {type(raw).__name__})")
    if not raw:
        raise ValueError("골든셋이 비어 있습니다(질의 1건 이상 필요)")

    out: list[dict] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"[{i}] 항목은 dict 여야 합니다(받음: {type(item).__name__})")
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"[{i}] 'query' 는 비어있지 않은 str 여야 합니다(받음: {query!r})")
        rel = item.get("relevant_asset_ids")
        if not isinstance(rel, list) or not rel:
            raise ValueError(
                f"[{i}] 'relevant_asset_ids' 는 비어있지 않은 list 여야 합니다(받음: {rel!r})"
            )
        if not all(isinstance(x, str) and x.strip() for x in rel):
            raise ValueError(f"[{i}] 'relevant_asset_ids' 원소는 비어있지 않은 str 여야 합니다: {rel!r}")
        out.append({"query": query, "relevant_asset_ids": list(rel)})
    return out


class TestLoadGolden(unittest.TestCase):
    """골든셋 로더 스키마 검증(순수 단위, DB 무관)."""

    def _write(self, obj) -> str:
        fd, name = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        Path(name).write_text(
            json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj,
            encoding="utf-8",
        )
        self.addCleanup(lambda: Path(name).unlink(missing_ok=True))
        return name

    def test_valid_golden_loads(self) -> None:
        path = self._write(
            [
                {"query": "재무 보고서 요약", "relevant_asset_ids": ["a1", "a2"]},
                {"query": "회의록 키워드", "relevant_asset_ids": ["b1"]},
            ]
        )
        out = load_golden(path)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["query"], "재무 보고서 요약")
        self.assertEqual(out[0]["relevant_asset_ids"], ["a1", "a2"])

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_golden(_REPO / "tests" / "fixtures" / "search" / "_does_not_exist.json")

    def test_top_level_not_list_raises(self) -> None:
        path = self._write({"query": "x", "relevant_asset_ids": ["a"]})
        with self.assertRaises(ValueError):
            load_golden(path)

    def test_empty_list_raises(self) -> None:
        path = self._write([])
        with self.assertRaises(ValueError):
            load_golden(path)

    def test_item_not_dict_raises(self) -> None:
        path = self._write(["not a dict"])
        with self.assertRaises(ValueError):
            load_golden(path)

    def test_missing_query_raises(self) -> None:
        path = self._write([{"relevant_asset_ids": ["a"]}])
        with self.assertRaises(ValueError):
            load_golden(path)

    def test_empty_query_raises(self) -> None:
        path = self._write([{"query": "   ", "relevant_asset_ids": ["a"]}])
        with self.assertRaises(ValueError):
            load_golden(path)

    def test_missing_relevant_ids_raises(self) -> None:
        path = self._write([{"query": "x"}])
        with self.assertRaises(ValueError):
            load_golden(path)

    def test_empty_relevant_ids_raises(self) -> None:
        path = self._write([{"query": "x", "relevant_asset_ids": []}])
        with self.assertRaises(ValueError):
            load_golden(path)

    def test_relevant_ids_non_str_element_raises(self) -> None:
        path = self._write([{"query": "x", "relevant_asset_ids": ["a", 3]}])
        with self.assertRaises(ValueError):
            load_golden(path)

    def test_malformed_json_raises(self) -> None:
        path = self._write("{not valid json")
        with self.assertRaises(ValueError):  # json.JSONDecodeError 는 ValueError 의 하위
            load_golden(path)


class TestTopicPrefixRule(unittest.TestCase):
    """골든셋 주제(파일명 prefix) 추출·질의 생성 규칙(순수 단위, DB 무관) — 결정적(T010 재설계).

    데이터 파일명은 ``<주제>_<YouTube ID(11자)>_<제목>.<ext>`` 로 자연 군집한다. 같은 주제 =
    관련 자산이므로 주제 prefix 를 정답군으로 쓴다. 추출 규칙(정규식)을 여러 케이스로 고정한다.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_draft_module()

    def test_simple_two_token_topic(self) -> None:
        # <주제(밑줄 포함)>_<11자 ID>_<제목> — ID 직전까지가 주제.
        name = "무선_충전기_7iTajgt8pec_Qi2 다 같은 Qi2 아니죠 UGREEN 무선 충전기.jpg"
        self.assertEqual(self.mod._topic_from_filename(name), "무선_충전기")

    def test_single_token_topic(self) -> None:
        name = "선거_DOYCfVbioXQ_정원오 44.9% 오세훈 39.8% 서울시장 선거.mp4"
        self.assertEqual(self.mod._topic_from_filename(name), "선거")

    def test_three_token_topic(self) -> None:
        name = "주식_투자_기초_AbCdEfGhIjK_초보 투자자를 위한 가이드.mp3"
        self.assertEqual(self.mod._topic_from_filename(name), "주식_투자_기초")

    def test_youtube_id_with_underscore_and_dash(self) -> None:
        # 실제 ID 는 [A-Za-z0-9_-] 11자 — 밑줄/하이픈을 포함할 수 있다(주제 토큰과 구분).
        name = "스마트폰_5ncp-_GXBsU_YENA (최예나) - SMARTPHONE MV.jpg"
        self.assertEqual(self.mod._topic_from_filename(name), "스마트폰")

    def test_double_extension_still_matches_prefix(self) -> None:
        # .ko.txt / .stt.txt 같은 이중 확장자도 prefix 추출에는 무관(앞에서만 매칭).
        name = "라면_끓이기_GMjx9GrF1nY_3분 라면 황금레시피.ko.txt"
        self.assertEqual(self.mod._topic_from_filename(name), "라면_끓이기")

    def test_non_matching_name_returns_none(self) -> None:
        # 주제 패턴이 아닌 파일명은 None(골든셋에서 제외).
        self.assertIsNone(self.mod._topic_from_filename("manifest.json"))
        self.assertIsNone(self.mod._topic_from_filename("그냥파일.txt"))

    def test_query_from_topic_underscore_to_space(self) -> None:
        # 질의 = 주제명(밑줄→공백) — 자연어 질의.
        self.assertEqual(self.mod._query_from_topic("무선_충전기"), "무선 충전기")
        self.assertEqual(self.mod._query_from_topic("선거"), "선거")

    def test_rule_is_deterministic(self) -> None:
        # 같은 입력 2회 → 같은 주제(결정성, 헌법 3조).
        name = "등산_입문_ZyXwVuTsRqP_초보 등산 가이드.mp4"
        self.assertEqual(
            self.mod._topic_from_filename(name), self.mod._topic_from_filename(name)
        )


class TestBuildDraftsTopicGrouping(unittest.TestCase):
    """주제별 그룹핑·min-group 필터·결정적 정렬(mock conn, DB 무관)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_draft_module()

    def _conn(self, rows):
        from unittest import mock

        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = rows
        return conn

    def test_groups_by_topic_and_sorts(self) -> None:
        # 주제별로 묶고, 주제·asset_id 정렬(결정적). 질의=주제(밑줄→공백), 정답=주제 자산 전부.
        conn = self._conn(
            [
                {"asset_id": "id-b", "fs_path": "/d/무선_충전기_aaaaaaaaaaa_x.jpg"},
                {"asset_id": "id-a", "fs_path": "/d/무선_충전기_bbbbbbbbbbb_y.mp4"},
                {"asset_id": "id-e", "fs_path": "/d/등산_입문_ccccccccccc_z.mp3"},
                {"asset_id": "id-d", "fs_path": "/d/등산_입문_ddddddddddd_w.mp4"},
            ]
        )
        drafts = self.mod.build_drafts(conn, min_group=2)
        self.assertEqual(len(drafts), 2)
        # 주제 정렬: '등산_입문' < '무선_충전기'(유니코드 코드포인트 기준, 결정적).
        self.assertEqual(drafts[0]["query"], "등산 입문")
        self.assertEqual(drafts[0]["relevant_asset_ids"], ["id-d", "id-e"])  # asset_id 정렬
        self.assertEqual(drafts[1]["query"], "무선 충전기")
        self.assertEqual(drafts[1]["relevant_asset_ids"], ["id-a", "id-b"])

    def test_min_group_filters_small_topics(self) -> None:
        # min_group 미만 자산 주제는 제외(노이즈↓). 주제 패턴 아닌 파일은 그룹에서 제외.
        rows = [
            {"asset_id": "id-1", "fs_path": "/d/김치_담그기_aaaaaaaaaaa_a.jpg"},
            {"asset_id": "id-2", "fs_path": "/d/김치_담그기_bbbbbbbbbbb_b.mp4"},
            {"asset_id": "id-3", "fs_path": "/d/요가_자세_ccccccccccc_c.mp4"},  # 단일 자산 주제
            {"asset_id": "id-4", "fs_path": "/d/manifest.json"},  # 주제 없음
        ]
        d2 = self.mod.build_drafts(self._conn(rows), min_group=2)
        self.assertEqual([x["query"] for x in d2], ["김치 담그기"])  # 요가(1건)·manifest 제외
        d1 = self.mod.build_drafts(self._conn(rows), min_group=1)
        self.assertEqual([x["query"] for x in d1], ["김치 담그기", "요가 자세"])  # 1건도 포함

    def test_duplicate_asset_ids_deduped(self) -> None:
        # 같은 자산이 (DISTINCT 누락 등으로) 중복돼도 정답은 유일·정렬.
        rows = [
            {"asset_id": "id-1", "fs_path": "/d/낚시_방법_aaaaaaaaaaa_a.jpg"},
            {"asset_id": "id-1", "fs_path": "/d/낚시_방법_aaaaaaaaaaa_a.jpg"},
            {"asset_id": "id-2", "fs_path": "/d/낚시_방법_bbbbbbbbbbb_b.mp4"},
        ]
        drafts = self.mod.build_drafts(self._conn(rows), min_group=2)
        self.assertEqual(drafts[0]["relevant_asset_ids"], ["id-1", "id-2"])


def _finite_sim(row: dict) -> float:
    """행의 ``similarity`` 를 유한 실수로 읽는다(None/NaN/inf/비수치 → 0.0).

    합산 정렬 키가 결정적이 되도록(헌법 3조) 비유한 점수를 0.0 으로 눌러 정렬 폭발을 막는다.
    """
    v = row.get("similarity")
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def _merge_ranked_ids(buckets: dict, pool: set[str]) -> list[str]:
    """text/audio/video 버킷 rows 를 합쳐 평가 풀 자산 id 단일 랭킹을 만든다(순수, DB 무관).

    멀티모달 확장 핵심: text_documents + audio + video 세 버킷을 한 랭킹으로 합쳐 BGE-M3 의
    긴 문맥 강점(긴 STT=audio·영상 자막)이 점수에 반영되게 한다. **image 버킷은 제외**한다 —
    image 의 검색 후보는 CLIP(``channel='clip'``)이라 텍스트 채널 A/B('st' vs 'st_bge')와
    무관하기 때문이다. 정렬은 ``similarity`` 내림차순 + ``asset_id`` 오름차순 tiebreak 로
    입력 순서에 의존하지 않는 결정적 순서를 보장한다(헌법 3조). 같은 자산 중복은 첫 등장만
    남기고(모달리티 분리로 실제론 disjoint 하나 방어적 dedup), 평가 풀('st'·'st_bge' 공존 +
    text/audio/video) 밖 자산은 랭킹에서 제외한다.
    """
    rows: list[dict] = []
    for key in ("text_documents", "audio", "video"):
        rows.extend(buckets.get(key) or [])
    rows = sorted(rows, key=lambda r: (-_finite_sim(r), str(r.get("id", ""))))
    ranked: list[str] = []
    seen: set[str] = set()
    for r in rows:
        rid = str(r.get("id"))
        if rid in seen:
            continue
        seen.add(rid)
        if rid in pool:
            ranked.append(rid)
    return ranked


class TestMergeRankedIds(unittest.TestCase):
    """text/audio/video 버킷 합산 단일 랭킹(순수 함수, DB 무관) — 멀티모달 확장 핵심 로직.

    text 버킷만 평가하면 BGE-M3 핵심 강점(긴 STT=audio·영상 자막)을 측정 못 한다. 그래서
    text_documents + audio + video 세 버킷을 합쳐 단일 랭킹을 만든다. 합성 버킷으로 ① image
    버킷 제외(텍스트 채널 A/B 무관 — CLIP 검색) ② similarity 내림차순 + asset_id 오름차순
    tiebreak(결정적, 헌법 3조) ③ 평가 풀 한정 ④ 비유한 similarity 0.0 처리를 고정한다.
    """

    def test_combines_text_audio_video_excludes_image(self) -> None:
        buckets = {
            "text_documents": [{"id": "t1", "similarity": 0.5}],
            "audio": [{"id": "a1", "similarity": 0.9}],
            "video": [{"id": "v1", "similarity": 0.7}],
            "image": [{"id": "img1", "similarity": 0.99}],  # 풀에 있어도 랭킹 제외돼야 함
        }
        pool = {"t1", "a1", "v1", "img1"}
        ranked = _merge_ranked_ids(buckets, pool)
        # similarity 내림차순: a1(0.9) > v1(0.7) > t1(0.5). image 는 텍스트 A/B 무관이라 빠진다.
        self.assertEqual(ranked, ["a1", "v1", "t1"])

    def test_tiebreak_by_asset_id_ascending(self) -> None:
        buckets = {
            "text_documents": [
                {"id": "t_b", "similarity": 0.5},
                {"id": "t_a", "similarity": 0.5},
            ],
            "audio": [],
            "video": [],
        }
        pool = {"t_a", "t_b"}
        # 동점 → asset_id 오름차순(입력 순서 무관, 결정적).
        self.assertEqual(_merge_ranked_ids(buckets, pool), ["t_a", "t_b"])

    def test_pool_filter_drops_non_pool_assets(self) -> None:
        buckets = {
            "text_documents": [{"id": "t1", "similarity": 0.8}],
            "audio": [{"id": "a_out", "similarity": 0.9}],  # 평가 풀 밖 → 제외
            "video": [],
        }
        self.assertEqual(_merge_ranked_ids(buckets, {"t1"}), ["t1"])

    def test_missing_buckets_tolerated(self) -> None:
        # 버킷 키가 없거나 None 이어도 KeyError 없이 빈 리스트로 취급.
        self.assertEqual(_merge_ranked_ids({"audio": None}, {"x"}), [])
        self.assertEqual(_merge_ranked_ids({}, {"x"}), [])

    def test_non_finite_similarity_treated_as_zero(self) -> None:
        buckets = {
            "text_documents": [{"id": "t1", "similarity": float("nan")}],
            "audio": [{"id": "a1", "similarity": 0.1}],
            "video": [{"id": "v1"}],  # similarity 누락 → 0.0
        }
        pool = {"t1", "a1", "v1"}
        # a1(0.1) 최상위, 나머지 0.0 동점 → asset_id 오름차순(t1 < v1).
        self.assertEqual(_merge_ranked_ids(buckets, pool), ["a1", "t1", "v1"])

    def test_deduped_across_buckets(self) -> None:
        # 모달리티 분리로 실제론 disjoint 하지만, 같은 id 가 중복돼도 첫 등장만(방어적 dedup).
        buckets = {
            "text_documents": [{"id": "x", "similarity": 0.9}],
            "audio": [{"id": "x", "similarity": 0.2}],
            "video": [],
        }
        self.assertEqual(_merge_ranked_ids(buckets, {"x"}), ["x"])

    def test_deterministic_two_calls_equal(self) -> None:
        buckets = {
            "text_documents": [{"id": "t1", "similarity": 0.5}],
            "audio": [{"id": "a1", "similarity": 0.5}],
            "video": [{"id": "v1", "similarity": 0.5}],
        }
        pool = {"t1", "a1", "v1"}
        self.assertEqual(_merge_ranked_ids(buckets, pool), _merge_ranked_ids(buckets, pool))


class TestEvalPoolModalityFilter(unittest.TestCase):
    """평가 풀 조회가 image 모달리티를 SQL 에서 제외하는지(mock conn, DB 무관).

    image 자산의 'st'/'st_bge' 임베딩은 검색에 안 쓰여(시각은 CLIP 채널) 텍스트 A/B 로 평가
    불가하므로, 평가 풀을 text/audio/video 모달리티(``MEDIA_TYPES_ST_CHUNK_SEARCH``)로 한정한다.
    """

    def _db(self, rows):
        from unittest import mock

        db = mock.MagicMock()
        conn = db.transaction.return_value.__enter__.return_value
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = rows
        return db, cur

    def test_pool_query_restricts_to_text_audio_video(self) -> None:
        db, cur = self._db([("id-1",), ("id-2",)])
        pool = TestEmbeddingABKpi._load_eval_pool(db)
        self.assertEqual(pool, {"id-1", "id-2"})
        # execute 에 넘긴 modality 파라미터: image 없음 + text/audio/video 전부 포함.
        args, _ = cur.execute.call_args
        params = args[1]
        modalities = set(params[0])
        self.assertNotIn(MediaKind.IMAGE.value, modalities)
        self.assertIn(MediaKind.AUDIO.value, modalities)
        self.assertIn(MediaKind.VIDEO.value, modalities)
        self.assertEqual(modalities, set(MEDIA_TYPES_ST_CHUNK_SEARCH))


def _p95(values: list[float]) -> float:
    """nearest-rank p95(결정적). 빈 입력은 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, math.ceil(0.95 * len(s)) - 1)
    return s[idx]


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만 실행하는 실 DB e2e (두 채널 백필·BGE 모델 필요)")
class TestEmbeddingABKpi(unittest.TestCase):
    """A/B 하니스: 같은 골든셋으로 두 채널 지표 산출 + 2회 동일(SC-003·SC-004).

    합격 단언은 "두 채널 지표가 같은 골든셋으로 산출됨 + 2회 동일"까지다 — 어느 모델이
    우수한지는 단정하지 않는다(측정이 목적). golden_ko.json 부재면 skip(사람 검수 확정 필요).
    """

    db = None  # type: ignore[assignment]

    @classmethod
    def setUpClass(cls) -> None:
        if not _GOLDEN_PATH.is_file():
            raise unittest.SkipTest("golden_ko.json 없음 — 사람 검수 확정 필요")

        from dotenv import load_dotenv

        load_dotenv(_ENV, override=False)
        from src.config.settings import init_settings, model_for_channel

        init_settings("dev")
        from src.database.postgres_util import PostgresUtil

        try:
            cls.db = PostgresUtil()
            cls.db.__enter__()
            cls.pool = cls._load_eval_pool(cls.db)
        except Exception as exc:  # noqa: BLE001 — 접속 불가 시 skip
            raise unittest.SkipTest(f"DB 미접속: {type(exc).__name__}: {exc}") from None

        cls.goldens = load_golden(_GOLDEN_PATH)
        # 채널 → 질의 임베딩 모델(FR-004 질의-문서 일치). 'st' 는 None(media_search 가 KoSimCSE 로 해소).
        cls.channels = [("st", None), ("st_bge", model_for_channel("st_bge"))]

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.db is not None:
            cls.db.__exit__(None, None, None)

    @staticmethod
    def _load_eval_pool(db) -> set[str]:
        """평가 풀: 두 채널('st'·'st_bge') 공존 + 텍스트 A/B 로 평가 가능한 modality 자산 id 집합.

        FR-005 Edge: 'st' 만 있는 비백필 자산을 빼 공정 비교한다(두 채널 공존만). 추가로 평가 풀을
        text/audio/video 모달리티(``MEDIA_TYPES_ST_CHUNK_SEARCH``)로 한정한다 — image 자산의
        'st'/'st_bge' 임베딩은 검색에 쓰이지 않고(시각 후보는 CLIP ``channel='clip'``) 텍스트 채널
        A/B 로 품질을 평가할 수 없기 때문이다(image 제외). 모달리티 파라미터는 결정적 정렬로 바인딩.
        """
        eval_modalities = sorted(MEDIA_TYPES_ST_CHUNK_SEARCH)
        with db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT ae.asset_id FROM asset_embedding ae "
                "JOIN asset a ON a.asset_id = ae.asset_id "
                "WHERE ae.channel IN ('st', 'st_bge') "
                "AND a.modality = ANY(%s) "
                "GROUP BY ae.asset_id HAVING count(DISTINCT ae.channel) = 2",
                (eval_modalities,),
            )
            return {str(r[0]) for r in cur.fetchall()}

    def _ranked_ids(self, query: str, channel: str, model: str | None) -> list[str]:
        """한 질의를 한 채널로 검색해 평가 풀에 속한 자산 id 순위를 반환한다(멀티모달 합산).

        LLM 질의 구조화를 건너뛰고(``structured`` 주입) **text+audio+video 모달리티**를 검색해
        세 버킷을 단일 랭킹으로 합친다(``_merge_ranked_ids``) — text 만 평가하면 BGE-M3 핵심 강점
        (긴 STT=audio·영상 자막)을 못 잰다. ST 하이브리드 경로(text_documents·audio·video_st)에는
        ``text_channel``/``text_query_model`` 이 자동 전파돼 'st' vs 'st_bge' 가 분리 평가된다.
        **image 버킷은 합산에서 제외**(CLIP 검색이라 텍스트 채널 A/B 무관). 평가 풀('st'·'st_bge'
        공존 + text/audio/video) 밖 자산도 제외 — 두 채널이 같은 후보 우주를 보게 해 공정 비교.
        합산 정렬은 similarity 내림차순 + asset_id 오름차순 tiebreak 로 결정적(헌법 3조).
        """
        res = search_hybrid(
            query,
            modalities=["text", "audio", "video"],
            limit_per_bucket=_FETCH,
            structured={"semantic_query": query, "semantic_query_en": ""},
            text_channel=channel,
            text_query_model=model,
        )
        return _merge_ranked_ids(res["results"], self.pool)

    def _compute(self) -> tuple[dict, dict, int]:
        """골든셋 전체를 두 채널로 검색해 채널별 per-query 지표·latency 를 모은다.

        반환: (per-query 지표 dict, latency dict, 평가 질의 수). 평가 풀에 정답이 하나도 없는
        질의는 건너뛴다(공정 비교 대상 아님).
        """
        metrics = ("recall@20", "MRR", "nDCG@20")
        agg = {ch: {m: [] for m in metrics} for ch, _ in self.channels}
        lat = {ch: [] for ch, _ in self.channels}
        evaluated = 0
        for g in self.goldens:
            rel = set(g["relevant_asset_ids"]) & self.pool
            if not rel:
                continue
            evaluated += 1
            for ch, model in self.channels:
                t0 = time.perf_counter()
                ranked = self._ranked_ids(g["query"], ch, model)
                lat[ch].append(time.perf_counter() - t0)
                agg[ch]["recall@20"].append(recall_at_k(ranked, rel, _K))
                agg[ch]["MRR"].append(mrr(ranked, rel))
                agg[ch]["nDCG@20"].append(ndcg_at_k(ranked, rel, _K))
        return agg, lat, evaluated

    def test_ab_metrics_are_produced_and_deterministic(self) -> None:
        run1, lat1, evaluated = self._compute()
        run2, _lat2, _ = self._compute()

        self.assertGreater(
            evaluated, 0, "평가 풀에 정답이 있는 골든셋 질의가 없습니다(백필·골든셋 확인)"
        )

        # 비교표 로그(측정 결과). 합격 단정은 두 채널 산출 + 2회 동일까지.
        metrics = ("recall@20", "MRR", "nDCG@20")
        print(
            f"\n[BGE-M3 A/B KPI] modality=text+audio+video(image 제외) | "
            f"평가 질의 {evaluated} / 평가 풀(두 채널 백필 ∩ text/audio/video) {len(self.pool)}"
        )
        for m in metrics:
            line = f"  {m:<10}"
            for ch, _ in self.channels:
                line += f"  {ch}={statistics.mean(run1[ch][m]):.4f}"
            print(line)
        for ch, _ in self.channels:
            print(f"  p95[{ch}] = {_p95(lat1[ch]) * 1000:.1f}ms")

        # SC-003: 두 채널 모두 같은 골든셋으로 지표가 산출됐다(개수 일치·유한값).
        for ch, _ in self.channels:
            for m in metrics:
                self.assertEqual(len(run1[ch][m]), evaluated)
                self.assertTrue(all(math.isfinite(v) for v in run1[ch][m]))

        # SC-004: 2회 실행 동일 수치(결정성). latency 는 변동하므로 제외.
        self.assertEqual(run1, run2)


if __name__ == "__main__":
    unittest.main()
