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
    ``('st_bge', BGE)`` 두 채널로 ``search_hybrid`` 검색해 ``metrics.py`` 로 지표를 산출하고,
    채널별 평균 비교표를 로그로 남긴다. 평가 풀은 **두 채널 모두 백필된 자산**으로 한정한다
    (FR-005 Edge — 'st' 만 있는 비백필 자산을 빼 공정 비교). 2회 실행 동일(SC-004).

결정성(헌법 3조): 질의 구조화 LLM 을 건너뛰고(``structured`` 주입) 모달리티를 text 로 고정,
검색 tiebreak(asset_id)·임베딩 normalize 가 결정적이라 2회 실행이 동일 수치를 낸다.
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


class TestDraftQueryRule(unittest.TestCase):
    """골든셋 초안 질의 생성 규칙(순수 단위, DB 무관) — 결정적·폴백 순서 검증(T010)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_draft_module()

    def test_keywords_preferred_and_capped(self) -> None:
        # keywords 우선, 앞 3개만 공백으로 묶는다(짧은 자연 질의).
        q = self.mod._draft_query_from_ext_meta(
            {"keywords": ["재무", "보고서", "2024", "분기"], "summary": "무시됨"}
        )
        self.assertEqual(q, "재무 보고서 2024")

    def test_summary_first_sentence_when_no_keywords(self) -> None:
        # keywords 없으면 summary 첫 문장(종결부호 경계)으로 폴백.
        q = self.mod._draft_query_from_ext_meta(
            {"summary": "2024년 연간 재무 실적 요약. 상세 내용은 본문 참고."}
        )
        self.assertEqual(q, "2024년 연간 재무 실적 요약")

    def test_summary_capped_to_max_len(self) -> None:
        long = "가" * 200
        q = self.mod._draft_query_from_ext_meta({"summary": long})
        self.assertEqual(len(q), self.mod._QUERY_MAX_LEN)

    def test_empty_meta_returns_none(self) -> None:
        # summary·keywords 둘 다 없으면 None(초안에서 제외).
        self.assertIsNone(self.mod._draft_query_from_ext_meta({}))
        self.assertIsNone(self.mod._draft_query_from_ext_meta({"summary": "  ", "keywords": []}))

    def test_keywords_string_form_tolerated(self) -> None:
        # 방어적: keywords 가 단일 str 로 와도 처리.
        q = self.mod._draft_query_from_ext_meta({"keywords": "단일키워드"})
        self.assertEqual(q, "단일키워드")

    def test_rule_is_deterministic(self) -> None:
        # 같은 입력 2회 → 같은 질의(결정성, 헌법 3조).
        meta = {"keywords": ["a", "b", "c", "d"]}
        self.assertEqual(
            self.mod._draft_query_from_ext_meta(meta),
            self.mod._draft_query_from_ext_meta(meta),
        )

    def test_build_drafts_shape_with_mock_conn(self) -> None:
        # build_drafts 가 {query, relevant_asset_ids:[자기자산]} 형태를 만들고, 질의 못 만든 자산은 skip.
        from unittest import mock

        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = [
            {"asset_id": "a1", "ext_meta": {"keywords": ["문서", "요약"]}},
            {"asset_id": "a2", "ext_meta": {}},  # 질의 생성 불가 → skip
            {"asset_id": "a3", "ext_meta": {"summary": "회의록 핵심 정리."}},
        ]
        drafts = self.mod.build_drafts(conn, limit=None)
        self.assertEqual(len(drafts), 2)
        self.assertEqual(drafts[0], {"query": "문서 요약", "relevant_asset_ids": ["a1"]})
        self.assertEqual(drafts[1]["relevant_asset_ids"], ["a3"])


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
        """두 채널('st'·'st_bge') 모두 백필된 자산 id 집합(평가 풀, FR-005 Edge)."""
        with db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT asset_id FROM asset_embedding "
                "WHERE channel IN ('st', 'st_bge') "
                "GROUP BY asset_id HAVING count(DISTINCT channel) = 2"
            )
            return {str(r[0]) for r in cur.fetchall()}

    def _ranked_ids(self, query: str, channel: str, model: str | None) -> list[str]:
        """한 질의를 한 채널로 검색해 평가 풀에 속한 자산 id 순위를 반환한다.

        LLM 질의 구조화를 건너뛰고(``structured`` 주입) text 모달리티만 검색해 결정적·텍스트
        채널 한정으로 만든다. 평가 풀('st'·'st_bge' 공존) 밖 자산은 제외 — 두 채널이 같은
        후보 우주를 보게 해 공정 비교(비백필 'st' 자산이 'st' 채널에만 유리하게 잡히는 편향 제거).
        """
        res = search_hybrid(
            query,
            modalities=["text"],
            limit_per_bucket=_FETCH,
            structured={"semantic_query": query, "semantic_query_en": ""},
            text_channel=channel,
            text_query_model=model,
        )
        ranked = [str(r["id"]) for r in res["results"]["text_documents"]]
        return [rid for rid in ranked if rid in self.pool]

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
            f"\n[BGE-M3 A/B KPI] 평가 질의 {evaluated} / 평가 풀(두 채널 백필) {len(self.pool)}"
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
