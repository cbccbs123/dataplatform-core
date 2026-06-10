"""021 G5 — OpenSearch 검색 read path 실OS e2e (게이트 ``RUN_OS_E2E=1``).

G1·G2 가 순수/가짜클라이언트로 검증한 쿼리 빌더·파이프라인·검색 seam 이 **실제 OpenSearch
하이브리드(nori 한국어 BM25 + ``knn_vector`` kNN + normalization-processor 검색 파이프라인)**에서
의도대로 동작하는지 확정 검증한다. 특히 ``opensearch_search`` docstring 이 "best-effort·G5 에서
확정"으로 남긴 **실 OS API 가정**의 최종 검증 지점이다:
    - ``client.search_pipeline.get/put``(검색 파이프라인 멱등 등록) 시맨틱(``ensure_search_pipeline``),
    - ``client.search(..., search_pipeline=name)`` 인자로 normalization-processor 가 실제 적용되는지,
    - ``hybrid`` 쿼리 DSL + ``sort:[{_score desc},{asset_id asc}]`` 결정적 tiebreaker(FR-009)가
      파이프라인 정규화 점수 위에서 동작하는지.

**자기완결 e2e — 실 PG 불요.** 테스트가 직접 작은 알려진 한국어 코퍼스(text/audio + 의료 1건 +
동점쌍 2건)를 임시 인덱스(``assets_021e2e``)에 ``asset_to_doc``(020 순수 문서 빌더)로 만들어
색인하고, 임시 검색 파이프라인(``assets-hybrid-021e2e``)을 등록한다. 질의·문서 임베딩은 **실제
모델**(활성 채널 KoSimCSE, 게이트 안이라 무방)로 같은 벡터 공간에서 계산해 kNN 이 비교 가능하게
구성한다. 임시 인덱스·파이프라인은 ``tearDownClass`` 에서 삭제하고 변경한 env 도 복원한다.

기본(``RUN_OS_E2E`` 미설정) 회귀에서는 클래스 전체가 **auto-skip** 되어 모델 로드·OS 접속·env
변경이 일어나지 않는다(헌법 8조 회귀 0). 실 측정은 사람이 OpenSearch 기동 후 ``RUN_OS_E2E=1`` 로.

────────────────────────────────────────────────────────────────────────────
실행 런북 (사람)
────────────────────────────────────────────────────────────────────────────
1) OpenSearch(+ k-NN, analysis-nori 플러그인) 기동 — docker 예시:

    docker run -d --name os021 -p 9200:9200 \
      -e discovery.type=single-node -e plugins.security.disabled=true \
      -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=Aurora!2026 \
      opensearchproject/opensearch:2.13.0
    # analysis-nori(한국어 형태소)·k-NN 은 배포 이미지에 기본 번들. nori 미포함 이미지면:
    #   docker exec os021 bin/opensearch-plugin install analysis-nori && docker restart os021

   ``.env.dev`` 의 ``OPENSEARCH_URL``(미설정 시 기본 http://localhost:9200)이 이 인스턴스를 가리키게 한다.
   (DEV 무인증 http 기준 — 보안 비활성. 보안 활성 환경은 024 범위.)

2) 본 e2e 실행 — 자기색인 코퍼스라 실 PG 불요, OpenSearch 만 있으면 된다:

    RUN_OS_E2E=1 conda run -n AuroraFS python -m unittest tests.test_opensearch_search_e2e -v

3) (참고) 실데이터로 검증하고 싶으면 — 020 동기화로 운영 인덱스(``assets``)를 채운 뒤,
   본 테스트 대신 ``SEARCH_BACKEND=opensearch`` 진입점(run_search 등)으로 육안 확인:

    OPENSEARCH_SYNC_ENABLED=true python -m src.app.run_ingest --env dev <FILE>...   # 증분 색인
    python -m src.app.run_opensearch_resync --env dev --recreate --ensure-pipeline  # 전체 재색인 + 파이프라인 등록
    SEARCH_BACKEND=opensearch python -m src.app.run_search --env dev "재무 보고서" --modalities text,audio

────────────────────────────────────────────────────────────────────────────
T010 — KPI A/B (사람 측정, 본 파일 밖·하드 정지점)
────────────────────────────────────────────────────────────────────────────
006 하니스(``scripts/measure_chunk_agg_kpi.py``)는 ``search_hybrid`` 를 통해 골든셋 recall@20 을
산출한다. ``search_hybrid`` 가 백엔드를 ``settings.search_backend``(=``SEARCH_BACKEND`` env)로
해소하므로, 같은 골든셋을 백엔드만 바꿔 A/B 한다(실 PG + 실 OS + 020 동기화 필요):

    RUN_DB_E2E=1 SEARCH_BACKEND=pg         conda run -n AuroraFS python scripts/measure_chunk_agg_kpi.py --env dev
    RUN_DB_E2E=1 SEARCH_BACKEND=opensearch conda run -n AuroraFS python scripts/measure_chunk_agg_kpi.py --env dev

   - **SC-002(품질)**: opensearch 의 text/audio recall@20 ≥ pg 인지 비교표로 확인.
   - **SC-003(성능)**: text/audio 검색 p95 ≤ 1000ms. 위 하니스는 recall 위주라 p95 는 측정 항목이
     아니므로, ``tests/test_search_kpi_e2e.py``(p95 측정 패턴) 또는 search_hybrid 호출을
     ``time.perf_counter`` 로 감싼 30회 측정으로 백엔드별 p95 를 별도 측정한다.
   - 결과(recall·p95 비교)는 ADR ``docs/decisions/2026-06-10-opensearch-search-cutover.md`` 측정
     기록에 남긴다.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

_RUN = os.getenv("RUN_OS_E2E") == "1"
_INDEX = "assets_021e2e"
_PIPELINE = "assets-hybrid-021e2e"
_REPO_ROOT = Path(__file__).resolve().parents[1]

# 코퍼스 자산 id(결정적·테스트 스코프). 동점쌍은 asset_id 사전순(a<b)으로 tiebreaker(FR-009) 검증.
_ID_FIN = "021e2e-text-fin"     # 재무/매출 (text)
_ID_COOK = "021e2e-text-cook"   # 요리 (text)
_ID_TRIP = "021e2e-text-trip"   # 여행 (text)
_ID_MEET = "021e2e-audio-meet"  # 회의 녹취 (audio)
_ID_MUSIC = "021e2e-audio-music"  # 음악 팟캐스트 (audio)
_ID_MED = "021e2e-text-med"     # 의료(domain_label='medical', text) — 검색에서 배제돼야 함
_ID_TWIN_A = "021e2e-twin-a"    # 동점쌍 — 동일 내용/임베딩, asset_id 만 다름
_ID_TWIN_B = "021e2e-twin-b"


def _os_reachable() -> bool:
    """``RUN_OS_E2E=1`` 이고 OpenSearch 가 응답하면 True(020 e2e 게이트 패턴 동형, 게이트만 RUN_OS_E2E)."""
    if not _RUN:
        return False
    try:
        import urllib.request

        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env.dev", override=False)
        url = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
        with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 — 로컬 DEV OS
            return r.status == 200
    except Exception:
        return False


# 작은 알려진 코퍼스. 각 질의의 키워드는 **목표 문서에만** 등장하도록 토픽을 분리해, nori BM25 가
# 결정적으로 목표를 상위로 올린다(recall sanity 의 안정성). (asset_id, modality, domain_label, summary, keywords, labels)
_CORPUS: tuple[tuple[str, str, str | None, str, list[str], list[str]], ...] = (
    (_ID_FIN, "txt", "general", "2024년 분기 재무 보고서 매출 영업이익 분석 자료", ["재무", "매출", "영업이익"], ["보고서"]),
    (_ID_COOK, "txt", "general", "김치찌개 끓이는 요리법 재료 손질 방법 정리", ["요리", "김치찌개", "레시피"], ["가이드"]),
    (_ID_TRIP, "txt", "general", "제주도 여행 일정 관광 명소 추천 코스 안내", ["여행", "제주도", "관광"], ["여행기"]),
    (_ID_MEET, "audio", "general", "프로젝트 주간 회의 녹취록 일정 공유 결정사항", ["회의", "녹취", "일정"], ["회의록"]),
    (_ID_MUSIC, "audio", "general", "인디 음악 팟캐스트 인터뷰 방송 에피소드", ["음악", "팟캐스트", "방송"], ["방송"]),
    (_ID_MED, "txt", "medical", "환자 진료 기록 처방 내역 병원 검사 결과 소견", ["환자", "진료", "처방"], ["의무기록"]),
    (_ID_TWIN_A, "txt", "general", "오늘의 날씨 기상 예보 강수 확률 안내", ["날씨", "기상", "예보"], ["기상"]),
    (_ID_TWIN_B, "txt", "general", "오늘의 날씨 기상 예보 강수 확률 안내", ["날씨", "기상", "예보"], ["기상"]),
)


@unittest.skipUnless(_os_reachable(), "RUN_OS_E2E=1 + 실 OpenSearch 필요")
class TestOpenSearchSearchE2E(unittest.TestCase):
    """실 OS 하이브리드 검색 e2e — 자기색인 코퍼스로 recall·결정성·의료배제·LLM 0·정규화 융합 검증."""

    @classmethod
    def setUpClass(cls) -> None:
        # 임시 인덱스·파이프라인을 settings 에 주입해 search_hybrid(backend=opensearch) 도 같은 코퍼스를
        # 보게 한다(LLM 0 테스트가 진입점 경로를 그대로 탄다). .env.dev 엔 이 두 키가 없으므로 override=False
        # 로드와 무관하게 우리 값이 유지된다. 변경 전 값은 복원용으로 보관.
        cls._saved_env = {
            k: os.environ.get(k)
            for k in ("OPENSEARCH_INDEX", "OPENSEARCH_SEARCH_PIPELINE")
        }
        os.environ["OPENSEARCH_INDEX"] = _INDEX
        os.environ["OPENSEARCH_SEARCH_PIPELINE"] = _PIPELINE

        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env.dev", override=False)
        from src.config.settings import active_embed_channel, init_settings

        init_settings("dev")
        from src.search.opensearch_search import embed_query, ensure_search_pipeline
        from src.search.opensearch_sync import asset_to_doc, ensure_index, get_client

        cls.client = get_client()
        cls.channel = active_embed_channel()

        # 임시 인덱스(020 매핑·nori·knn_vector) + 정규화 검색 파이프라인 멱등 등록.
        ensure_index(cls.client, _INDEX, recreate=True)
        ensure_search_pipeline(cls.client, _PIPELINE, weights=(0.5, 0.5))

        # 코퍼스를 실제 모델로 임베딩 → asset_to_doc(순수) → 색인. 문서·질의가 같은 임베더(채널 모델)를
        # 써 같은 벡터 공간에서 kNN 비교된다(FR-004 질의-문서 일치).
        for asset_id, modality, domain, summary, keywords, labels in _CORPUS:
            text = summary + " " + " ".join(keywords)
            row = {
                "asset_id": asset_id,
                "modality": modality,
                "domain_label": domain,
                "status": "registered",
                "fs_path": f"/data/{asset_id}.txt",
                "ext_meta": {"summary": summary, "keywords": keywords, "labels": labels},
                "emb": embed_query(text, channel=cls.channel),
                "chunk_count": 1,
            }
            doc = asset_to_doc(row, cls.channel)
            cls.client.index(index=_INDEX, id=doc["asset_id"], body=doc)
        cls.client.indices.refresh(index=_INDEX)

    @classmethod
    def tearDownClass(cls) -> None:
        # 임시 인덱스·파이프라인 삭제(best-effort) + env 복원.
        try:
            if cls.client.indices.exists(index=_INDEX):
                cls.client.indices.delete(index=_INDEX)
        finally:
            try:
                cls.client.search_pipeline.delete(id=_PIPELINE)
            except Exception:
                pass  # 파이프라인 삭제 API 부재/오류는 정리 실패로 보지 않음
            for k, v in cls._saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _search(
        self, query: str, modalities: list[str], *, k: int = 10
    ) -> dict[str, list[dict]]:
        """임시 인덱스·파이프라인으로 하이브리드 검색(seam 그대로). 버킷 {modality: [rows]} 반환."""
        from src.search.opensearch_search import search_assets_os

        return search_assets_os(
            self.client,
            query,
            modalities=modalities,
            k=k,
            channel=self.channel,
            index=_INDEX,
            pipeline_name=_PIPELINE,
            exclude_medical=True,
        )

    # ── 1) recall sanity (SC-002 방향) ──────────────────────────────────────
    def test_recall_sanity_top1(self) -> None:
        """특정 한국어 질의가 의도한 text/audio 문서를 상위로 회수한다(nori BM25 + kNN)."""
        text = self._search("재무 매출 보고서", ["text"])["text"]
        self.assertTrue(text, "text 버킷에 결과가 있어야 한다")
        self.assertEqual(text[0]["id"], _ID_FIN, "재무 질의의 top1 은 재무 문서여야 한다")
        self.assertEqual(text[0]["modality"], "txt")  # 저장 modality 값(라벨 'text' → 값 'txt')

        audio = self._search("회의 녹취 일정", ["audio"])["audio"]
        self.assertTrue(audio, "audio 버킷에 결과가 있어야 한다")
        self.assertEqual(audio[0]["id"], _ID_MEET, "회의 질의의 top1 은 회의 녹취 문서여야 한다")
        self.assertEqual(audio[0]["modality"], "audio")

    # ── 2) 결정성 (헌법 3조 · FR-009 tiebreaker) ─────────────────────────────
    def test_determinism_same_query_same_order(self) -> None:
        """같은 질의 2회 → 동일 top-k 순서. 동점쌍은 asset_id 오름차순(결정적 tiebreaker)."""
        run1 = [r["id"] for r in self._search("재무 매출 보고서", ["text"])["text"]]
        run2 = [r["id"] for r in self._search("재무 매출 보고서", ["text"])["text"]]
        self.assertEqual(run1, run2, "같은 질의 2회는 동일 순서여야 한다(결정성)")

        # 동점쌍(동일 내용·동일 임베딩 → 동일 정규화 점수)은 asset_id 오름차순으로 결정(FR-009).
        ids = [r["id"] for r in self._search("날씨 기상 예보", ["text"])["text"]]
        self.assertIn(_ID_TWIN_A, ids)
        self.assertIn(_ID_TWIN_B, ids)
        self.assertLess(
            ids.index(_ID_TWIN_A), ids.index(_ID_TWIN_B),
            "동점 시 asset_id 오름차순 tiebreaker — twin-a 가 twin-b 보다 앞이어야 한다",
        )

    # ── 3) 의료 배제 (FR-011 · SC-004 · 헌법 10조) ───────────────────────────
    def test_medical_excluded_from_results(self) -> None:
        """의료(domain_label='medical') 문서는 색인돼 있어도 검색 결과에서 제외된다(쿼리 단 실효)."""
        # 의료 문서가 인덱스엔 존재함(배제가 '없어서'가 아니라 '필터'임을 증명).
        src = self.client.get(index=_INDEX, id=_ID_MED)["_source"]
        self.assertEqual(src.get("domain_label"), "medical")

        # 의료 내용에 들어맞는 질의로도 의료 자산이 어느 버킷에도 안 나와야 한다.
        buckets = self._search("환자 진료 처방", ["text", "audio"])
        found = {r["id"] for rows in buckets.values() for r in rows}
        self.assertNotIn(_ID_MED, found, "의료 자산은 검색 결과에 없어야 한다(FR-011)")

    # ── 4) LLM 0 (SC-004 · FR-004) ──────────────────────────────────────────
    def test_search_hybrid_opensearch_calls_no_llm(self) -> None:
        """search_hybrid(backend='opensearch') text/audio 경로에서 LLM 질의 구조화 0회(seam 미접촉)."""
        from src.search import search_service

        with mock.patch(
            "src.search.media_search.structure_user_query"
        ) as m_llm:
            result = search_service.search_hybrid(
                "재무 매출 보고서",
                modalities=["text", "audio"],
                backend="opensearch",
            )
        self.assertEqual(m_llm.call_count, 0, "OS text/audio 경로는 LLM 질의 구조화 0(FR-004·SC-004)")
        # 응답 동형(SC-005): pg 분기와 같은 버킷 키.
        self.assertIn("text_documents", result["results"])
        self.assertIn("audio", result["results"])
        self.assertEqual(result["meta"].get("backend"), "opensearch")

    # ── 5) 정규화 융합 동작 (FR-005 — 파이프라인 실적용 최종 검증) ───────────
    def test_normalization_pipeline_applied(self) -> None:
        """normalization-processor(min-max + 가중평균)가 실제 적용된다 — 결합 점수 ≤ 1.0 상한이 증거.

        min-max 정규화는 서브쿼리별 최고점을 1.0 으로 스케일하고, arithmetic_mean(가중치 합 1.0)으로
        결합하므로 결합 점수의 이론 상한은 1.0(양 서브쿼리 동시 1등). 정규화 미적용(원시 BM25)이면
        보통 1.0 을 초과한다 — 이 상한이 ``client.search(search_pipeline=)`` 로 파이프라인이 실제
        적용됐다는 신호다(G1·G2 가 best-effort 로 가정한 실 OS API 의 최종 검증 지점).
        """
        rows = self._search("재무 매출 보고서", ["text"])["text"]
        self.assertTrue(rows, "결과가 있어야 융합 점수를 검증할 수 있다")
        top = rows[0]["similarity"]
        self.assertGreater(top, 0.0, "정규화 융합 점수는 양수여야 한다")
        self.assertLessEqual(top, 1.0 + 1e-6, "min-max 정규화 + 가중평균 결합 점수 상한 1.0(파이프라인 실적용)")
        # 점수 단조 비증가(정렬 일관성 — _score desc).
        sims = [r["similarity"] for r in rows]
        self.assertEqual(sims, sorted(sims, reverse=True), "결과는 점수 내림차순이어야 한다")


if __name__ == "__main__":
    unittest.main()
