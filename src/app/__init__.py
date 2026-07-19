"""src.app — 플랫폼의 모든 실행 진입점(CLI · HTTP API)을 모은 얇은 오케스트레이션 레이어.

각 모듈은 도메인 로직을 직접 들고 있지 않다 — argparse/FastAPI 로 인자를 받아 부트스트랩
(load_dotenv(.env.{env}) → init_settings → PostgresUtil)한 뒤, 코어 계층(pipeline·registry·
search·relations·portal)으로 위임만 한다. v2 아키텍처 "고정 뼈대"의 바깥 껍데기에 해당한다.

진입점 지도:
  · run_ingest            — per-asset 수집·적재 (route→classify→extract→embed→persist)
  · run_relations         — cross-asset 관계 제안 배치 (candidates→propose→graph_edge)
  · run_search            — 하이브리드 검색 CLI(OpenSearch BM25+kNN 클라이언트 융합·037)
  · run_opensearch_resync — PG→OpenSearch 전체 재색인 복구 도구(spec 020)
  · run_about_backfill    — aboutness 개체 소급 확정 배치(summary→ext_meta['about']·spec 073)
  · run_topic_backfill    — 자산 자기주제 소급 부여 배치(summary/keywords→asset_topic·spec 065)
  · portal_api            — 일반 도메인 포탈 백엔드(FastAPI, spec 010)

069 T406/T407: ``run_relations_review``(→052 포탈 관계검토 API 상위호환)·``sample_search_api``
(→포탈 /search 디버그 opt-in no_cutoff·compact·group_by_relation 이관) 두 진입점을 제거했다.

부트스트랩 관용(공통): main()/lifespan 이 .env.{env} 를 override=False(OS 기존 환경변수 우선)로
로드한 뒤 init_settings 로 frozen 설정을 만든다. **per-asset/cross-asset 진입점(run_ingest·
run_relations)만** `src.pipeline.builtins` 를 import 해 DEFAULT_REGISTRY 에 슬롯 전략을 등록하는
부수효과를 일으킨다 — 검색·검토·포탈 진입점은 레지스트리·도메인 팩이 필요 없어 import 하지 않는다.
"""
