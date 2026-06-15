"""src.app — 플랫폼의 모든 실행 진입점(CLI · HTTP API)을 모은 얇은 오케스트레이션 레이어.

각 모듈은 도메인 로직을 직접 들고 있지 않다 — argparse/FastAPI 로 인자를 받아 부트스트랩
(load_dotenv(.env.{env}) → init_settings → PostgresUtil)한 뒤, 코어 계층(pipeline·registry·
search·relations·portal)으로 위임만 한다. v2 아키텍처 "고정 뼈대"의 바깥 껍데기에 해당한다.

진입점 지도:
  · run_ingest            — per-asset 수집·적재 (route→classify→extract→embed→persist)
  · run_relations         — cross-asset 관계 제안 배치 (candidates→propose→graph_edge)
  · run_relations_review  — 관계 HITL 검토(proposed 엣지 승인/반려·relation_kind 승격)
  · run_search            — 하이브리드 검색 CLI(임베딩+FTS)
  · run_opensearch_resync — PG→OpenSearch 전체 재색인 복구 도구(spec 020)
  · portal_api            — 일반 도메인 포탈 백엔드(FastAPI, spec 010)
  · sample_search_api     — 샘플·개발용 검색 API(개발계획 미등재, 프로덕션 금지)

부트스트랩 관용(공통): main()/lifespan 이 .env.{env} 를 override=False(OS 기존 환경변수 우선)로
로드한 뒤 init_settings 로 frozen 설정을 만든다. **per-asset/cross-asset 진입점(run_ingest·
run_relations)만** `src.pipeline.builtins` 를 import 해 DEFAULT_REGISTRY 에 슬롯 전략을 등록하는
부수효과를 일으킨다 — 검색·검토·포탈 진입점은 레지스트리·도메인 팩이 필요 없어 import 하지 않는다.
"""
