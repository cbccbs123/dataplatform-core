"""자산 영속화 패키지 — per-asset 파이프라인 persist 스테이지의 조립 블록.

각 모듈은 psycopg ``Connection`` 을 첫 인자로 받아 한 테이블 묶음만 담당하고,
트랜잭션 경계는 호출하는 오케스트레이터(``app.run_ingest``)가 제어한다(단계별 짧은 트랜잭션):
- ``asset_persist``         : ``asset``/``asset_metadata``/``asset_embedding`` — 모델 A(received 조기 INSERT → finalize registered) + 해시 dedup
- ``classification_persist``: ``asset_classification`` 적재 + ``asset.domain_label``/``domain_confidence`` 갱신
- ``lineage_persist``       : ``asset_lineage`` PROV-DM 처리 이력 한 행
- ``schema_registry``       : ``ext_meta`` 키 허용목록 검증(``schema_registry`` 테이블 기반)
"""
