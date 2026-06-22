"""자산 영속화 패키지 — per-asset 파이프라인 persist 스테이지의 조립 블록.

각 모듈은 psycopg ``Connection`` 을 첫 인자로 받아 한 테이블 묶음만 담당하고,
트랜잭션 경계는 호출하는 오케스트레이터(``app.run_ingest``)가 제어한다.

모듈 맵
    ``asset_persist``          — asset / metadata / embedding (모델 A)
    ``classification_persist`` — domain_label 갱신
    ``lineage_persist``        — asset_lineage
    ``ext_meta_field_registry`` — **039~041** ext_meta 거버넌스
        · 039 키·JSON Schema 값 검증 (write)
        · 040 ``fetch_access_tiers`` (read, 042 키 omit projection)
        · 041 ``ext_meta_field_registry`` 테이블 정본
"""
