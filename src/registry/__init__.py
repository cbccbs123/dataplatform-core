"""순수 코어 레지스트리 — 접근 등급(access_tier) + ext_meta 거버넌스.

과거 이 패키지가 담던 **자산 영속화**는 077(G1)에서 per-asset 쪽으로 이동했다(registry↔ingest/classify
커플링 제거·백엔드 미참조 순수 코어만 잔류). 최종 소속이 갈린다:
- ``asset_persist``·``classification_persist`` → **파이프라인 레포** ``dataplatform-pipeline`` ``processing.ingest``.
- ``lineage_persist`` → **코어** ``src.database.lineage_persist``(078 재승격 — asset_lineage 표 쓰기·ingest/relations 공용 cross-cutting).
현재 이 패키지는 조회·거버넌스 전용이다.

모듈 맵
    ``access_tier``             — 042 접근 등급(clearance) StrEnum·판정
    ``ext_meta_field_registry`` — **039~041** ext_meta 거버넌스
        · 039 키·JSON Schema 값 검증 (write)
        · 040 ``fetch_access_tiers`` (read, 042 키 omit projection)
        · 041 ``ext_meta_field_registry`` 테이블 정본
"""
