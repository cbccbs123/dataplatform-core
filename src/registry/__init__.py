"""순수 코어 레지스트리 — 접근 등급(access_tier) + ext_meta 거버넌스.

과거 이 패키지가 담던 **자산 영속화**(``asset_persist``·``classification_persist``·``lineage_persist``)는
077(G1)에서 per-asset 파이프라인 쪽 ``ingest`` 로 이동했다(078 레포 분리 후 파이프라인 레포 ``processing.ingest``) — registry↔ingest/classify 커플링을
끊어 백엔드가 참조하지 않는 **순수 코어만** 잔류시키기 위함. 현재 이 패키지는 조회·거버넌스 전용이다.

모듈 맵
    ``access_tier``             — 042 접근 등급(clearance) StrEnum·판정
    ``ext_meta_field_registry`` — **039~041** ext_meta 거버넌스
        · 039 키·JSON Schema 값 검증 (write)
        · 040 ``fetch_access_tiers`` (read, 042 키 omit projection)
        · 041 ``ext_meta_field_registry`` 테이블 정본
"""
