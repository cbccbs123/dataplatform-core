"""관계 품질 측정 하니스 (spec 031).

순수 측정 로직(LLM/DB 0·결정적·단위테스트)을 모은 패키지.
- `golden`: 골든 관계셋 모델 + `parse_golden`(순수 검증).
- `snapshot`: LLM 제안 동결 스냅샷 모델 + JSON round-trip.
- `metrics`: 후보 recall·관계 P/R·임계 스윕(순수 메트릭).
- `report`: 골든+스냅샷+키매핑 → 리포트 조립(순수).

DB/LLM 글루(키 해소·러너)는 이 패키지 밖(러너/사람 몫)이다.
"""
from src.relations.quality.golden import Golden, GoldenPair, parse_golden

__all__ = ["Golden", "GoldenPair", "parse_golden"]
