-- 301 — 명시적 3종(same_series·references·derived_from) 설명문을 좁힌다 (2026-08-03 채택)
--
-- 왜: `relation_kind.description` 은 **LLM 프롬프트에 그대로 실린다**(관계 카탈로그 블록).
-- 전량 재생성 실측에서 이 3종의 이름표 정확도가 25~33% 로 무너졌고(합계 16건), 원인은 종류
-- 정의가 프롬프트 세 곳에 흩어져 어긋난 것이었다:
--   ① 엣지케이스 안내(`src/relations/prompt.py` 모듈 docstring) — "같은 stem + 순번/버전" (정확)
--   ② 선택 힌트(`RELATION_KIND_HINTS_KO`) — "브랜드 라인업 등 연속·묶음" (느슨)
--   ③ 이 컬럼 — "같은 시리즈·연작·라인업 연결" (느슨)
-- "라인업"이 LLM 에게 *"같은 범주에 속하는 것들"* 로 읽혀 창덕궁↔덕수궁을 "연작",
-- 첼로↔바이올린을 "연작"으로 묶었다. ①이 옳은 정의였으므로 ②(코드)와 ③(여기)을 그쪽에 맞춘다.
--
-- 측정(330자산 3층 표본): A층 개선 2·악화 0 · 전체 이름표 정확도 83.0→85.0% · 회귀 없음
-- (dup +1.3pp · same_domain +4.7pp). 상세·한계는 `src/relations/prompt.py`
-- `RELATION_KIND_HINTS_KO` 주석과 `docs/설계_변경이력.md`.
--
-- 데이터 무접촉 — `description` 텍스트만 갱신한다(kind_id·status·엣지 무변경).
-- 멱등: 같은 값을 다시 써도 결과가 같다.

UPDATE relation_kind
   SET description = '파일명이 같은 어간 + 순번/버전인 연작(범주 묶음이 아니다)'
 WHERE kind_code = 'same_series';

UPDATE relation_kind
   SET description = '한쪽이 다른쪽을 명시적으로 가리키는 인용·링크·제목 참조'
 WHERE kind_code = 'references';

UPDATE relation_kind
   SET description = '한쪽이 다른쪽에서 생성된 파생(원문→요약·번역·전사)'
 WHERE kind_code = 'derived_from';
