"""OpenSearch 검색 융합·게이트·컷의 기본값 단일 출처(027 F1 — 순수 상수·의존 0).

027 독립 비평이 발견한 즉시 결함 F1: 임계·가중치 기본값이 **3곳에 복제·구식화**돼 있었다
(settings resolver·검색 모듈 중복 상수·calibrate 스크립트 — 코드 0.15/0.43 vs 운영 0.17/0.50,
새 환경에 설정 없이 배포하면 구식 코드 기본값이 적용되는 지뢰). 이 모듈을 **유일 출처**로 두고
``src.config.settings`` resolver·``src.search.opensearch_search``·calibrate 가 공유한다(FR-006).

순수 상수만 둔다(import 0·IO 0) — settings 미초기화 환경(순수 단위)에서도 안전하게 참조된다.
임계는 **코사인 스케일**(클라이언트 융합 전환으로 정규화 스케일 임계 4종이 폐기됨, 027). 실측
확정치(EPS·FLOOR·RESULT_FLOOR)는 G4 실OS 재보정 후 **여기 1곳만** 갱신한다.
044 evidence 가중·rescue 임계·generic seed·``SEARCH_EVIDENCE_*`` 토글 기본값도 동일 원칙.
"""

from __future__ import annotations

# ── 버킷 게이트(robust baseline) 기본값 ──────────────────────────────────────
# OS 검색은 기본 게이트 on(프로덕션 경로) — settings 미설정 환경에서도 무관 결과 표출을 차단한다.
OS_CUTOFF_ENABLED_DEFAULT: bool = True
# 상대 신호 임계: keep = (top − baseline) ≥ EPS. baseline = '하위 절반 평균'(robust — 밀집 토픽
# '충전' 오컷 구조 해결). **G4 실OS 재보정 확정치**(있음 Δmin 0.224 vs 없음 Δmax 0.189 분포 기준).
OS_CUTOFF_EPS_DEFAULT: float = 0.18
# 절대 backstop 임계: keep 의 두 번째 조건 top ≥ FLOOR(코퍼스 전체가 평탄할 때의 느슨한 하한).
OS_CUTOFF_FLOOR_DEFAULT: float = 0.50

# ── per-result 컷(단일 코사인 스케일) 기본값 ────────────────────────────────
# 행 유지 = BM25 매칭(어휘 증거) OR 원시 코사인 ≥ RESULT_FLOOR. 024 의 모달리티별 정규화 스케일 임계
# 4종을 대체하는 전역 1개(코사인 스케일). **G4 재보정 확정치**(없음 top 최대 0.538 위·차단 96% 운영점).
OS_RESULT_FLOOR_DEFAULT: float = 0.55

# ── 융합·서브검색 기본값 ────────────────────────────────────────────────────
# 융합 가중치 (w_bm25, w_knn) — 서브검색 순서 [bm25, knn] 과 동일 의미. min-max 정규화 후 가중평균.
OS_FUSION_WEIGHTS_DEFAULT: tuple[float, float] = (0.5, 0.5)
# BM25 multi_match operator 기본값 — **'and'(전 토큰 매칭·운영 검증값)**. 025 는 회귀0 원칙으로
# 'or' 기본+.env 옵트인이었으나, 027 F1 원칙(코드 기본=운영 보정값 — 새 환경 지뢰 제거)과 어휘
# 구제 규칙의 전제(BM25 매칭=전 토큰 증거)에 따라 기본을 운영점으로 통일했다(리뷰 후속).
OS_BM25_OPERATOR_DEFAULT: str = "and"
# 게이트 표본 하한 — 모달리티당 kNN 을 max(요청 k, 이 값)으로 뽑아 robust baseline(하위 절반 평균)이
# 충분한 표본에서 안정되게 한다(요청 k 가 작아도 background 추정이 흔들리지 않게).
OS_KNN_SAMPLE_K: int = 50

# ── reranker 평가(028) 기본값 — cross-encoder 쌍별 절대 판정(추론만·헌법 1조 inference-only) ──
# 기본 off(회귀 0 — 평가 opt-in). 모델은 다국어 cross-encoder(한국어 포함), τ 는 쌍별 절대 점수
# (sigmoid 0~1) 하한 — **모델 기준 1회 보정**(쌍 점수는 코퍼스 불변이라 코퍼스 추적 재보정 불필요).
OS_RERANK_ENABLED_DEFAULT: bool = False
OS_RERANK_MODEL_DEFAULT: str = "BAAI/bge-reranker-v2-m3"
OS_RERANK_TOP_R_DEFAULT: int = 10   # 모달리티당 채점 후보 상한(지연 통제)
OS_RERANK_TAU_DEFAULT: float = 0.0  # augment 기본 = 재정렬만(드롭 0). τ>0 은 선택적 정밀 필터 —
# G3 실OS 측정: τ 드롭이 약한-정답 행을 잘라 recall 0.94→0.90·차단 23→22 회귀. 선별은 cut_rows 가
# 맡고 리랭크는 순서만 바꾼다(rerank_reorder). τ 는 정밀이 더 필요한 운영점에서만 >0 으로 옵트인.

# ── 029: LLM 질의 명사구 정규화 토글 기본값 ─────────────────────────────────
# 기본 off(회귀 0 — 021 FR-004 가 제거한 검색시점 LLM 을 029 가 기본 off 토글로만 재허용한다). True 면
# 검색 직전 질의를 gemma 핵심 명사구로 정규화(temp=0·env 입력 0·src/llm/client 단일 seam)한 뒤 임베딩·
# BM25 양쪽에 동일 적용한다(028 측정: '별 보는 방법'→'천체 관측' 0.009→0.227, 패러프레이즈 오컷 직격).
# 기본 False 라 settings 미설정 환경·계약 테스트·027 폴백 불변 — 채택은 .env.dev opt-in(헌법 §Governance·
# 021 FR-004 정식 개정 동반). 단일 출처(F1)이므로 settings resolver·search_service 배선이 이 값을 공유한다.
OS_QUERY_NORM_ENABLED_DEFAULT: bool = False

# ── 072: 검색 질의 형태소 정규화 상수 (단일 출처 F1) ──────────────────────────
# 029 query-norm seam 의 정규화기를 gemma(LLM)에서 **nori 형태소 명사 추출 + 스톱워드 제거**로 교체한다
# (측정 2026-07-13: 자연어 nDCG@10 baseline 0.490 → 형태소 0.591 > LLM 0.575, 검색시점 LLM 0·결정적).
# 형태소가 효과 낸 본질 = kNN 입력 문장의 껍데기어 제거(복합어 토큰정확도·사전등록·재색인은 측정상 무효).
#
# 자연어 vs 단어 판별: 어절 수(공백 분리) < MIN 이면 원문 그대로(단어 검색은 정규화·_analyze IO 스킵·지연 0).
OS_QUERY_NORM_MIN_WORD_TOKENS: int = 3
# nori decompound_mode. none=복합명사 보존("김밥"을 "김 밥"으로 안 쪼갬 — discard 대비 근소 우위).
OS_QUERY_NORM_DECOMPOUND: str = "none"
# 추출 대상 명사류 품사(nori leftPOS 앞 코드): 일반/고유명사·외래어(SL)·한자(SH)·숫자(SN).
# 조사·어미·동사·부사는 자동 탈락(품사 필터). 의존명사(NNB)·대명사(NP)는 제외(검색 변별력 낮음).
OS_QUERY_NORM_NOUN_POS: frozenset[str] = frozenset({"NNG", "NNP", "SL", "SH", "SN"})
# 명사지만 검색 변별력 없는 모달리티어·지시성 명사(스톱워드). 이 제거가 개선의 최대 레버(+0.075).
# 닫힌·안정적 집합(매체 종류 + 검색 지시어)이라 자산 증가와 무관하게 거의 불변(유지보수 소).
OS_QUERY_NORM_STOPWORDS: frozenset[str] = frozenset({
    "영상", "사진", "이미지", "동영상", "그림", "문서", "자료", "정보", "추천",
    "소개", "방법", "법", "모습", "관련", "내용", "종류", "장면", "클립",
})

# ── 044 evidence · lexical rescue (단일 출처 — G0) ───────────────────────────
# OpenSearch BM25를 필드별 named query(`hit_keywords` 등 `_name`)로 쪼갠 뒤, 게이트 실패 버킷에서
# ``matched_queries`` 로 **어느 필드에서 hit 됐는지** 관측한다. ``query_evidence.evidence_score`` 가
# 아래 가중치로 1/0 합산하고, ``lexical_rescue_keep`` 가 policy·임계로 행 단위 keep/drop을 판단한다.
# (코사인 게이트 **통과** 행은 invariant — 이 블록과 무관·순서·점수 불변.)
# 정본: ``docs/search_query_filter_evidence_design.md`` · spec 044 · ADR 2026-06-24.
#
# ── 필드 evidence 가중치 (strong / weak tier) ─────────────────────────────────
# ``build_bm25_body`` 의 named clause `_name` 과 1:1 대응. hit 된 clause 만 가산(연속 BM25 점수 아님).
# strong: keywords·labels·file_name — 의도적 메타·식별자 필드. weak: summary·cross_meta — 본문·교차
# 필드(우연 substring·"비거리 테스트" 류 과매칭의 주범). ``hit_cross_meta`` 는 strong hit 가
# 이미 있으면 dedup 스킵(설계 §10.1 — summary 파생 중복 가산 금지).
EVIDENCE_HIT_KEYWORDS_WEIGHT: float = 3.0   # strong — ingest keywords·BM25 `hit_keywords`
EVIDENCE_HIT_LABELS_WEIGHT: float = 2.0     # strong — domain/tag labels·`hit_labels`
EVIDENCE_HIT_FILE_NAME_WEIGHT: float = 1.5  # strong — 정제 file_name·`hit_file_name`
EVIDENCE_HIT_SUMMARY_WEIGHT: float = 0.7    # weak — chunk summary·`hit_summary`
EVIDENCE_HIT_CROSS_META_WEIGHT: float = 0.3  # weak — cross_fields summary+keywords·`hit_cross_meta`
#
# ── rescue 임계 (가설 — G5 골든·q=테스트 스모크 후 **이 세 줄만** 재보정) ─────
# 적용 경로: 게이트 실패 ∧ BM25 행(`_bm25=True`) ∧ ``bm25_operator=and`` 일 때만(027 lexical rescue 잎).
# ``SEARCH_EVIDENCE_RESCUE_ENABLED=0`` 이면 임계 **미사용** — legacy `has_lexical` 전부 keep.
EVIDENCE_NORMAL_THRESHOLD: float = 1.5
# ``lexical_rescue=normal``(일반 질의·또는 generic+keyword) — **전체** evidence_score 하한.
# 예: keywords만 hit(3.0) → keep; summary+cross_meta 만(1.0) → drop.
EVIDENCE_RESTRICTED_STRONG_THRESHOLD: float = 2.5
# ``lexical_rescue=restricted``(generic single term + auto) — **strong tier 합** 만 본다(weak 무시).
# 예: q=테스트 → 낚시(summary weak만) drop; 반도체(keywords strong≥2.5) keep.
EVIDENCE_KEYWORD_THRESHOLD: float = 0.7
# ``mode=keyword``(사용자 명시) — weak evidence 도 rescue 허용. 전체 evidence_score 하한(낮게 설정).
# 포탈 ``GET /search?mode=keyword`` · suggestion "단어 포함 문서…" 와 쌍.
#
# ── generic single term seed (v1 · brainstorming YAGNI 6개) ─────────────────
# ``query_plan.is_generic_single_term``: (1) 공백 없는 단일 토큰·len≤12 이고 (2) NFKC+casefold 가
# seed 와 일치 → ``generic_single_term=True``. auto 모드에서 ``lexical_rescue=restricted`` 승격.
# **자동 필터 승격 금지**(FR-302): `테스트`→`tags=test` 변환 없음 — policy 플래그·suggestion 만.
# 추가 seed(`샘플`·`예시` 등)는 코퍼스 DF 측정 후 v2 — 무분별 확대 시 정상 단어까지 restricted.
GENERIC_SINGLE_TERM_SEED: tuple[str, ...] = (
    "테스트",
    "검증",
    "가이드",
    "test",
    "sample",
    "demo",
)
# 045 v2a: 운영 추가 seed — ``SEARCH_GENERIC_TERM_SEED_EXTRA`` env(콤마 구분). 미설정 시 빈.
# merge·dedup은 ``query_plan.merge_generic_term_seed`` · settings ``search_generic_term_seed``.
SEARCH_GENERIC_TERM_SEED_EXTRA_ENV = "SEARCH_GENERIC_TERM_SEED_EXTRA"
#
# ── env 토글 기본값 (settings ``SEARCH_EVIDENCE_*`` 로 덮어씀) ───────────────
SEARCH_EVIDENCE_DEBUG_DEFAULT: bool = False
# True → ``fuse_hybrid`` 결과 행에 debug 필드 부착(FR-405): ``matched_queries``,
# ``evidence_score``, ``strong_evidence_score``, ``gate_passed``, ``keep_reason``.
# 포탈·run_search JSON 관측용. 프로덕션 기본 off(응답 부풀림·내부 _name 노출 최소화).
SEARCH_EVIDENCE_RESCUE_ENABLED_DEFAULT: bool = True
# lexical rescue **정책 집행** 스위치. False=027 호환(어휘 BM25 hit 이면 게이트 실패 후에도 전부 keep,
# ``keep_reason=legacy_lexical``). True=``lexical_rescue_keep`` live(weak-only drop·restricted 등).
# 044 G2 merge 시점엔 회귀 방지로 default=0(shadow)였고, 045 G2b 골든(RESCUE 0/1 동등·SC-A2)·
# ``q=테스트`` 스모크로 검증한 뒤 045 stabilization 에서 default=True(live)로 채택했다(044 spec D8 갱신).
# ``.env`` 로 0 강제 가능. flip 전후 025 골든(recall@20·p@3): gate-on 시 weak-only precision ↑·recall 소폭 ↓.

__all__ = [
    "EVIDENCE_HIT_FILE_NAME_WEIGHT",
    "EVIDENCE_HIT_KEYWORDS_WEIGHT",
    "EVIDENCE_HIT_LABELS_WEIGHT",
    "EVIDENCE_HIT_CROSS_META_WEIGHT",
    "EVIDENCE_HIT_SUMMARY_WEIGHT",
    "EVIDENCE_KEYWORD_THRESHOLD",
    "EVIDENCE_NORMAL_THRESHOLD",
    "EVIDENCE_RESTRICTED_STRONG_THRESHOLD",
    "GENERIC_SINGLE_TERM_SEED",
    "SEARCH_GENERIC_TERM_SEED_EXTRA_ENV",
    "OS_BM25_OPERATOR_DEFAULT",
    "OS_CUTOFF_ENABLED_DEFAULT",
    "OS_CUTOFF_EPS_DEFAULT",
    "OS_CUTOFF_FLOOR_DEFAULT",
    "OS_FUSION_WEIGHTS_DEFAULT",
    "OS_KNN_SAMPLE_K",
    "OS_QUERY_NORM_ENABLED_DEFAULT",
    "OS_QUERY_NORM_MIN_WORD_TOKENS",
    "OS_QUERY_NORM_DECOMPOUND",
    "OS_QUERY_NORM_NOUN_POS",
    "OS_QUERY_NORM_STOPWORDS",
    "OS_RESULT_FLOOR_DEFAULT",
    "OS_RERANK_ENABLED_DEFAULT",
    "OS_RERANK_MODEL_DEFAULT",
    "OS_RERANK_TOP_R_DEFAULT",
    "OS_RERANK_TAU_DEFAULT",
    "SEARCH_EVIDENCE_DEBUG_DEFAULT",
    "SEARCH_EVIDENCE_RESCUE_ENABLED_DEFAULT",
]
