"""OpenSearch 검색 융합·게이트·컷의 기본값 단일 출처(027 F1 — 순수 상수·의존 0).

027 독립 비평이 발견한 즉시 결함 F1: 임계·가중치 기본값이 **3곳에 복제·구식화**돼 있었다
(settings resolver·검색 모듈 중복 상수·calibrate 스크립트 — 코드 0.15/0.43 vs 운영 0.17/0.50,
새 환경에 설정 없이 배포하면 구식 코드 기본값이 적용되는 지뢰). 이 모듈을 **유일 출처**로 두고
``src.config.settings`` resolver·``src.search.opensearch_search``·calibrate 가 공유한다(FR-006).

순수 상수만 둔다(import 0·IO 0) — settings 미초기화 환경(순수 단위)에서도 안전하게 참조된다.
임계는 **코사인 스케일**(클라이언트 융합 전환으로 정규화 스케일 임계 4종이 폐기됨, 027). 실측
확정치(EPS·FLOOR·RESULT_FLOOR)는 G4 실OS 재보정 후 **여기 1곳만** 갱신한다.
044 evidence 가중·rescue 임계·``SEARCH_EVIDENCE_*`` 토글 기본값도 동일 원칙.
"""

from __future__ import annotations

# ── 버킷 게이트(robust baseline) 기본값 ──────────────────────────────────────
# OS 검색은 기본 게이트 on(프로덕션 경로) — settings 미설정 환경에서도 무관 결과 표출을 차단한다.
OS_CUTOFF_ENABLED_DEFAULT: bool = True
# 상대 신호 임계: keep = (top − baseline) ≥ EPS. baseline = '하위 절반 평균'(robust — 밀집 토픽
# '충전' 오컷 구조 해결). **G4 실OS 재보정 확정치**(있음 Δmin 0.224 vs 없음 Δmax 0.189 분포 기준).
OS_CUTOFF_EPS_DEFAULT: float = 0.18
# 절대 backstop 임계: keep 의 두 번째 조건 top ≥ FLOOR(코퍼스 전체가 평탄할 때의 느슨한 하한).
# ⚠️ **2026-08-03 재보정: 0.50 → 0.35**(아래 재보정 주석). 0.50 은 정답이 든 버킷을 통째로 막았다.
OS_CUTOFF_FLOOR_DEFAULT: float = 0.35

# ── per-result 컷(단일 코사인 스케일) 기본값 ────────────────────────────────
# 행 유지 = BM25 매칭(어휘 증거) OR 원시 코사인 ≥ RESULT_FLOOR. 024 의 모달리티별 정규화 스케일 임계
# 4종을 대체하는 전역 1개(코사인 스케일). 원 확정치는 027-G4 의 0.55.
# ⚠️ **2026-08-03 재보정: 0.55 → 0.20**(아래 재보정 주석). 행 단위 재판단을 사실상 접고, "답이
# 있나 없나" 판단은 게이트(EPS·FLOOR)에 맡긴다 — 두 장치가 같은 일을 이중으로 하며 정답을 깎고 있었다.
OS_RESULT_FLOOR_DEFAULT: float = 0.20

# ── 2026-08-03 컷오프 재보정 (골든 386질의 실측 · 사용자 결정 B안) ──────────
# **계기**: 027(2026-06-11)에서 확정한 EPS 0.18·FLOOR 0.50·RESULT_FLOOR 0.55 는 그 시점 코퍼스에
# 맞춰진 값이다. 그 뒤 유사도 점수 분포를 바꾸는 변경이 6건 있었고(026 추출품질·BGE 전환 / 049 VLM
# 요약 v2 / 050 영상 재캡션 / 062 원격 임베딩 채널 / 065·068 주제 정본화·보정) **2026-07-09 에
# 코퍼스 1,148건이 전량 재수집**됐다. 요약이 바뀌면 임베딩이 바뀌고, 임베딩이 바뀌면 코사인 분포가
# 바뀌며, 이 세 값은 그 분포에 걸린 숫자다.
#
# **감지하지 못한 이유**: 골든의 정답이 `asset_id` 로 동결돼 있어 재수집 후 `recall@20 = 0.0000` 을
# 내는 무효 상태였다. 계기판이 죽어 있어 4주간 아무도 몰랐다. → 골든 재생성(386질의) 후 첫 측정에서
# **컷오프가 정답의 13.8%(191/1,388건)를 버리고 있음**이 드러났고, 그중 179건(94%)은 임계만 풀면
# 되살아났다 — 회수 엔진은 이미 찾아냈고 컷오프가 버린 것이다.
#
# **측정 방법**: 컷오프는 OS 조회 결과에 사후 적용되는 순수 로직이므로, 원본 융합행·top·baseline 을
# 한 번 포착해 `apply_bucket_policy` 를 임계만 바꿔 재생했다(조합 324개 · 재생이 라이브와 동일함을
# 5질의로 선검증). 상세는 `docs/검색_골든_재생성_채점_20260803.md`.
#
# **채택(B안 · 회수 우선)**: EPS 0.18 유지 · FLOOR 0.50→0.35 · RESULT_FLOOR 0.55→0.20
#   회수 0.8945→0.9471(+5.3pp) · 완전회수 208→261질의 · 조합질의 회수 0.483→0.612(+12.9pp)
#   부재질의 차단 23/24 **유지**(악화 없음) · p@3 0.8833→0.8800(−0.33pp) · 평균 반환 4.6→5.4행
#   자연어 질의 부작용 0(빈결과 2/10 불변 · 결과만 증가)
# 대안 A안(EPS 0.24·차단 24/24·회수 0.9099)은 차단이 완벽해지는 대신 회수 개선폭이 1/3 이라 미채택.
#
# ⚠️ **이 값은 코퍼스 의존이다.** 코퍼스·임베딩·요약 프롬프트가 바뀌면 다시 어긋난다. 재보정 트리거
# 자동화는 **spec 082**(등재만·미착수) — 그것이 붙기 전까지는 코퍼스 변경 시 사람이 재보정해야 한다.

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

# 075: 정규화 방식 선택. enabled=on 일 때 "morph"(nori 형태소·072·LLM 0) 또는 "llm"(gemma·029 경로·
# 검색시점 LLM)을 고른다. 코드 기본은 morph(072 동작 불변·회귀 0)이나, 076 strict 프롬프트 측정에서
# llm 이 형태소를 재역전(자연어 nDCG 0.438→0.531)해 **운영은 llm 을 선택**한다(.env.dev). 방식 선택은
# 운영자 몫(025 bm25_operator·029 토글 관례 동형).
OS_QUERY_NORM_METHOD_DEFAULT: str = "morph"

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

# ── 026 T006(FR-004): nori 인덱스 analyzer 외래어 고유명사 사전 기본 목록(단일 출처) ──────
# 색인 시 커스텀 nori analyzer 의 user_dictionary_rules 로 실어 외래어 고유명사가 분해되지 않게 한다
# ('아이패드'→'아이'+'패드' 분해 시 '아이패드' 정확매칭 무력화·가짜매칭 발생). settings resolver
# (``resolve_opensearch_nori_user_words``·운영 단일 출처·CSV 오버라이드)와 순수 인덱스 빌더
# (``opensearch_sync.build_index_body`` 기본)가 **이 상수 1벌**을 공유한다(069 T302·D6 — 예전엔 두 모듈에
# 목록이 복제돼 계약 테스트가 동치를 감시했으나, 단일 출처화로 드리프트 원천 차단). **값 불변**(재색인 시
# analyzer 동일). 닫힌·안정적 목록(대표 외래어 브랜드명)이라 자산 증가와 무관하게 거의 불변.
NORI_USER_WORDS_DEFAULT: tuple[str, ...] = (
    "아이패드",
    "아이폰",
    "스마트워치",
    "맥세이프",
    "에어팟",
    "갤럭시",
    "애플워치",
)

# ── 073: aboutness OR-증거 필터 상수(단일 출처 F1) ─────────────────────────────
# 적재시 확정한 about 개체 + keywords 를 증거로, 질의 개체와 무증거 행을 버킷에서 걸러낸다
# (검색시점 LLM 0·전체 노출 깊이 적용). 측정(2026-07-13): @10 무관 −4~7%p·연관 무손실.
# 기본 off — dev 는 aboutness 백필 완료 후 .env opt-in(백필 전엔 about 부재로 실효 없음·fail-safe 는 동작).
SEARCH_ABOUT_FILTER_ENABLED_DEFAULT: bool = False
# 판별력 명사 선별 임계(질의별 상대 DF): 질의 명사가 후보 행의 이 비율을 초과해 keywords 에 등장하면
# 흔한 명사('기술'·'풍경')로 보고 kmatch 에서 제외한다. 정적 불용어 사전 불요(자산 증가 자가적응).
ABOUT_FILTER_NOUN_MAX_MATCH_RATIO: float = 0.5

# ── 074: 검색시점 top-3 개별 LLM 검증(L2) 상수(단일 출처 F1) ──────────────────
# 073(L1) 후 잔존 자연어 상위3 무관을, 노출 직전 상위 3 자산만 gemma **개별** 병렬 판정해 제거한다
# (측정 2026-07-13: 개별=무관 44.7→0%·연관 1.36→1.60 / 배치는 판정 흔들림(일치율 66~77%)으로 기각).
# 검색시점 LLM 은 021 FR-004 의 029 거버넌스 토글 개정 선례를 따른다 — 기본 off·opt-in·단일 seam·temp=0.
# 2026-07-14: 독립지표 재평가로 운영 off 유지·코드 보존(L3 재접근 예정·ADR 2026-07-14-llm-verify-off-reeval).
SEARCH_LLM_VERIFY_ENABLED_DEFAULT: bool = False
# 검증 대상 상위 자산 수. 3=측정 실용점(동시 30검색 p95 1.06s) — 전수 검증은 동시 4~5검색 포화(기각).
SEARCH_LLM_VERIFY_TOP_N: int = 3
# 전체 데드라인(초). 초과·오류 시 **전량 폴백**(미검증 결과 그대로 — 부분 적용은 타이밍 의존
# 비결정이라 금지·헌법 §3). 실측 e2e 0.31s 라 정상 부하에선 미발동.
SEARCH_LLM_VERIFY_DEADLINE_S: float = 1.5
# (정규화 질의, asset_id)→판정 프로세스 내 캐시 상한(초과 시 오래된 항목부터 제거). temp=0 결정적이라
# TTL 불요 — 같은 쌍은 첫 판정으로 고정(라이브 경로 near-tie 섭동 완화·029 ADR 논점).
SEARCH_LLM_VERIFY_CACHE_MAX: int = 50_000

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
# ``mode=auto``(기본) — **전체** evidence_score 하한. 예: keywords만 hit(3.0) → keep; summary+cross_meta(1.0) → drop.
EVIDENCE_KEYWORD_THRESHOLD: float = 0.7
# ``mode=keyword``(사용자 명시) — weak evidence 도 rescue 허용(낮은 하한). 포탈 ``GET /search?mode=keyword`` 와 쌍.
# (2026-07-24 mode 슬림: generic seed·restricted 임계·keyword 안내 suggestion 제거.)
#
# ── env 토글 기본값 (settings ``SEARCH_EVIDENCE_*`` 로 덮어씀) ───────────────
SEARCH_EVIDENCE_DEBUG_DEFAULT: bool = False
# True → ``fuse_hybrid`` 결과 행에 debug 필드 부착(FR-405): ``matched_queries``,
# ``evidence_score``, ``strong_evidence_score``, ``gate_passed``, ``keep_reason``.
# 포탈·run_search JSON 관측용. 프로덕션 기본 off(응답 부풀림·내부 _name 노출 최소화).
SEARCH_EVIDENCE_RESCUE_ENABLED_DEFAULT: bool = True
# lexical rescue **정책 집행** 스위치. False=027 호환(어휘 BM25 hit 이면 게이트 실패 후에도 전부 keep,
# ``keep_reason=legacy_lexical``). True=``lexical_rescue_keep`` live(weak-only drop·keyword 관대/그 외 일반 임계).
# 044 G2 merge 시점엔 회귀 방지로 default=0(shadow)였고, 045 G2b 골든(RESCUE 0/1 동등·SC-A2)·
# ``q=테스트`` 스모크로 검증한 뒤 045 stabilization 에서 default=True(live)로 채택했다(044 spec D8 갱신).
# ``.env`` 로 0 강제 가능. flip 전후 025 골든(recall@20·p@3): gate-on 시 weak-only precision ↑·recall 소폭 ↓.

__all__ = [
    "ABOUT_FILTER_NOUN_MAX_MATCH_RATIO",
    "SEARCH_ABOUT_FILTER_ENABLED_DEFAULT",
    "SEARCH_LLM_VERIFY_ENABLED_DEFAULT",
    "SEARCH_LLM_VERIFY_TOP_N",
    "SEARCH_LLM_VERIFY_DEADLINE_S",
    "SEARCH_LLM_VERIFY_CACHE_MAX",
    "EVIDENCE_HIT_FILE_NAME_WEIGHT",
    "EVIDENCE_HIT_KEYWORDS_WEIGHT",
    "EVIDENCE_HIT_LABELS_WEIGHT",
    "EVIDENCE_HIT_CROSS_META_WEIGHT",
    "EVIDENCE_HIT_SUMMARY_WEIGHT",
    "EVIDENCE_KEYWORD_THRESHOLD",
    "EVIDENCE_NORMAL_THRESHOLD",
    "NORI_USER_WORDS_DEFAULT",
    "OS_BM25_OPERATOR_DEFAULT",
    "OS_CUTOFF_ENABLED_DEFAULT",
    "OS_CUTOFF_EPS_DEFAULT",
    "OS_CUTOFF_FLOOR_DEFAULT",
    "OS_FUSION_WEIGHTS_DEFAULT",
    "OS_KNN_SAMPLE_K",
    "OS_QUERY_NORM_ENABLED_DEFAULT",
    "OS_QUERY_NORM_METHOD_DEFAULT",
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
