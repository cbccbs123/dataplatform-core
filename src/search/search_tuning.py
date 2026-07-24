"""OS 검색 튜닝 파라미터 묶음 (069 US-E FR-E5② — 인자 축소·getattr 릴레이 제거).

종전 ``search_service`` 가 ``getattr(cfg, …)`` 12곳으로 설정값을 풀어 ``search_assets_os`` 에 14개
kwarg 로, 다시 ``apply_bucket_policy`` 에 12개로 전달했다. 이 **config-파생 튜닝값**을 한 frozen
dataclass 로 묶어 ``SearchTuning.from_settings(cfg)`` 1회 해소로 대체한다 — 호출부 인자 수·getattr
릴레이가 함께 줄고, 오타는 정적 검사(직접 속성 접근)로 잡힌다.

per-request·주입 인자(client·query·modalities·k·channel·index·embed_fn·rerank_fn·query_norm_fn·
search_filters·search_policy·search_mode·exclude_medical)는 **튜닝이 아니라**
호출마다 달라지므로 이 묶음에 넣지 않는다. 기본값은 ``search_constants`` 단일 출처(F1)와 일치한다 —
``SearchTuning()`` 무인자 생성 = 현행 기본 동작(회귀 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import search_constants


@dataclass(frozen=True, slots=True)
class SearchTuning:
    """OS 검색 게이트·컷·rerank·융합·질의정규화 튜닝(전부 config 파생·기본=search_constants).

    ``search_assets_os`` 가 쓰는 필드(weights)와 ``apply_bucket_policy`` 가 쓰는 필드(cutoff·rerank·
    evidence·about·bm25_operator)를 한데 묶는다. 두 함수는 각자 필요한 필드만 읽는다(bm25_operator 는
    양쪽 공용 — BM25 body 생성 + lexical rescue 분기).

    ⚠️ ``query_norm_enabled`` 는 **의도적으로 제외**한다 — 유일 프로덕션 호출부 ``search_service`` 가
    검색 직전 ``os_normalize_query`` 로 질의를 **먼저** 정규화한 뒤 그 결과를 ``search_assets_os`` 에
    넘기므로(query_norm_enabled 미전달=기본 off), 이 필드를 tuning 으로 넘기면 **이중 정규화**가 된다.
    질의정규화는 tuning 이 아니라 search_assets_os 의 독립 파라미터로 남긴다(현행 동작 보존).
    """

    weights: tuple[float, float] = search_constants.OS_FUSION_WEIGHTS_DEFAULT
    cutoff_enabled: bool = search_constants.OS_CUTOFF_ENABLED_DEFAULT
    cutoff_eps: float = search_constants.OS_CUTOFF_EPS_DEFAULT
    cutoff_floor: float = search_constants.OS_CUTOFF_FLOOR_DEFAULT
    result_floor: float = search_constants.OS_RESULT_FLOOR_DEFAULT
    bm25_operator: str = search_constants.OS_BM25_OPERATOR_DEFAULT
    rerank_enabled: bool = search_constants.OS_RERANK_ENABLED_DEFAULT
    rerank_top_r: int = search_constants.OS_RERANK_TOP_R_DEFAULT
    rerank_tau: float = search_constants.OS_RERANK_TAU_DEFAULT
    rerank_model: str = search_constants.OS_RERANK_MODEL_DEFAULT
    about_filter_enabled: bool = search_constants.SEARCH_ABOUT_FILTER_ENABLED_DEFAULT
    evidence_rescue_enabled: bool = search_constants.SEARCH_EVIDENCE_RESCUE_ENABLED_DEFAULT
    evidence_debug: bool = search_constants.SEARCH_EVIDENCE_DEBUG_DEFAULT

    @classmethod
    def from_settings(cls, cfg: Any) -> SearchTuning:
        """PipelineSettings 에서 검색 튜닝값을 **1회** 해소한다.

        069 US-E FR-E4(PR4b): 검색 튜닝 필드는 ``cfg.search`` 하위로 중첩됐다 — ``cfg.search.<field>``
        직접 접근으로 읽는다(종전 getattr 릴레이 폐지; 오타는 정적 검사로 잡힌다). SearchTuning 의 짧은
        이름(weights·cutoff_eps 등)과 SearchConfig 의 필드명(fusion_weights·os_cutoff_eps 등)이 달라
        매핑은 명시한다. ``cutoff_enabled`` 의 디버그 우회(``disable_os_cutoff``)는 호출부가
        ``dataclasses.replace(tuning, cutoff_enabled=False)`` 로 덮으므로 여기선 cfg 값만 읽는다.
        """
        s = cfg.search
        return cls(
            weights=s.fusion_weights,
            cutoff_enabled=s.os_cutoff_enabled,
            cutoff_eps=s.os_cutoff_eps,
            cutoff_floor=s.os_cutoff_floor,
            result_floor=s.os_result_floor,
            bm25_operator=s.os_bm25_operator,
            rerank_enabled=s.os_rerank_enabled,
            rerank_top_r=s.os_rerank_top_r,
            rerank_tau=s.os_rerank_tau,
            rerank_model=s.os_rerank_model,
            about_filter_enabled=s.about_filter_enabled,
            evidence_rescue_enabled=s.evidence_rescue_enabled,
            evidence_debug=s.evidence_debug,
        )
