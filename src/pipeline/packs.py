"""v2 도메인 팩 — 슬롯별 전략 선택 + 정책.

**설계 의도**: 도메인별 분기를 코드 if/else 로 두지 않고, 팩이 "각 슬롯에 어떤 전략 이름을 쓸지"만
선언한다. 실제 구현은 StrategyRegistry(registry.py)에 등록된 Callable 이 담당하므로,
신규 도메인 추가 = 팩 정의 + 레지스트리 등록만으로 완결된다(core 파이프라인 수정 불요).

**슬롯 구분**:
- per_asset  : 파일 1건 처리(classify/extract/embed/persist). run_ingest 가 순서대로 호출.
- cross_asset: 자산 간 관계 제안(candidates/score/decide/persist_edges). run_relations 가 위임.

단계 B: per-asset 슬롯만. 일반·의료는 동일 전략(by_modality)을 쓰되 정책이 다르다.
도메인별 전략 차이(의료 by_signature/medclip)와 cross_asset 슬롯 포함(관계 파이프라인 seam).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainPack:
    """도메인 하나의 전략 배선표 + 정책 이름.

    **불변식**: per_asset/cross_asset 값은 반드시 DEFAULT_REGISTRY 에 등록된 이름이어야 한다.
    등록되지 않은 이름을 지정하면 resolve() 호출 시점에 KeyError 가 발생한다(조기 실패 아님).

    **주의**: ``is_symmetric`` 등 관계 속성은 팩이 아니라 relation_kind 카탈로그(DB)에 속한다.
    팩은 "어느 전략을 쓸지"만 고르고, 개별 관계 종류의 의미론(방향·대칭·is_causal 등)은 관리하지 않는다.
    """
    name: str
    per_asset: dict[str, str]    # 슬롯 → 전략 이름
    cross_asset: dict[str, str]  # 슬롯 → 전략 이름 (candidates/score/decide/persist_edges)
    policy: str                  # POLICIES 키(policy.py 참조)


# 일반 도메인 cross-asset 묶음.
# - candidates : 임베딩 코사인 유사도 top-k 후보 선별(deterministic).
# - score      : 온프레미스 LLM 이 JSON 엣지 목록을 생성(zero-shot, temperature=0).
# - decide     : confidence 임계 기반 자동 승인(propose 내부에서 처리, 별도 resolve 없음).
# - persist_edges: graph_edge + 관계 카탈로그 upsert.
# 단계 D 의료는 candidates 를 blocking_5keys 등으로 교체 예정.
_GENERAL_CROSS = {
    "candidates": "embedding_topk",
    "score": "llm_propose",
    "decide": "confidence",
    "persist_edges": "graph_upsert",
}

GENERAL_PACK = DomainPack(
    name="general",
    per_asset={"classify": "cascade_v1", "extract": "by_modality", "embed": "by_modality", "persist": "asset_upsert"},
    cross_asset=dict(_GENERAL_CROSS),
    policy="general_default",
)
MEDICAL_PACK = DomainPack(
    name="medical",
    # per_asset: 단계 D 이전에는 일반 전략(by_modality)으로 stopgap 처리.
    # DICOM 등 의료 포맷은 run_ingest 에서 status='deferred' 로 보류되므로
    # 실질적으로 일반 임베딩이 적재되는 자산은 비의료 파일에 한정된다.
    per_asset={"classify": "cascade_v1", "extract": "by_modality", "embed": "by_modality", "persist": "asset_upsert"},
    cross_asset=dict(_GENERAL_CROSS),   # 의료 전략은 단계 D 에서 교체
    policy="medical_strict",            # ForbidTag("external_llm") — 온프레미스 LLM 만 허용
)

_PACKS: dict[str, DomainPack] = {"general": GENERAL_PACK, "medical": MEDICAL_PACK}


def for_domain(label: str) -> DomainPack:
    """분류 라벨 → 도메인 팩. 미지정/review 는 일반으로 보수적 폴백.

    **폴백 의도**: 분류 미완료(review 상태)나 알 수 없는 라벨은 일반 팩으로 처리해
    파이프라인이 중단되지 않도록 한다. 의료 팩이 필요하면 label='medical' 이 명시적으로 와야 한다.
    """
    return _PACKS.get(label, GENERAL_PACK)
