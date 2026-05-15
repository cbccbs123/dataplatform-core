"""LLM이 제시한 **관계 종류 코드**를 서버 규칙과 허용 집합에 맞춰 **최종 kind 코드**로 확정한다.

``entry`` / ``persist`` 에서 ``normalize_relation_type_code`` 로 정규화한 뒤,
이 모듈의 ``resolve_relation_type_code`` 로 **구조적 신호**(파생·인용)와 **후보 집합·허용 카탈로그**를 반영한다.

주의
    현재 MVP 에서는 ``citation_detected`` / ``derived_path_detected`` 가 항상 False 로 호출된다.
    향후 규칙·RAG 파이프라인에서 True 로 넘기면 ``references`` / ``derived_from`` 으로 강제할 수 있다.
"""

from __future__ import annotations

TYPE_CODE_REFERENCES = "references"
TYPE_CODE_DERIVED_FROM = "derived_from"


def resolve_relation_type_code(
    *,
    target_in_candidate_set: bool,
    citation_detected: bool = False,
    derived_path_detected: bool = False,
    llm_relation_kind_code: str | None = None,
    allowed_relation_kind_codes: frozenset[str] | None = None,
) -> str | None:
    """
    우선순위 규칙
        1. ``derived_path_detected`` 이면 ``derived_from`` (호출부가 신호를 켠 경우만 의미 있음).
        2. ``citation_detected`` 이면 ``references``.
        3. 타깃이 임베딩 후보 집합에 **없으면** None (이웃이 아닌 엣지는 확정하지 않음).
        4. 그 외에는 LLM 코드가 ``allowed_relation_kind_codes`` 에 있으면 그대로 채택, 아니면 None.

    Args:
        target_in_candidate_set: 호출부가 넘기는 ``tid in candidate_ids`` (임베딩 후보 집합 여부).
        llm_relation_kind_code: 이미 정규화된 snake_case (보통 ``canon``).
        allowed_relation_kind_codes: ``entry`` 에서 만든 ``llm_prompt_type_codes`` (DB∩상수).
    """
    if derived_path_detected:
        return TYPE_CODE_DERIVED_FROM
    if citation_detected:
        return TYPE_CODE_REFERENCES
    if not target_in_candidate_set:
        return None
    allowed = allowed_relation_kind_codes or frozenset()
    if llm_relation_kind_code and llm_relation_kind_code in allowed:
        return llm_relation_kind_code
    return None
