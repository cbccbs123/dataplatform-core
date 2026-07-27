"""온프레미스 LLM으로 **관계 엣지 JSON** 을 받아 파싱·정규화한다.

흐름
    1. ``propose_edges_json``: 공통 seam ``src.llm.client.complete_json`` 으로 LLM 호출.
    2. ``parse_and_normalize_edges``: 응답 dict → 영속화 계층이 기대하는 엣지 dict 리스트
       (타깃 id·관계 코드·신뢰도·주제 라벨 정규화 포함).
"""

from __future__ import annotations

import json
import logging
import math
import sys
from typing import Any

from src.relations.schema import (
    extract_topic_fields_from_edge,
    normalize_relation_type_code,
    parse_llm_edges,
)

_LLM_LOG = logging.getLogger("meta_extract.relations.llm")


class RelationProposalParseError(RuntimeError):
    """LLM 관계 제안 응답을 알아볼 수 없을 때(파싱 실패·빈 응답·엣지 구조 부재).

    ``{}``(client 의 파싱실패 폴백)나 인식 키가 전혀 없는 응답을 정상 빈 제안(``{"edges": []}``)과
    구분해 예외로 승격한다 — ``run_relations`` 자산 단위 except 가 이를 받아 ``pending`` 재시도로
    보낸다 — 예외로 올리지 않으면 "관계 없음"으로 영구 확정돼 재시도 기회를 잃는다.
    """


def _configure_llm_logging() -> None:
    """LLM 전용 로거를 stderr 에 한 번만 붙인다(디버그·감사용).

    이미 핸들러가 있으면 아무 것도 하지 않는다 — 호출마다 붙이면 같은 줄이 중복 출력된다.

    ⚠️ 이 로거는 **LLM 응답 전문을 평문으로 남긴다**. 의료 도메인(단계 D) 운용 시에는 PHI 가
    로그로 새지 않도록 마스킹·레벨 조정이 선행되어야 한다(미착수).
    """
    if _LLM_LOG.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _LLM_LOG.addHandler(h)
    _LLM_LOG.setLevel(logging.INFO)
    _LLM_LOG.propagate = False


def propose_edges_json(prompt: str) -> dict[str, Any]:
    """관계 제안 프롬프트를 온프레미스 LLM에 넘겨 응답 JSON(dict)을 받는다.

    **외부 호출**이 일어난다(단일 seam ``complete_json``·temperature=0). 응답 전문을 stderr 로
    로깅한다.

    "제안이 없음"과 "응답이 망가짐"을 반드시 구분한다 — ``{"edges": []}`` 는 **정상 빈 제안**이라
    그대로 통과시키고, 빈 dict 나 인식 키가 하나도 없는 응답은 예외로 올린다. 구분하지 않으면
    조용한 실패가 "관계 없음(isolated)"으로 영구 확정되어 재시도 기회를 잃는다.

    Args:
        prompt: ``build_relation_proposal_prompt`` 가 만든 단일 프롬프트 문자열.

    Returns:
        LLM 응답 dict. ``edges``/``items``/단일 엣지(``target_media_item_id``) 중 하나는 반드시 있다.

    Raises:
        RelationProposalParseError: 응답이 비었거나 엣지 구조가 전혀 없을 때. 호출자
            (``run_relations`` 의 자산 단위 except)가 받아 ``pending`` 재시도로 보낸다.
    """
    from src.llm.client import complete_json

    out = complete_json(prompt)
    _configure_llm_logging()
    _LLM_LOG.info("response_json=%s", json.dumps(out, ensure_ascii=False))
    if not out or not any(k in out for k in ("edges", "items", "target_media_item_id")):
        raise RelationProposalParseError(
            "LLM 관계 제안 응답 파싱 실패(빈/스키마 불능) — pending 재시도 대상: "
            + json.dumps(out, ensure_ascii=False)[:200]
        )
    return out


def _clamp_confidence(raw: Any) -> float:
    """LLM 신뢰도를 [0, 1] 범위 안으로 가둔다(결정적).

    - 1.5 → 1.0, -0.3 → 0.0 처럼 범위를 벗어난 값은 양 끝으로 클램프.
    - NaN·무한대·파싱 불가 문자열·누락(None)은 비교가 무의미하므로 결정적 기본값 0.0.
      (자동승인 임계 판정·DB 저장이 항상 안정적인 수치를 받도록.)

    Args:
        raw: LLM이 준 신뢰도 값. 숫자·문자열·``None`` 무엇이든 받는다.

    Returns:
        0.0~1.0 사이 실수. 판정 불가면 0.0.
    """
    if raw is None:
        return 0.0
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return 0.0
    # NaN·±inf 는 max/min 비교가 비결정적이므로 0.0 으로 강제.
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, x))


def parse_and_normalize_edges(data: dict[str, Any]) -> list[dict[str, Any]]:
    """LLM 응답 dict 를 영속화 계층이 순회할 **엣지 dict 리스트**로 정규화한다.

    순수 함수(DB·네트워크 없음). 잘못된 항목은 예외 대신 **조용히 건너뛴다** — 한 엣지가 망가졌다고
    나머지 제안까지 버릴 이유가 없기 때문이다.

    제외
        - ``target_media_item_id`` 없음(빈 값 포함)
    정규화
        - ``target_media_item_id``: 원본 식별자(asset_id UUID 문자열)를 문자열로 보존.
          UUID 형식·후보 집합 소속 검증은 ``sync_graph_edges`` 가 맡는다.
        - ``relation_type_code``: ``normalize_relation_type_code``
        - 주제 라벨: ``extract_topic_fields_from_edge`` (키 별칭·길이 제한은 그 함수 체인)

    Args:
        data: LLM 응답 루트 dict(``propose_edges_json`` 결과).

    Returns:
        엣지 dict 리스트. 각 항목은 ``target_media_item_id``·``relation_type_code``·``confidence``
        (0~1)·``reason``·주제 4필드를 갖는다. 유효한 엣지가 없으면 빈 리스트.
    """
    out: list[dict[str, Any]] = []
    for edge in parse_llm_edges(data):
        tid = edge.get("target_media_item_id")
        if tid is None or str(tid).strip() == "":
            continue
        # 여기의 topic/subtopic 은 **이 쌍(관계)의 맥락 라벨**이지 자산의 주제가 아니다 —
        # 자산 주제는 별도 정본이 정한다.
        tk, sk, ten, sen = extract_topic_fields_from_edge(edge)
        row = {
            "target_media_item_id": str(tid).strip(),
            "relation_type_code": normalize_relation_type_code(
                edge.get("relation_type_code")
            ),
            "confidence": _clamp_confidence(edge.get("confidence")),
            "reason": str(edge.get("reason") or "").strip(),
            "topic_ko": tk,
            "subtopic_ko": sk,
            "topic_en": ten,
            "subtopic_en": sen,
        }
        out.append(row)
    return out
