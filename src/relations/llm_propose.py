"""온프레미스 LLM으로 **관계 엣지 JSON** 을 받아 파싱·정규화한다.

흐름
    1. ``propose_edges_json``: 공통 seam ``src.llm.client.complete_json`` 으로 LLM 호출.
    2. ``parse_and_normalize_edges``: 응답 dict → ``persist`` 가 기대하는 엣지 dict 리스트(토피 정규화 포함).
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
    """LLM 관계 제안 응답이 스키마 불능(파싱 실패·빈 응답·엣지 구조 부재)일 때(069 P1-3).

    ``{}``(client 의 파싱실패 폴백)나 인식 키가 전혀 없는 응답을 정상 빈 제안(``{"edges": []}``)과
    구분해 예외로 승격한다 — ``run_relations`` 자산 단위 except 가 이를 받아 ``pending`` 재시도로
    보낸다(기존엔 edges=0·error=None 으로 흘러 **isolated 영구 오확정**되던 조용한 실패).
    """


def _configure_llm_logging() -> None:
    """중복 핸들러 방지 후 stderr 로 프롬프트·응답 로깅(디버그·감사용)."""
    if _LLM_LOG.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _LLM_LOG.addHandler(h)
    _LLM_LOG.setLevel(logging.INFO)
    _LLM_LOG.propagate = False


def propose_edges_json(prompt: str) -> dict[str, Any]:
    """관계 제안 프롬프트를 온프레미스 LLM에 넘겨 JSON dict 반환.

    069 P1-3: 파싱 실패·빈 응답(``{}``)·엣지 구조 부재를 ``RelationProposalParseError`` 로 승격 —
    "edges"/"items"/단일 엣지(``target_media_item_id``) 중 하나라도 있으면 정상 경로(빈 리스트
    ``{"edges": []}`` 는 **정상 빈 제안**으로 그대로 통과·isolated 유지). 예외는 호출자
    (run_relations 자산 단위 except)가 pending 재시도로 처리한다.
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
    """LLM confidence 를 결정적으로 [0,1] 범위에 가둔다(#2, FR-010, 헌법 3조).

    - 1.5 → 1.0, -0.3 → 0.0 처럼 범위를 벗어난 값은 양 끝으로 클램프.
    - NaN·무한대·파싱 불가 문자열·누락(None)은 비교가 무의미하므로 결정적 기본값 0.0.
      (자동승인 임계 판정·DB 저장이 항상 안정적인 수치를 받도록.)
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
    """
    LLM 루트 ``dict`` → ``register_new_relation_kinds`` / ``sync_graph_edges`` 가 순회할 **엣지 dict** 리스트.

    제외
        - ``target_media_item_id`` 없음(빈 값 포함)
    정규화
        - ``target_media_item_id``: 원본 식별자(asset_id UUID 문자열)를 문자열로 보존.
          유효성(UUID·후보 집합 소속)은 ``sync_asset_relation_edges`` 가 검증한다.
        - ``relation_type_code``: ``normalize_relation_type_code``
        - 토피: ``extract_topic_fields_from_edge`` (키 별칭·길이·맵은 그 함수 체인)
    """
    out: list[dict[str, Any]] = []
    for edge in parse_llm_edges(data):
        tid = edge.get("target_media_item_id")
        if tid is None or str(tid).strip() == "":
            continue
        # 065 FR-405: 아래 topic/subtopic 은 **관계 맥락 라벨(자산 주제 아님)** — 관계 LLM 이 이 쌍을
        #   설명하려 붙이는 메타이며, 자산 주제는 asset_topic 정본이 결정한다(엣지 topic 소비 중단·065).
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
