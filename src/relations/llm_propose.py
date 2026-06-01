"""온프레미스 LLM으로 **관계 엣지 JSON** 을 받아 파싱·정규화한다.

흐름
    1. ``propose_edges_json``: 공통 seam ``src.llm.client.complete_json`` 으로 LLM 호출.
    2. ``parse_and_normalize_edges``: 응답 dict → ``persist`` 가 기대하는 엣지 dict 리스트(토피 정규화 포함).
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from src.relations.schema import (
    extract_topic_fields_from_edge,
    normalize_relation_type_code,
    parse_llm_edges,
)

_LLM_LOG = logging.getLogger("meta_extract.relations.llm")


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
    """관계 제안 프롬프트를 온프레미스 LLM에 넘겨 JSON dict 반환."""
    from src.llm.client import complete_json

    out = complete_json(prompt)
    _configure_llm_logging()
    _LLM_LOG.info("response_json=%s", json.dumps(out, ensure_ascii=False))
    return out


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
        tk, sk, ten, sen = extract_topic_fields_from_edge(edge)
        row = {
            "target_media_item_id": str(tid).strip(),
            "relation_type_code": normalize_relation_type_code(
                edge.get("relation_type_code")
            ),
            "confidence": float(edge.get("confidence") or 0.0),
            "reason": str(edge.get("reason") or "").strip(),
            "topic_ko": tk,
            "subtopic_ko": sk,
            "topic_en": ten,
            "subtopic_en": sen,
        }
        out.append(row)
    return out
