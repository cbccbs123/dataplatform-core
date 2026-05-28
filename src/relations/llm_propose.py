"""온프레미스 LLM으로 **관계 엣지 JSON** 을 받아 파싱·정규화한다.

흐름
    1. ``propose_edges_json``: 공통 seam ``src.llm.client.complete_json`` 으로 LLM 호출.
    2. ``parse_and_normalize_edges``: 응답 dict → ``persist`` 가 기대하는 엣지 dict 리스트(토피 정규화 포함).

보조
    ``propose_new_relation_type_ko_labels`` / ``normalize_type_ko_labels_from_llm_response``:
    신규 kind 코드에 대한 한글 라벨을 LLM 으로 받을 때 사용(주로 ``relation_kind`` 메타 보강).
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


def build_new_relation_type_ko_labels_prompt(*, type_code: str, edge_reason: str) -> str:
    """신규 kind 코드용 한글 라벨 LLM 프롬프트 본문(규칙·출력 스키마 포함)."""
    r = (edge_reason or "").strip()[:800]
    return f"""너는 JSON 객체만 출력하는 도우미다. 코드 블록·설명 금지.

아래 ``type_code`` 는 DB에 비활성으로 넣을 신규 **관계 종류** 코드(snake_case, 영문 소문자·밑줄)다.
``relation_kind`` 행에 넣을 한글 ``kind_name_ko`` 보조용과 ``description`` 첫 줄을 만든다(자동 등록 시 사용).

규칙:
- ``type_name_ko``: 간결한 한글(필요하면 띄어쓰기로 구간). 120자 이내. 코드 의미를 읽기 쉽게.
- ``description_ko``: ``type_name_ko`` 의 핵심을 ``·`` 로 잇고 **반드시** 문장 끝을 `` 도메인 연관`` 으로 맞춘다.
  예: type_name_ko ``게이밍 하드웨어`` → description_ko ``게이밍·하드웨어 도메인 연관``
  예: type_name_ko ``의료`` 한 단어 → ``의료 도메인 연관``

type_code: {type_code}
엣지 연결 근거(reason, 한국어): {r if r else "(없음)"}

출력 형식:
{{"type_name_ko":"...","description_ko":"... 도메인 연관"}}
"""


def propose_new_relation_type_ko_labels(type_code: str, edge_reason: str = "") -> dict[str, Any]:
    """신규 ``type_code`` 에 대한 한글 라벨·설명을 LLM JSON으로 받는다(주로 ``relation_kind`` 보강용)."""
    from src.llm.client import complete_json

    prompt = build_new_relation_type_ko_labels_prompt(type_code=type_code, edge_reason=edge_reason)
    return complete_json(prompt)


def normalize_type_ko_labels_from_llm_response(
    raw: dict[str, Any],
    *,
    type_code: str,
) -> tuple[str, str]:
    """
    LLM 응답에서 ``(type_name_ko, description_ko)`` 추출.

    ``description_ko`` 가 ``… 도메인 연관`` 패턴이 아니거나 길이 초과 시
    ``schema.type_label_from_kind_code`` 등 **규칙 기반 폴백**.
    """
    from src.relations.schema import description_ko_from_type_name_ko, type_label_from_kind_code

    suffix = " 도메인 연관"
    tn = str(raw.get("type_name_ko") or "").strip()
    desc = str(raw.get("description_ko") or "").strip()
    if tn and desc.endswith(suffix) and len(tn) <= 200:
        if len(tn) > 120:
            tn = tn[:120].rstrip()
        if len(desc) > 2000:
            desc = desc[:2000].rstrip()
        return tn, desc
    fb = type_label_from_kind_code(type_code)
    return fb, description_ko_from_type_name_ko(fb)


def parse_and_normalize_edges(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    LLM 루트 ``dict`` → ``sync_relation_catalog_from_llm_edges`` 가 순회할 **엣지 dict** 리스트.

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
