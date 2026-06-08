"""검색 결과 평탄화 + cursor 페이지네이션 — **완전 순수 함수**(plan 010 D-2).

``search_service.search_hybrid`` 의 모달리티 버킷 결과를 전 모달리티 합친 단일 결정적
랭킹으로 평탄화하고(``flatten_ranked``), offset 아닌 **opaque cursor**(마지막 본 항목의
정렬 키)로 페이지한다(``encode_cursor``/``decode_cursor``/``paginate``).

설계 요지
    - 정렬 키(결정성·헌법 3조): ``(-round(similarity, 6), asset_id)`` — 유사도 내림차순,
      동점은 asset_id 오름차순. similarity 정화는 ``search_service._row_similarity`` 재사용
      (None/NaN/inf → 0.0).
    - cursor payload ``{"s": <round(sim,6)>, "id": <asset_id>}`` → ``json.dumps(sort_keys=True)``
      → base64url. 키 순서 고정으로 동일 입력 2회 동일 토큰(결정성). 위조/깨진 토큰은 ``ValueError``.
    - 한계(plan R-1): ``search_hybrid`` 가 ``limit_per_bucket`` 까지만 반환하므로 본 페이징은
      "한 질의가 반환한 상위 결과 집합 내" 페이징이다(전체 코퍼스 무한 스크롤 아님).

DB·네트워크·파일 IO 없음. 표준 라이브러리 + ``search_service`` 정화 함수만 import.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from src.search.search_service import _row_similarity

# 결과 버킷 키 → 모달리티 라벨. sample_search_api._BUCKET_TO_MODALITY 와 동일 매핑을
# 재현한다(샘플 모듈에 묶이지 않도록 포탈 계층에 작은 사본을 둔다). 미지정 버킷은 키 그대로.
_BUCKET_TO_MODALITY = {
    "text_documents": "text",
    "audio": "audio",
    "image": "image",
    "video": "video",
}


def _basename(uri: str) -> str:
    """file_uri(전체 경로/URI)에서 파일명만 뽑는다(결정적·순수). sample_search_api 와 동형."""
    if not uri:
        return ""
    tail = uri.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail.split("?", 1)[0].split("#", 1)[0] or tail


def _sort_key(item: dict[str, Any]) -> tuple[float, str]:
    """결정적 정렬 키: 유사도 내림차순(round 6), 동점은 asset_id 오름차순."""
    return (-round(_row_similarity(item), 6), str(item.get("asset_id", "")))


def flatten_ranked(
    search_result: dict[str, Any],
    *,
    exclude_domains: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """모달리티 버킷 dict 를 전 모달리티 합친 단일 결정적 랭킹으로 평탄화한다.

    각 항목: ``{"asset_id","modality","similarity","summary","file_name"}``. 정렬 키는
    ``(-round(similarity,6), asset_id)`` (유사도 desc·asset_id asc). similarity 는
    ``search_service._row_similarity`` 로 정화(None/NaN/inf → 0.0).
    행에 ``domain_label`` 이 있고 ``exclude_domains`` 에 들면 제외(없으면 포함, FR-014).
    """
    flat: list[dict[str, Any]] = []
    for bucket, rows in (search_result.get("results") or {}).items():
        modality = _BUCKET_TO_MODALITY.get(bucket, bucket)
        for row in rows or []:
            label = row.get("domain_label")
            if label is not None and label in exclude_domains:
                continue  # 의료 등 배제 도메인(FR-014)
            flat.append(
                {
                    "asset_id": str(row.get("id", "")),
                    "modality": modality,
                    "similarity": _row_similarity(row),
                    "summary": row.get("summary", "") or "",
                    "file_name": _basename(str(row.get("file_uri", ""))),
                }
            )
    flat.sort(key=_sort_key)
    return flat


def encode_cursor(payload: dict[str, Any]) -> str:
    """cursor payload(``{"s","id"}``)를 base64url(json) opaque 토큰으로 인코딩한다.

    ``json.dumps(sort_keys=True)`` 로 키 순서를 고정해 동일 입력 2회 동일 토큰(결정성·헌법 3조).
    """
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_cursor(token: str) -> dict[str, Any]:
    """opaque cursor 토큰을 payload dict 로 디코딩한다. 위조/깨진 토큰은 ``ValueError``.

    base64url 디코드 실패·JSON 파싱 실패·필수 키(``s``/``id``) 누락·타입 불일치를 모두
    ``ValueError`` 로 거부한다(API 가 400 으로 매핑, Edge Case).
    """
    if not isinstance(token, str) or not token:
        raise ValueError("빈 cursor 토큰")
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"손상된 cursor 토큰: {exc}") from exc
    if not isinstance(payload, dict) or "s" not in payload or "id" not in payload:
        raise ValueError("cursor payload 형식 오류(필수 키 s/id 누락)")
    if not isinstance(payload["s"], (int, float)) or isinstance(payload["s"], bool):
        raise ValueError("cursor payload 의 s 가 수치가 아님")
    return payload


def paginate(
    ranked: list[dict[str, Any]],
    *,
    cursor: str | None = None,
    page_size: int,
) -> dict[str, Any]:
    """결정적 랭킹을 cursor 기준으로 한 페이지(``page_size`` 개)만큼 잘라 반환한다.

    반환: ``{"items": [...], "next_cursor": str | None}``.
    cursor 없으면 상위 ``page_size``. cursor 있으면 그 정렬 키 ``(-round(s,6), id)`` 보다
    **엄격히 뒤**인 첫 항목부터 ``page_size``. 마지막 페이지면 ``next_cursor=None`` (US1 Acc 2).
    동일 cursor 2회 동일 페이지(결정성·SC-002) — cursor 가 정렬 키 기반이라 안정적.
    """
    if page_size < 1:
        raise ValueError("page_size 는 1 이상")

    start = 0
    if cursor is not None:
        payload = decode_cursor(cursor)
        cursor_key = (-round(float(payload["s"]), 6), str(payload["id"]))
        # 정렬 키가 cursor 키보다 엄격히 뒤인 첫 항목을 찾는다(ranked 는 같은 키로 정렬됨).
        start = len(ranked)
        for i, item in enumerate(ranked):
            if _sort_key(item) > cursor_key:
                start = i
                break

    page = ranked[start : start + page_size]
    next_cursor: str | None = None
    # 페이지가 비지 않고 뒤에 항목이 더 남아 있으면 마지막 본 항목의 정렬 키로 next_cursor 발급.
    if page and start + page_size < len(ranked):
        last = page[-1]
        next_cursor = encode_cursor(
            {"s": round(_row_similarity(last), 6), "id": str(last.get("asset_id", ""))}
        )
    return {"items": page, "next_cursor": next_cursor}
