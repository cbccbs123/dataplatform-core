"""검색 결과를 **모달리티별 독립 랭킹**으로 묶는다 — 완전 순수 함수(spec 010 grouped).

``search_service.search_hybrid`` 의 모달리티 버킷 결과를 전 모달리티 합치지 않고(평탄화 금지),
각 모달리티 섹션 안에서만 결정적으로 랭킹해 ``{modality: [rows]}`` 로 돌려준다.

왜 평탄화하지 않나(원인 ③ 회피)
    버킷마다 점수 산식이 달라(텍스트 하이브리드 alpha vs 영상 2단계+bm25) **모달리티 간 점수
    척도가 비교 불가**다. 이를 단일 ``similarity`` 로 합쳐 정렬하면 구조적으로 점수가 높은 영상이
    상단을 독식한다. 그래서 cross-modal 비교를 아예 하지 않고, **모달리티 내부에서만** 비교한다 —
    포탈은 어차피 섹션(text/image/video/audio)별로 데이터를 분류해 제공하면 되기 때문(설계 승인).

설계 요지
    - 정렬 키(결정성·헌법 3조): 버킷 내 ``(-round(similarity, 6), asset_id)``. similarity 정화는
      ``search_service._row_similarity`` 와 동일 규칙의 작은 사본(None/NaN/inf → 0.0).
    - 각 버킷은 ``limit_per_modality`` 까지만(섹션별 top-N, 페이징 없음 — 설계 승인).
    - ``exclude_domains`` 에 드는 ``domain_label`` 행은 해당 버킷에서 제외(FR-014, 의료 배제).
    - ``domain_label`` 을 응답 항목에 포함(042) — ``portal_api`` tier projection 에 사용.

DB·네트워크·파일 IO 없음. 표준 라이브러리만 import(완전 순수).
"""

from __future__ import annotations

import math
from typing import Any

# 결과 버킷 키 → 모달리티 라벨. search_service._MODALITY_BUCKETS 의 역매핑과 같은 표를 포탈
# 계층에 작은 사본으로 둔다(검색 서비스에 묶이지 않도록). 미지정 버킷은 키 그대로 노출.
_BUCKET_TO_MODALITY = {
    "text_documents": "text",
    "audio": "audio",
    "image": "image",
    "video": "video",
}


def _row_similarity(row: dict[str, Any]) -> float:
    """행의 ``similarity`` 를 유한 실수로 읽는다(None/NaN/inf/비수치 → 0.0).

    ``search_service._row_similarity`` 와 동일 규칙의 작은 사본이다. search_service 를 import 하면
    opensearch_search→임베더(torch) 체인을 끌어와 "완전 순수" 주장이 깨지므로, 정화 함수만 사본화한다.
    """
    value = row.get("similarity")
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def _basename(uri: str) -> str:
    """file_uri(전체 경로/URI)에서 파일명만 뽑는다(결정적·순수)."""
    if not uri:
        return ""
    tail = uri.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail.split("?", 1)[0].split("#", 1)[0] or tail


def _sort_key(item: dict[str, Any]) -> tuple[float, str]:
    """버킷 내 결정적 정렬 키: 유사도 내림차순(round 6), 동점은 asset_id 오름차순."""
    return (-round(_row_similarity(item), 6), str(item.get("asset_id", "")))


def _shape(row: dict[str, Any], modality: str) -> dict[str, Any]:
    """원시 검색 행 → 포탈 응답 항목.

    ``domain_label``(042): ``fetch_access_tiers``·``domain_floor`` 합성용.
    OS hit 에 없으면 ``general`` 폴백.
    """
    return {
        "asset_id": str(row.get("id", "")),
        "modality": modality,
        "similarity": _row_similarity(row),
        "summary": row.get("summary", "") or "",
        "file_name": _basename(str(row.get("file_uri", ""))),
        "domain_label": row.get("domain_label") or "general",
    }


def group_ranked(
    search_result: dict[str, Any],
    *,
    limit_per_modality: int,
    exclude_domains: frozenset[str] = frozenset(),
) -> dict[str, list[dict[str, Any]]]:
    """모달리티 버킷 dict 를 모달리티별 독립 랭킹 ``{modality: [rows]}`` 로 묶는다.

    각 항목: ``asset_id``·``modality``·``similarity``·``summary``·``file_name``·``domain_label``.
    ``(-round(similarity,6), asset_id)`` 로 정렬하고(cross-modal 병합 없음), 의료 배제 후 상위
    ``limit_per_modality`` 건만 담는다. 입력에 존재하는 버킷만 결과 키로 등장한다(빈 입력 → ``{}``).
    """
    if limit_per_modality < 1:
        raise ValueError("limit_per_modality 는 1 이상")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for bucket, rows in (search_result.get("results") or {}).items():
        modality = _BUCKET_TO_MODALITY.get(bucket, bucket)
        shaped: list[dict[str, Any]] = []
        for row in rows or []:
            label = row.get("domain_label")
            if label is not None and label in exclude_domains:
                continue  # 의료 등 배제 도메인(FR-014)
            shaped.append(_shape(row, modality))
        shaped.sort(key=_sort_key)
        grouped[modality] = shaped[:limit_per_modality]
    return grouped
