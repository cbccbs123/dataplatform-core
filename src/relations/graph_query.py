"""그래프 **read seam** — 자산 하나의 관계 이웃을 양방향·정규화해 조회.

왜 이 seam이 필요한가 (ADR 2026-05-28)
    대칭 kind(``relation_kind.is_symmetric=True``)는 ``graph_persist._canonical_pair`` 가
    ``(min(node_id), max(node_id))`` 캐논 순서 **단일 행**으로 저장한다. 그래서 쓰기는
    중복 없이 깔끔하지만, "자산 X의 이웃"을 순진하게 ``WHERE src_node = X`` 로만 찾으면
    **X가 큰 쪽(dst)으로 접힌 대칭 엣지를 통째로 누락**한다. 또 ``is_symmetric``·``kind_code``
    는 엣지 행에 없고 ``relation_kind`` 에만 있다.
    → 반드시 ``src.asset_id = X OR dst.asset_id = X`` 양방향 매칭 + ``relation_kind`` 조인 +
      질의 자산 관점의 방향 정규화가 필요하다. 이 read 경로를 한 곳에 고정해, 향후 검색·상세·
      묶음 소비자가 순진한 단방향 쿼리를 재발명하지 못하게 막는다(FR-006/008 공유 seam).

읽기 전용
    스키마·쓰기 경로 변경 0(헌법 6조). VIEW 미사용 — ``src/relations/`` 파이썬 쿼리 함수 관례.
"""
from __future__ import annotations

import os
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

# 양 끝점 asset 역조인(node_kind='asset')으로 src/dst asset_id 를 함께 끌어온 뒤,
# 파이썬에서 질의 자산 관점으로 방향·이웃을 정규화한다.
# 057 FR-102: 이웃 표시필드(file_name·modality) 하향을 위해 양끝 node→asset 조인을 하나 더 건다
#   (review._build_review_where 와 동일 패턴). modality·fs_path 는 asset 에만 있고
#   기존 소비자를 깨지 않도록 WHERE·정렬·기존 컬럼은 불변 — 필드/조인 추가만.
# ORDER BY: confidence DESC NULLS LAST 동점 시 순서가 불안정하므로 edge_id 2차 키로
# 결정성(헌법 3조)을 보장한다(ADR 원안 대비 강화 — plan R-3).
_FETCH_RELATIONS_SQL = """
SELECT ge.edge_id, rk.kind_code, rk.is_symmetric,
       ge.confidence, ge.reason, ge.topic, ge.status,
       sn.asset_id AS src_asset, dn.asset_id AS dst_asset,
       sa.modality AS src_modality, sa.fs_path AS src_fs_path,
       da.modality AS dst_modality, da.fs_path AS dst_fs_path
FROM graph_edge ge
JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
JOIN node sn ON sn.node_id = ge.src_node AND sn.node_kind = 'asset'
JOIN node dn ON dn.node_id = ge.dst_node AND dn.node_kind = 'asset'
JOIN asset sa ON sa.asset_id = sn.asset_id
JOIN asset da ON da.asset_id = dn.asset_id
WHERE (sn.asset_id = %s OR dn.asset_id = %s)
  AND ge.status = %s
ORDER BY ge.confidence DESC NULLS LAST, ge.edge_id
"""
# 도메인 제외 없음(2026-07-23 전면 제거) — 의료 특수 트랙 미운용이라 도메인 무관 균일 취급.
# 이전엔 양끝 domain_label='medical' 배제(PHI·FR-603)를 SQL 단에 걸었으나 삭제. 의료 복귀(3년차) 시 재도입.


def fetch_active_relations_for_asset(
    conn: Connection[Any], *, asset_id: str, status: str = "active"
) -> list[dict[str, Any]]:
    """``asset_id`` 의 관계 이웃을 양방향으로 조회해 질의 자산 관점으로 정규화한다.

    Args:
        asset_id: 관점이 되는 자산. src/dst 어느 쪽에 있든 매칭한다.
        status: 엣지 상태 필터(기본 ``active``). ``proposed`` 등 다른 상태도 조회 가능.

    Returns:
        이웃 dict 리스트. 각 dict 키:
        ``asset_id``(상대 자산)·``kind_code``·``is_symmetric``·``direction``
        (``undirected``|``outbound``|``inbound``)·``confidence``·``status``·``topic``·
        ``reason``·``edge_id``·``file_name``(상대 자산 fs_path basename·FR-102)·
        ``modality``(상대 자산 모달리티·FR-102). 정렬은 confidence desc(동점 edge_id asc)로 결정적.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_FETCH_RELATIONS_SQL, (asset_id, asset_id, status))
        rows = cur.fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        is_symmetric = bool(r["is_symmetric"])
        src_asset = str(r["src_asset"])
        # 질의 자산 관점 정규화: 이웃은 항상 반대편, 방향은 대칭이면 무방향
        # 비대칭이면 질의 자산이 src 인지 dst 인지로 outbound/inbound 결정.
        is_query_src = src_asset == str(asset_id)
        other_asset = str(r["dst_asset"]) if is_query_src else src_asset
        # FR-102: 이웃(반대편) 자산의 표시필드 — 질의 자산이 src 면 이웃=dst, 아니면 이웃=src.
        other_modality = r["dst_modality"] if is_query_src else r["src_modality"]
        other_fs_path = r["dst_fs_path"] if is_query_src else r["src_fs_path"]
        if is_symmetric:
            direction = "undirected"
        else:
            direction = "outbound" if is_query_src else "inbound"
        out.append(
            {
                "asset_id": other_asset,
                "kind_code": r["kind_code"],
                "is_symmetric": is_symmetric,
                "direction": direction,
                "confidence": r["confidence"],
                "status": r["status"],
                # 065 FR-405: topic 은 이 관계(쌍)의 **맥락 라벨**이며 자산 주제가 아니다 — 관계 검토
                #   UI 표시용으로만 존치. 자산 주제는 asset_topic 정본이 결정한다(엣지 topic 소비 중단).
                "topic": r["topic"],
                "reason": r["reason"],
                "edge_id": str(r["edge_id"]),
                # FR-102(057): 이웃 표시필드 하향(하위호환 필드 추가·N+1 재조회 제거).
                "file_name": os.path.basename(other_fs_path or ""),
                "modality": other_modality,
            }
        )
    return out
