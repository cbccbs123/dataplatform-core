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

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

# 양 끝점 asset 역조인(node_kind='asset')으로 src/dst asset_id 를 함께 끌어온 뒤,
# 파이썬에서 질의 자산 관점으로 방향·이웃을 정규화한다.
# ORDER BY: confidence DESC NULLS LAST 동점 시 순서가 불안정하므로 edge_id 2차 키로
# 결정성(헌법 3조)을 보장한다(ADR 원안 대비 강화 — plan R-3).
_FETCH_RELATIONS_SQL = """
SELECT ge.edge_id, rk.kind_code, rk.is_symmetric,
       ge.confidence, ge.reason, ge.topic, ge.status,
       sn.asset_id AS src_asset, dn.asset_id AS dst_asset
FROM graph_edge ge
JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
JOIN node sn ON sn.node_id = ge.src_node AND sn.node_kind = 'asset'
JOIN node dn ON dn.node_id = ge.dst_node AND dn.node_kind = 'asset'
WHERE (sn.asset_id = %s OR dn.asset_id = %s)
  AND ge.status = %s
ORDER BY ge.confidence DESC NULLS LAST, ge.edge_id
"""


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
        ``reason``·``edge_id``. 정렬은 confidence desc(동점 edge_id asc)로 결정적.
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
                "topic": r["topic"],
                "reason": r["reason"],
                "edge_id": str(r["edge_id"]),
            }
        )
    return out
