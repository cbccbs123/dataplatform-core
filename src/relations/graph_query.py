"""그래프 **read seam** — 자산 하나의 관계 이웃을 양방향·정규화해 조회.

왜 이 통로가 필요한가
    대칭 kind(``relation_kind.is_symmetric=True``)는 ``graph_persist._canonical_pair`` 가
    ``(min(node_id), max(node_id))`` 캐논 순서 **단일 행**으로 저장한다. 그래서 쓰기는
    중복 없이 깔끔하지만, "자산 X의 이웃"을 순진하게 ``WHERE src_node = X`` 로만 찾으면
    **X가 큰 쪽(dst)으로 접힌 대칭 엣지를 통째로 누락**한다. 또 ``is_symmetric``·``kind_code``
    는 엣지 행에 없고 ``relation_kind`` 에만 있다.
    → 그래서 **양방향으로 매칭**하고(``src`` 든 ``dst`` 든), 관계 종류를 조인해 붙이고, 질의
      자산 관점으로 방향을 정규화해야 한다. 이 읽기 경로를 여기 하나로 고정해 두는 이유는
      검색·상세·묶음 같은 소비자들이 각자 단방향 쿼리를 다시 만들어 쓰지 못하게 하기 위해서다.

읽기 전용
    스키마·쓰기 경로 변경 0(헌법 6조). VIEW 미사용 — ``src/relations/`` 파이썬 쿼리 함수 관례.
"""
from __future__ import annotations

import os
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

# 엣지 양 끝을 node → asset 으로 두 번 조인한다. 앞의 조인은 asset_id 를 얻기 위한 것이고,
# 뒤의 조인은 화면에 보일 파일명·모달리티를 함께 가져오기 위한 것이다 — 이게 없으면 소비자가
# 이웃마다 자산을 다시 조회해야 한다.
# 정렬에 edge_id 를 2차 키로 둔 이유: 신뢰도가 같은 엣지들의 순서가 실행 계획에 따라 흔들리면
# 같은 질의가 매번 다른 순서를 낸다.
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
# 도메인별 제외 조건은 없다 — 모든 도메인을 균일하게 노출한다.


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
        ``reason``·``edge_id``·``file_name``(상대 자산 파일명)·``modality``(상대 자산 모달리티).
        정렬은 신뢰도 내림차순, 동점은 edge_id 오름차순으로 **결정적**이다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_FETCH_RELATIONS_SQL, (asset_id, asset_id, status))
        rows = cur.fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        is_symmetric = bool(r["is_symmetric"])
        src_asset = str(r["src_asset"])
        # 질의 자산 관점으로 뒤집는다: 이웃은 늘 **반대편**이고, 방향은 대칭 관계면 무방향,
        # 비대칭이면 질의 자산이 어느 쪽에 있느냐로 나가는·들어오는 방향이 갈린다.
        is_query_src = src_asset == str(asset_id)
        other_asset = str(r["dst_asset"]) if is_query_src else src_asset
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
                # topic 은 이 관계(쌍)의 **맥락 라벨**이며 자산 주제가 아니다 — 관계 검토
                #   UI 표시용으로만 존치. 자산 주제는 asset_topic 정본이 결정한다(엣지 topic 소비 중단).
                "topic": r["topic"],
                "reason": r["reason"],
                "edge_id": str(r["edge_id"]),
                # 이웃의 표시 정보를 함께 내려 준다 — 없으면 소비자가 이웃마다 자산을 다시 조회해야 한다.
                "file_name": os.path.basename(other_fs_path or ""),
                "modality": other_modality,
            }
        )
    return out
