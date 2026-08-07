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

from src.relations.approval_policy import choose_folded_edge, exposure_tier

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
  AND ge.status = ANY(%s)
ORDER BY ge.confidence DESC NULLS LAST, ge.edge_id
"""
# 도메인별 제외 조건은 없다 — 모든 도메인을 균일하게 노출한다.

# 노출 등급의 정렬 우선순위 — 강칸이 먼저다. 신뢰도만으로 정렬하면 **고신뢰 약칸이 저신뢰
# 강칸을 밀어내** 사용자가 "확실한 관계"라 믿은 것이 아래로 내려간다.
_TIER_RANK = {"strong": 0, "weak": 1}


def fetch_relations_for_asset(
    conn: Connection[Any],
    *,
    asset_id: str,
    statuses: list[str] | None = None,
    include_weak: bool = False,
    min_conf_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """``asset_id`` 의 관계 이웃을 **노출 등급과 함께** 조회한다(081 조각③ · 2단 노출).

    등급은 ``tier`` 필드로 실린다 — ``"strong"``(연관 자료 · 승인됨) · ``"weak"``(참고 자료 ·
    미승인). 표시 문구는 사용자 결정 2026-08-07 — 종전의 *"비슷한 주제"* 는 쓰지 않는다(자산
    자기주제 기반 주제 기능과 이름이 겹쳐 다른 두 기능이 같은 것으로 읽힌다 · `approval_policy`
    참조). 약칸이 필요한 이유: 자동승인 게이트가 ``same_domain`` 을 강등하면 관계 보유
    자산이 26% 줄어 화면이 빈다. 강등된 관계를 약칸으로 살려 **커버리지를 지키면서** 강칸의
    정밀도만 올린다. 실측으로 미승인 제안 중 진짜 관계가 31.6%(strong 187건 중 59건)라
    이 칸이 없으면 그만큼이 사용자에게서 사라진다(`docs/관계_품질_측정_20260728.md` §9.1).

    ``include_weak=False``(기본)면 **애초에 ``proposed`` 를 조회하지 않는다** — 읽어서 버리면
    쓸데없이 무겁다.

    Args:
        asset_id: 관점이 되는 자산. src/dst 어느 쪽에 있든 매칭한다.
        statuses: 조회할 엣지 상태 목록. 주면 ``include_weak`` 보다 **우선**한다
            (기존 함수의 ``status`` 인자를 그대로 흘려보내기 위한 통로).
        include_weak: ``True`` 면 ``proposed`` 도 함께 읽어 약칸으로 노출한다.
        min_conf_similarity: 약칸 노출 하한. **영속화 게이트와 같은 값을 써야 한다** —
            다르면 "행은 있는데 화면에 없는" 유령 구간이 생긴다(`approval_policy` 참조).

    Returns:
        이웃 dict 리스트. 기존 키(``asset_id``·``kind_code``·``is_symmetric``·``direction``·
        ``confidence``·``status``·``topic``·``reason``·``edge_id``·``file_name``·``modality``)에
        ``tier`` 가 **추가**된다(하위호환). 정렬은 **등급 → 신뢰도 내림차순 → edge_id** 로
        결정적이다.
    """
    wanted = statuses if statuses is not None else (
        ["active", "proposed"] if include_weak else ["active"])
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_FETCH_RELATIONS_SQL, (asset_id, asset_id, wanted))
        rows = cur.fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        # 등급을 **먼저** 정한다 — None 이면 이 행은 노출 대상이 아니므로 정규화 비용도 아낀다.
        tier = exposure_tier(str(r["status"] or ""), str(r["kind_code"] or ""),
                             r["confidence"], min_conf_similarity=min_conf_similarity)
        if tier is None:
            continue
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
                # 081 조각③ — 강칸/약칸 구분. 기존 키는 그대로 두고 **추가**만 한다(하위호환).
                "tier": tier,
            }
        )
    out = _fold_by_neighbor(out)
    # 등급 우선 정렬. SQL 은 신뢰도로만 정렬하므로 여기서 등급을 앞세운다 — 안정 정렬이라
    # 같은 등급 안에서는 SQL 이 정한 (신뢰도 desc, edge_id) 순서가 그대로 유지된다.
    out.sort(key=lambda e: _TIER_RANK.get(str(e["tier"]), len(_TIER_RANK)))
    return out


def _fold_by_neighbor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """이웃 자산당 엣지 하나만 남긴다 — **동시보유 접기**(081 조각⑤).

    규칙·실측 근거·재측정 방법은 `approval_policy` 상단 **"동시보유 접기"** 주석이 정본이다.
    여기서는 *그룹을 만들어 넘기고 결과를 조립*하기만 한다 — 판정을 여기 두면 소비처마다
    복제될 것이고, 그러면 화면마다 결과가 갈린다(이 모듈이 존재하는 이유와 같다).

    **왜 이웃 자산 단위인가**: 화면은 이웃 하나를 카드 하나로 그린다. DB 의 쌍
    ``(src_node, dst_node)`` 단위로 접으면 A→B 와 B→A 가 각각 남아 **한 이웃이 두 번**
    나온다(비대칭 kind 에서 실제로 발생 — v4 에 `derived_from` 양방향 2행이 있다).

    ``folded_kind_codes`` 는 **모든 행에** 실린다(대개 빈 리스트). 접힌 쪽에만 넣으면 소비처가
    ``.get()`` 유무로 분기하게 되고, 그 분기가 곧 "접힘을 모르는 코드"를 만든다.

    Args:
        rows: 등급이 매겨진 이웃 엣지 목록. ``asset_id`` 로 묶는다.

    Returns:
        이웃당 한 건으로 접힌 목록. **입력 순서를 보존**한다(첫 등장 순).
        각 항목에 ``folded_kind_codes``(접힌 종류·정렬됨)가 추가된다.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r["asset_id"]), []).append(r)

    folded_out: list[dict[str, Any]] = []
    for edges in groups.values():
        keep_idx, folded_kinds = choose_folded_edge(edges)
        kept = dict(edges[keep_idx])
        kept["folded_kind_codes"] = folded_kinds
        folded_out.append(kept)
    return folded_out


def fetch_active_relations_for_asset(
    conn: Connection[Any], *, asset_id: str, status: str = "active"
) -> list[dict[str, Any]]:
    """``asset_id`` 의 관계 이웃을 **한 상태만** 조회한다(하위호환 래퍼).

    081 이전부터 있던 계약이다 — 포탈 상세·다운로드 번들 등 기존 호출부가 이 이름과 반환 키에
    의존하므로 그대로 남긴다. 새 코드는 등급까지 받는 ``fetch_relations_for_asset`` 을 쓴다.

    Args:
        asset_id: 관점이 되는 자산. src/dst 어느 쪽에 있든 매칭한다.
        status: 엣지 상태 필터(기본 ``active``). ``proposed``·``rejected`` 등도 조회 가능.

    Returns:
        이웃 dict 리스트(``fetch_relations_for_asset`` 과 같은 키 + ``tier``).
        ``status`` 를 명시했으므로 노출 하한은 적용하지 않는다 — 호출자가 원한 상태를
        조용히 걸러내면 "왜 안 나오나"를 추적할 수 없다.
    """
    return fetch_relations_for_asset(conn, asset_id=asset_id, statuses=[status])
