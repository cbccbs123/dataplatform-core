"""검색 KPI용 **결정적 합성 코퍼스** 생성기(T021).

실 임베딩 모델 없이, 쿼리 방향과 배경을 1536D 공간에 합성한다:
- 쿼리 q(0..n_queries-1)의 relevant 자산: 임베딩이 e_q(인덱스 q) 근처 → 코사인≈1.
- 배경 자산: 고인덱스(500+) 방향 → 어느 쿼리와도 코사인≈0.
- FTS: relevant 만 ``질의{q}토큰`` 을 포함 → bm25 도 분리.

검색은 ``_run_hybrid_search(query_vector=e_q)`` 로 벡터를 직접 주입해 모델 없이 측정한다.
시드 PRNG 로 100% 재현 가능(헌법 3조). 정리는 ``fs_path`` 마커로 일괄 삭제.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

from psycopg.rows import dict_row

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.database.ids import uuid7

_DIM = FIX_EMBEDDING_DIMENSION
CORPUS_FS_PREFIX = "/kpi_corpus/"  # 정리용 마커(이 prefix 행만 삭제)


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _vec(strong_idx: int, noise_idx: int, noise: float) -> list[float]:
    """strong_idx 에 1.0, noise_idx 에 작은 값 → 정규화(자산별로 약간씩 구별)."""
    v = [0.0] * _DIM
    v[strong_idx] = 1.0
    v[noise_idx] = noise
    return _normalize(v)


def clear_corpus(db: Any) -> int:
    """이전에 적재한 KPI 코퍼스(fs_path 마커)를 모두 삭제. 반환: 삭제 행수."""
    with db.transaction() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM asset WHERE fs_path LIKE %s", (CORPUS_FS_PREFIX + "%",))
        return cur.rowcount


def build_corpus(
    db: Any,
    *,
    n_background: int = 2000,
    n_queries: int = 10,
    relevant_per_query: int = 5,
    seed: int = 20260601,
) -> list[dict[str, Any]]:
    """결정적 합성 코퍼스를 적재하고 골든 질의 목록을 반환한다.

    반환 각 항목: ``{"query_vec": [..], "query_text": str, "relevant_ids": set[str]}``.
    총 자산 수 = n_queries*relevant_per_query + n_background.
    """
    rng = random.Random(seed)
    assets: list[tuple] = []
    metas: list[tuple] = []
    embs: list[tuple] = []
    goldens: list[dict[str, Any]] = []

    def _add(strong_idx: int, fts: str, summary: str) -> str:
        aid = uuid7()
        # 노이즈 인덱스는 1536D 안쪽이면서 쿼리(0~9)·배경(500~899) 방향과 겹치지 않게 1000~1499.
        noise_idx = 1000 + rng.randrange(500)
        vec = _vec(strong_idx, noise_idx, 0.05)
        assets.append((aid, "txt", f"{CORPUS_FS_PREFIX}{aid}.txt", aid.hex))
        metas.append((aid, json.dumps({"summary": summary}, ensure_ascii=False), fts))
        embs.append((aid, "st", 0, vec))
        return str(aid)

    # 쿼리별 relevant(쿼리 방향 e_q 근처 + 질의 토큰 FTS)
    for q in range(n_queries):
        rel_ids: set[str] = set()
        for _ in range(relevant_per_query):
            sid = _add(q, f"질의{q}토큰 공통문서 자료", f"질의{q} 관련 문서")
            rel_ids.add(sid)
        qvec = [0.0] * _DIM
        qvec[q] = 1.0
        goldens.append({"query_vec": qvec, "query_text": f"질의{q}토큰", "relevant_ids": rel_ids})

    # 배경(고인덱스 방향 → 쿼리와 직교, 질의 토큰 없음)
    for b in range(n_background):
        _add(500 + (b % 400), f"배경{b % 50}어휘 공통문서 자료", "")

    # 벌크 적재(배치 executemany, 단일 트랜잭션)
    with db.transaction() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO asset (asset_id, modality, fs_path, file_hash, domain_label, status) "
                "VALUES (%s, %s, %s, %s, 'general', 'registered')",
                assets,
            )
            cur.executemany(
                "INSERT INTO asset_metadata (asset_id, core_meta, ext_meta, tags, search_vector) "
                "VALUES (%s, '{}'::jsonb, %s::jsonb, '{}', to_tsvector('simple', coalesce(%s,'')))",
                metas,
            )
            cur.executemany(
                f"INSERT INTO asset_embedding (asset_id, channel, chunk_index, embedding, model_name) "
                f"VALUES (%s, %s, %s, %s::vector({_DIM}), 'kpi')",
                embs,
            )
    return goldens


def count_corpus(db: Any) -> int:
    with db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT count(*) AS n FROM asset WHERE fs_path LIKE %s", (CORPUS_FS_PREFIX + "%",))
        return int(cur.fetchone()["n"])
