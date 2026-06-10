"""PostgreSQL(`asset_*`) → OpenSearch 데이터 동기화 (검색 엔진 도입 — spec 020).

PG 는 **읽기 전용**(SELECT 만), OpenSearch 에만 색인을 쓴다(CQRS — 원본 DB 무수정, 헌법 6조).

설계
    - **자산 1건 = OpenSearch 문서 1개**. 임베딩은 활성 채널 청크들의 **평균 풀링**(`avg(embedding)`)
      한 벡터를 `knn_vector` 로 색인한다 — 019 측정에서 평균 집계가 MAX 보다 검색 품질이 좋았고,
      자산당 단일 벡터라 색인·질의가 단순하다(청크별 색인은 후속 선택지).
    - **하이브리드 한 인덱스**: 텍스트(summary·keywords·labels·file_name)는 한국어 형태소 분석기
      `nori` 로 BM25, 임베딩은 `knn_vector`(코사인). 메타(modality·domain_label·status·channel)는
      keyword 필터.

이 모듈의 **순수 함수**(`build_index_body`·`asset_to_doc`·`parse_vector`)는 DB·OS·opensearch-py
없이 결정적으로 동작하며 단위 게이트에서 항상 검증된다. **IO 함수**(get_client·ensure_index·
색인 실행)는 후속 그룹(G2)에서 추가하며, opensearch-py·psycopg 의존은 모듈 상단이 아니라
**해당 함수 내부에서 지연 import** 한다 — 플래그 off(미도입) 환경의 순수성을 보존하기 위함이다.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION

# ── 읽기전용 SELECT (FR-004, 헌법 6조) — 원본 PG 무수정 ──
# registered 자산 + 메타(LEFT JOIN) + 활성 채널 청크 **평균 임베딩**(avg, 자산당 1행)을 한 행으로 모은다.
# avg(embedding) 은 pgvector 집계(>=0.5.0). 임베딩 없는 자산은 INNER JOIN 으로 자연 제외(→ 색인 대상 아님).
_ASSET_SELECT = """
SELECT a.asset_id, a.modality, a.domain_label, a.status, a.fs_path,
       am.ext_meta, e.emb AS emb, e.n AS chunk_count
FROM asset a
LEFT JOIN asset_metadata am ON am.asset_id = a.asset_id
JOIN (
    SELECT asset_id, avg(embedding) AS emb, count(*) AS n
    FROM asset_embedding
    WHERE channel = %s AND embedding IS NOT NULL
    GROUP BY asset_id
) e ON e.asset_id = a.asset_id
"""
# 전체 재동기화(복구 도구) — registered 자산 전부, asset_id 정렬로 결정적 순서(FR-005).
_SYNC_SQL = _ASSET_SELECT + "WHERE a.status = 'registered'\nORDER BY a.asset_id\n"
# 단건 색인(증분 훅) — 그 자산 1건만. 전체 재동기화(_SYNC_SQL)와 **대칭**으로 status='registered' 만
# 색인한다(비-registered → 행 없음 → index_asset no-op). 두 경로의 게이트를 맞춰, deferred/failed/medical
# 이 증분 경로로만 새던 비대칭(번들 게이트 우회와 같은 결의 누출)을 SQL 단에서 차단한다.
# 파라미터 순서 (channel, asset_id): 서브쿼리 channel 이 먼저.
_ASSET_ONE_SQL = _ASSET_SELECT + "WHERE a.asset_id = %s AND a.status = 'registered'\n"


def parse_vector(value: Any) -> list[float]:
    """pgvector 반환값(리스트 또는 ``'[0.1,0.2,...]'`` 문자열)을 float 리스트로 정규화(순수)."""
    if isinstance(value, list | tuple):
        return [float(x) for x in value]
    s = str(value).strip()
    inner = s[1:-1] if s.startswith("[") and s.endswith("]") else s
    return [float(x) for x in inner.split(",") if x.strip()]


def _basename(uri: str) -> str:
    """경로/URI 에서 파일명만 추출한다(결정적·순수). 쿼리·프래그먼트는 제거."""
    if not uri:
        return ""
    tail = uri.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail.split("?", 1)[0].split("#", 1)[0] or tail


def build_index_body(*, dim: int = FIX_EMBEDDING_DIMENSION) -> dict[str, Any]:
    """자산 인덱스 settings+mappings(순수). nori 한국어 분석기 + knn_vector(코사인).

    ``index.knn=true`` 로 kNN 검색을 켜고, 텍스트 필드는 ``analyzer='nori'``(analysis-nori 플러그인
    내장 분석기)로 한국어 형태소 BM25 를 쓴다. 임베딩은 lucene HNSW + cosinesimil. 차원은 단일
    출처 ``FIX_EMBEDDING_DIMENSION``(1536D, 헌법 6조)을 기본값으로 따른다.
    """
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "asset_id": {"type": "keyword"},
                "modality": {"type": "keyword"},
                "domain_label": {"type": "keyword"},
                "status": {"type": "keyword"},
                "channel": {"type": "keyword"},
                "file_name": {
                    "type": "text",
                    "analyzer": "nori",
                    "fields": {"raw": {"type": "keyword"}},
                },
                "fs_uri": {"type": "keyword"},
                "summary": {"type": "text", "analyzer": "nori"},
                "keywords": {"type": "text", "analyzer": "nori"},
                "labels": {"type": "keyword"},
                "search_text": {"type": "text", "analyzer": "nori"},
                "chunk_count": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
            }
        },
    }


def asset_to_doc(row: dict[str, Any], channel: str) -> dict[str, Any]:
    """PG 행(asset+metadata+평균임베딩) → OpenSearch 문서(순수·결정적).

    ``search_text`` 는 summary·file_name·keywords·labels 를 합쳐 BM25 대상으로 둔다.
    ext_meta 가 None/비-리스트(스키마 위반)여도 빈 값으로 안전 처리한다.
    """
    ext = row.get("ext_meta") or {}
    file_name = _basename(str(row.get("fs_path") or ""))
    summary = str(ext.get("summary") or "")
    keywords = ext.get("keywords") if isinstance(ext.get("keywords"), list) else []
    labels = ext.get("labels") if isinstance(ext.get("labels"), list) else []

    parts = [summary, file_name]
    parts += [str(k) for k in keywords]
    parts += [str(label) for label in labels]
    search_text = " ".join(p for p in parts if p)

    doc = {
        "asset_id": str(row["asset_id"]),
        "modality": row.get("modality"),
        "domain_label": row.get("domain_label"),
        "status": row.get("status"),
        "channel": channel,
        "file_name": file_name,
        "fs_uri": str(row.get("fs_path") or ""),
        "summary": summary,
        "keywords": [str(k) for k in keywords],
        "labels": [str(label) for label in labels],
        "search_text": search_text,
        "chunk_count": int(row.get("chunk_count") or 0),
    }
    # 영벡터(퇴화 임베딩 — 빈 STT 등)는 cosinesimil knn 이 거부하므로 embedding 필드를 **생략**한다.
    # 해당 자산은 텍스트(BM25)로만 검색되고 벡터 검색 대상에서만 빠진다(색인 실패 대신 우아한 처리).
    vec = parse_vector(row["emb"])
    if any(x != 0.0 for x in vec):
        doc["embedding"] = vec
    return doc


# ──────────────────────────────────────────────────────────────────────────
# IO 함수 (G2) — opensearch-py·psycopg 는 모듈 상단이 아니라 **함수 내부에서 지연 import**.
# 플래그 off(미도입) 환경의 모듈 순수성(상단 import 없음)을 보존하기 위함이다 — 위 순수 함수만
# 쓰는 단위 게이트는 opensearch-py·psycopg 미설치여도 import 가능해야 한다.
# 실제 OS·DB 동작 검증은 G5(실OS·실DB e2e). 여기 IO 는 가짜 클라이언트/conn 으로 액션 조립을 단위 검증.
# ──────────────────────────────────────────────────────────────────────────


def get_client(url: str | None = None) -> Any:
    """현재 설정(`OPENSEARCH_URL`, 미지정 시 기본 http://localhost:9200)의 OpenSearch 클라이언트.

    DEV 무인증(http) 기준. ``url`` 미지정 시 020 에서 추가된 정식 선택 필드 ``settings.opensearch_url``
    (기본 http://localhost:9200)을 참조한다.
    """
    from opensearchpy import OpenSearch

    from src.config.settings import get_current_settings

    if url is None:
        url = get_current_settings().opensearch_url
    return OpenSearch(
        hosts=[url],
        http_compress=True,
        use_ssl=url.startswith("https"),
        verify_certs=False,
        ssl_show_warn=False,
        timeout=60,
    )


def ensure_index(
    client: Any, index: str, *, recreate: bool = False, dim: int = FIX_EMBEDDING_DIMENSION
) -> str:
    """인덱스가 없으면 생성한다. ``recreate=True`` 면 **명시적으로** 삭제 후 재생성(파괴적·옵트인).

    반환: ``'created'`` | ``'recreated'`` | ``'exists'``. 매핑은 단일 출처 ``build_index_body``.
    """
    exists = client.indices.exists(index=index)
    if exists and recreate:
        client.indices.delete(index=index)
        client.indices.create(index=index, body=build_index_body(dim=dim))
        return "recreated"
    if not exists:
        client.indices.create(index=index, body=build_index_body(dim=dim))
        return "created"
    return "exists"


def _fetch_one(conn: Any, sql: str, params: tuple) -> dict[str, Any] | None:
    """단건 행을 dict 로 조회(읽기전용). psycopg 는 지연 import."""
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def check_pgvector_version(conn: Any, *, minimum: tuple[int, int] = (0, 5)) -> str:
    """pgvector 확장 버전이 최소 요구(기본 0.5)를 만족하는지 선검사한다(읽기전용 1쿼리, FR-004).

    동기화 SELECT 가 자산당 청크 임베딩을 ``avg(embedding)`` 으로 평균 풀링하는데, vector 타입
    집계(avg/sum)는 pgvector **0.5.0** 에서 추가됐다(그 이전엔 집계 함수 자체가 없다). 미설치/구버전
    환경에서 동기화가 런타임에 모호한 SQL 오류로 깨지는 대신, 복구 도구(run_opensearch_resync)
    시작 시 한 번 선검사해 원인이 분명한 오류로 막는다. 반환: 확인된 확장 버전 문자열.
    미설치·구버전이면 ``RuntimeError``.
    """
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    if not row or not row.get("extversion"):
        raise RuntimeError(
            "pgvector 확장이 설치돼 있지 않다 — OpenSearch 동기화는 avg(embedding) 집계"
            "(pgvector>=0.5)를 요구한다. 'CREATE EXTENSION vector' 후 재시도."
        )
    version = str(row["extversion"])
    # 앞 두 컴포넌트(major.minor)를 **위치 보존**해 파싱한다. 'v' 접두는 허용하되, 비숫자 컴포넌트로
    # minor 가 major 자리로 밀려 0.4 가 4.x 로 오인·통과되는 일을 막는다. semver 로 못 읽으면 (0,0)
    # 으로 두어 보수적 차단(모호한 통과보다 명확한 거부). pgvector 는 깨끗한 semver 만 내지만 견고성용.
    m = re.match(r"\s*v?(\d+)\.(\d+)", version)
    parts = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    if parts < minimum:
        need = ".".join(str(n) for n in minimum)
        raise RuntimeError(
            f"pgvector {version} < {need} — avg(embedding) 집계 미지원. 확장 업그레이드 필요."
        )
    return version


def iter_asset_docs(conn: Any, channel: str) -> Iterator[dict[str, Any]]:
    """PG 에서 registered 자산을 읽어 OpenSearch 문서를 yield(읽기 전용)."""
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_SYNC_SQL, (channel,))
        for row in cur:
            yield asset_to_doc(row, channel)


def index_asset(
    client: Any, conn: Any, asset_id: str, *, index: str, channel: str
) -> dict[str, Any] | None:
    """자산 1건을 OpenSearch 에 색인한다(증분 훅의 정상 경로 — PG 읽기 전용 → OS 쓰기).

    그 자산의 (메타 + 활성 채널 평균 임베딩) 1행을 조회해 ``asset_to_doc`` 로 문서를 만들고
    ``client.index(_id=asset_id)`` 로 **upsert**(재실행 멱등) 한다. 자산/임베딩이 없으면(INNER
    JOIN 제외) 색인하지 않고 ``None`` 을 반환한다(no-op). 반환: 색인한 문서 또는 ``None``.
    """
    row = _fetch_one(conn, _ASSET_ONE_SQL, (channel, asset_id))
    if row is None:
        return None
    doc = asset_to_doc(row, channel)
    client.index(index=index, id=str(asset_id), body=doc)
    return doc


def _bulk_actions(index: str, docs: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """문서를 bulk upsert 액션으로 감싼다(``_id=asset_id`` → 재실행 시 덮어쓰기·중복 없음)."""
    for doc in docs:
        yield {"_index": index, "_id": doc["asset_id"], "_source": doc}


def sync_all(
    client: Any,
    conn: Any,
    *,
    index: str,
    channel: str,
    recreate: bool = False,
    dim: int = FIX_EMBEDDING_DIMENSION,
    bulk_fn: Callable[..., tuple[int, list]] | None = None,
) -> tuple[str, int, list[Any]]:
    """registered 자산 전체를 OpenSearch 로 재동기화한다(복구 도구 — PG 읽기 전용 → OS 쓰기).

    ``_id=asset_id`` upsert 라 재실행은 멱등(중복 없음). 기본은 비파괴(없으면 생성·있으면 upsert);
    스키마 변경 시에만 ``recreate=True``. ``bulk_fn`` 은 색인 seam(기본 opensearch-py ``helpers.bulk``)
    으로, 단위 테스트가 가짜를 주입해 OS 없이 액션을 검증한다. 반환: ``(인덱스상태, 색인 건수, 오류 목록)``.
    """
    if bulk_fn is None:
        from opensearchpy import helpers

        bulk_fn = helpers.bulk

    status = ensure_index(client, index, recreate=recreate, dim=dim)
    actions = _bulk_actions(index, iter_asset_docs(conn, channel))
    ok, errors = bulk_fn(client, actions, stats_only=False, raise_on_error=False)
    client.indices.refresh(index=index)
    return status, ok, list(errors)


def resolve_channel(channel: str | None) -> str:
    """미지정이면 운영 활성 임베딩 채널(018)로 해소한다(적재·검색과 같은 채널을 색인)."""
    from src.config.settings import active_embed_channel

    return channel if channel is not None else active_embed_channel()
