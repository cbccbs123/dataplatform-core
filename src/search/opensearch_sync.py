"""PostgreSQL(`asset_*`) → OpenSearch 데이터 동기화 (검색 엔진 도입 — spec 020).

PG 는 **읽기 전용**(SELECT 만), OpenSearch 에만 색인을 쓴다(CQRS — 원본 DB 무수정, 헌법 6조).

설계
    - **자산 1건 = OpenSearch 문서 1개**. 임베딩은 활성 채널 청크들의 **평균 풀링**(`avg(embedding)`)
      한 벡터를 `knn_vector` 로 색인한다 — 019 측정에서 평균 집계가 MAX 보다 검색 품질이 좋았고,
      자산당 단일 벡터라 색인·질의가 단순하다(청크별 색인은 후속 선택지).
    - **하이브리드 한 인덱스**: 텍스트(summary·keywords·labels·file_name)는 한국어 형태소 분석기
      `nori` 로 BM25, 임베딩은 `knn_vector`(코사인). 메타(modality·domain_label·filter_kw 등)는
      keyword/date 필터. ``status``·``channel``·``chunk_count`` 는 색인하지 않는다(047 — 동기화 SQL·
      단일 active channel 전제).

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
from src.search.filter_index_fields import build_filter_index_fields

# ── 읽기전용 SELECT (FR-004, 헌법 6조) — 원본 PG 무수정 ──
# registered 자산 + 메타(LEFT JOIN) + 활성 채널 청크 **평균 임베딩**(avg, 자산당 1행)을 한 행으로 모은다.
# avg(embedding) 은 pgvector 집계(>=0.5.0). 임베딩 없는 자산은 INNER JOIN 으로 자연 제외(→ 색인 대상 아님).
_ASSET_SELECT = """
SELECT a.asset_id, a.modality, a.domain_label, a.fs_path, a.created_at,
       am.ext_meta, e.emb AS emb
FROM asset a
LEFT JOIN asset_metadata am ON am.asset_id = a.asset_id
JOIN (
    SELECT asset_id, avg(embedding) AS emb
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


# 파일명 정제(spec 026 FR-003②) — ID스러움 판정 상수.
# 순수 영숫자(하이픈 포함) 토큰만 ID 후보로 본다. 한글·기타 비-ASCII 가 섞이면 자연어로 간주해 보존.
_ALNUM_TOKEN_RE = re.compile(r"^[A-Za-z0-9-]+$")
_VOWELS = frozenset("aeiouAEIOU")
_ID_MIN_LEN = 8          # 길이≥8 (짧은 영숫자 'Qi2'·'xyz' 는 보수적으로 보존)
_ID_VOWEL_RATIO = 0.25   # 모음 비율<25% 면 ID스러움(영문 자연어는 모음이 더 많다)


def _looks_like_natural_word(token: str) -> bool:
    """표기 규칙성으로 '일반 영단어/고유명사'와 무작위 ID 를 가른다(사전 없이·순수).

    사전을 두지 않고, **숫자 없음 + 규칙적 대소문자**(전부 소문자/전부 대문자/첫글자만 대문자)면
    자연어로 본다 — 'Maintenance'·'SAMSUNG'·'galaxy' 는 보존되고, 숫자 혼입·불규칙 대소문자 교차
    ('HAi1OZD1OMM' 등)는 ID 로 분류된다. 하이픈 포함은 토큰성이 약해 자연어로 보지 않는다.
    """
    if any(c.isdigit() for c in token) or "-" in token:
        return False
    if token.islower() or token.isupper():
        return True
    return token[0].isupper() and token[1:].islower()


def _is_id_like_token(token: str) -> bool:
    """토큰이 유튜브 ID 같은 '식별자성' 잡음인지 판정한다(보수적·결정적, FR-003②).

    제거 조건(AND): 순수 영숫자([A-Za-z0-9-]) · 길이≥8 · (모음 비율<25% **또는**
    (대문자·소문자·숫자 중 2종 이상 혼합 and 사전식 단어 아님)). 한글 등 비-ASCII 토큰은
    첫 정규식에서 탈락해 **항상 보존**된다(외래어 명사 = 검색 신호이지 잡음이 아님).
    """
    if not _ALNUM_TOKEN_RE.match(token) or len(token) < _ID_MIN_LEN:
        return False
    vowel_ratio = sum(1 for c in token if c in _VOWELS) / len(token)
    if vowel_ratio < _ID_VOWEL_RATIO:
        # 모음 희소여도 표기가 규칙적이고 **모음(또는 y)이 하나라도 있는** 자연어('strength'·
        # 'blacksmith' 등 자음 과다 영단어)는 보존한다(리뷰 후속). 모음·y 가 전무한 토큰
        # ('QWXZBKLMN')은 영단어일 수 없으므로 표기가 규칙적이어도 ID 로 본다.
        has_vowelish = any(c in _VOWELS or c in "yY" for c in token)
        return not (_looks_like_natural_word(token) and has_vowelish)
    kinds = sum(
        (
            any(c.isupper() for c in token),
            any(c.islower() for c in token),
            any(c.isdigit() for c in token),
        )
    )
    return kinds >= 2 and not _looks_like_natural_word(token)


def clean_file_name(name: str, *, noise_patterns: Iterable[str] = ()) -> str:
    """파일명에서 ID스러운(유튜브 ID 등) 잡음 토큰을 보수적으로 제거한다(순수·결정적, FR-003②).

    절차: 확장자 분리(stem 만 — 확장자는 검색 신호가 아님) → ``_``/공백으로 토큰화 →
    ① 설정 잡음 패턴(``noise_patterns`` regex)에 매칭되는 토큰 제거(수집원별 규약 — 코드 수정 없이
    새 명명 규약 대응) ② ``_is_id_like_token`` 잡음 토큰 제거 → 남은 토큰을 공백으로 결합.
    한글 토큰은 항상 보존되고, 모든 토큰이 잡음이면 빈 문자열을 돌려준다(파일명 신호 0·안전).

    ``vlm_text_for_embedding`` 의 파일명 처리와 결을 맞춰, 파일명 노이즈가
    BM25·임베딩을 오염시키지 않게 한다(F8 — 어떤 명명 규약의 파일이 와도 피해 반경을 제한).
    """
    if not name:
        return ""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    compiled = [re.compile(p) for p in noise_patterns]
    kept: list[str] = []
    for token in re.split(r"[_\s]+", stem):
        if not token:
            continue
        if any(rx.search(token) for rx in compiled):
            continue
        if _is_id_like_token(token):
            continue
        kept.append(token)
    return " ".join(kept)


# nori user_dictionary 외래어 고유명사 기본 목록(spec 026 FR-004). 내장 'nori' analyzer 는
# user_dictionary 를 받지 못하므로 커스텀 analyzer 의 사전 규칙으로 넣어 분해를 막는다(아이패드→아이/
# 패드 분해 시 '아이패드' 정확매칭 무력화·가짜매칭 발생). settings.opensearch_nori_user_words 가 운영
# 단일 출처이며, build_index_body 는 순수 함수라 인자 미지정 시 이 모듈 기본을 쓴다(IO 층이 settings
# 값을 주입). 둘의 동치는 test_settings 의 계약 테스트가 봉인한다.
_DEFAULT_NORI_USER_WORDS: tuple[str, ...] = (
    "아이패드",
    "아이폰",
    "스마트워치",
    "맥세이프",
    "에어팟",
    "갤럭시",
    "애플워치",
)


def build_index_body(
    *,
    dim: int = FIX_EMBEDDING_DIMENSION,
    nori_user_words: Iterable[str] | None = None,
) -> dict[str, Any]:
    """자산 인덱스 settings+mappings(순수). 커스텀 nori 한국어 분석기 + knn_vector(코사인).

    ``index.knn=true`` 로 kNN 검색을 켜고, 텍스트 필드는 ``analyzer='nori_user'`` 를 쓴다 — 이는
    ``nori_tokenizer`` + ``user_dictionary_rules``(외래어 고유명사 목록)로 만든 **커스텀** analyzer 다.
    내장 'nori' analyzer 는 user_dictionary 를 받지 못해(설정 불가) 외래어가 분해되므로, 사전을 받는
    커스텀 토크나이저를 반드시 정의한다(FR-004). ``nori_user_words`` 미지정 시 모듈 기본 목록을 쓴다.
    임베딩은 lucene HNSW + cosinesimil. 차원은 단일 출처 ``FIX_EMBEDDING_DIMENSION``(1536D, 헌법 6조).
    """
    words = list(nori_user_words) if nori_user_words is not None else list(_DEFAULT_NORI_USER_WORDS)
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            # 커스텀 분석기: nori_tokenizer 에 user_dictionary_rules 를 직접 실어 외래어 고유명사를
            # 한 토큰으로 보존한다(내장 'nori' 로는 불가). 텍스트 필드들이 이 analyzer 를 공유한다.
            "analysis": {
                "tokenizer": {
                    "nori_user_tokenizer": {
                        "type": "nori_tokenizer",
                        "user_dictionary_rules": words,
                    }
                },
                "analyzer": {
                    "nori_user": {
                        "type": "custom",
                        "tokenizer": "nori_user_tokenizer",
                    }
                },
            },
        },
        "mappings": {
            "properties": {
                "asset_id": {"type": "keyword"},
                "modality": {"type": "keyword"},
                "domain_label": {"type": "keyword"},
                "file_name": {
                    "type": "text",
                    "analyzer": "nori_user",
                    "fields": {"raw": {"type": "keyword"}},
                },
                "fs_uri": {"type": "keyword"},
                "summary": {"type": "text", "analyzer": "nori_user"},
                "keywords": {"type": "text", "analyzer": "nori_user"},
                "labels": {"type": "keyword"},
                "filter_kw": {
                    "properties": {
                        "file_ext": {"type": "keyword"},
                        "source_dataset": {"type": "keyword"},
                    }
                },
                "filter_date": {
                    "properties": {
                        "created_at": {"type": "date"},
                    }
                },
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


def _flatten_labels(raw: Any) -> list[str]:
    """labels 항목을 색인용 문자열 리스트로 평탄화한다(순수, spec 026 FR-002 — P0 교정).

    과거엔 ``str(label)`` 직렬화라 ``{'label':'텍스트','score':0.51}`` dict 가 통째로
    ``"{'label': '텍스트', 'score': 0.519}"`` 로 색인돼 'label'·'score'·숫자가 BM25 를 오염시키고
    labels 정확매칭을 무력화했다. dict 면 ``label`` 문자열만, str 은 그대로 — ``vlm_text_for_embedding``
    의 처리와 **동형**(임베딩 입력과 색인 입력의 labels 표현을 일치). 빈/공백·비대상 타입은 제외.
    """
    out: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            lab = item.get("label")
            if lab is not None and str(lab).strip():
                out.append(str(lab).strip())
        elif isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def asset_to_doc(
    row: dict[str, Any],
    channel: str,
    *,
    noise_patterns: Iterable[str] = (),
) -> dict[str, Any]:
    """PG 행(asset+metadata+평균임베딩) → OpenSearch 문서(순수·결정적).

    BM25 교차 필드 AND(047)는 ``summary``·``keywords`` 필드로 쿼리 시 ``cross_fields`` 처리 —
    합본 ``search_text`` 색인은 제거했다. ``file_name`` 은 별도 필드 + 낮은 boost(026 FR-003①).
    ``channel`` 인자는 resync SQL 파라미터와 call-site 호환용 — **문서 필드로는 저장하지 않는다**
    (단일 active channel 인덱스 전제). ``status``·``chunk_count`` 도 색인 제외(registered 만 sync).
    ``file_name`` 필드 자체는 ``clean_file_name`` 으로 ID스러운 잡음 토큰을 정제한 값이다.
    ``labels`` 는 ``_flatten_labels`` 로 dict→label 문자열만 추출한다(P0·FR-002). ext_meta 가 None/
    비-리스트(스키마 위반)여도 빈 값으로 안전 처리한다. ``noise_patterns`` 는 settings 정제 패턴(IO 층 주입).
    """
    _ = channel  # resync SQL·call-site 호환 — 문서 필드 아님(단일 active channel 인덱스).
    ext = row.get("ext_meta") or {}
    file_name = clean_file_name(
        _basename(str(row.get("fs_path") or "")), noise_patterns=noise_patterns
    )
    summary = str(ext.get("summary") or "")
    keywords = ext.get("keywords") if isinstance(ext.get("keywords"), list) else []
    labels = _flatten_labels(ext.get("labels"))

    doc = {
        "asset_id": str(row["asset_id"]),
        "modality": row.get("modality"),
        "domain_label": row.get("domain_label"),
        "file_name": file_name,
        "fs_uri": str(row.get("fs_path") or ""),
        "summary": summary,
        "keywords": [str(k) for k in keywords],
        "labels": labels,
    }
    # 영벡터(퇴화 임베딩 — 빈 STT 등)는 cosinesimil knn 이 거부하므로 embedding 필드를 **생략**한다.
    # 해당 자산은 텍스트(BM25)로만 검색되고 벡터 검색 대상에서만 빠진다(색인 실패 대신 우아한 처리).
    vec = parse_vector(row["emb"])
    if any(x != 0.0 for x in vec):
        doc["embedding"] = vec
    doc.update(
        build_filter_index_fields(
            fs_path=str(row.get("fs_path") or ""),
            created_at=row.get("created_at"),
        )
    )
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
    client: Any,
    index: str,
    *,
    recreate: bool = False,
    dim: int = FIX_EMBEDDING_DIMENSION,
    nori_user_words: Iterable[str] | None = None,
) -> str:
    """인덱스가 없으면 생성한다. ``recreate=True`` 면 **명시적으로** 삭제 후 재생성(파괴적·옵트인).

    반환: ``'created'`` | ``'recreated'`` | ``'exists'``. 매핑은 단일 출처 ``build_index_body``.
    ``nori_user_words`` 미지정 시 ``build_index_body`` 기본 외래어 목록을 쓴다(IO 층이 settings 주입).
    """
    body = build_index_body(dim=dim, nori_user_words=nori_user_words)
    exists = client.indices.exists(index=index)
    if exists and recreate:
        client.indices.delete(index=index)
        client.indices.create(index=index, body=body)
        return "recreated"
    if not exists:
        client.indices.create(index=index, body=body)
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


def iter_asset_docs(
    conn: Any, channel: str, *, noise_patterns: Iterable[str] = ()
) -> Iterator[dict[str, Any]]:
    """PG 에서 registered 자산을 읽어 OpenSearch 문서를 yield(읽기 전용).

    ``noise_patterns`` 는 파일명 정제용 settings 패턴(IO 층이 주입 — 순수 ``asset_to_doc`` 으로 전달).
    """
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_SYNC_SQL, (channel,))
        for row in cur:
            yield asset_to_doc(row, channel, noise_patterns=noise_patterns)


def index_asset(
    client: Any,
    conn: Any,
    asset_id: str,
    *,
    index: str,
    channel: str,
    noise_patterns: Iterable[str] = (),
) -> dict[str, Any] | None:
    """자산 1건을 OpenSearch 에 색인한다(증분 훅의 정상 경로 — PG 읽기 전용 → OS 쓰기).

    그 자산의 (메타 + 활성 채널 평균 임베딩) 1행을 조회해 ``asset_to_doc`` 로 문서를 만들고
    ``client.index(_id=asset_id)`` 로 **upsert**(재실행 멱등) 한다. 자산/임베딩이 없으면(INNER
    JOIN 제외) 색인하지 않고 ``None`` 을 반환한다(no-op). 반환: 색인한 문서 또는 ``None``.
    """
    row = _fetch_one(conn, _ASSET_ONE_SQL, (channel, asset_id))
    if row is None:
        return None
    doc = asset_to_doc(row, channel, noise_patterns=noise_patterns)
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
    nori_user_words: Iterable[str] | None = None,
    noise_patterns: Iterable[str] = (),
) -> tuple[str, int, list[Any]]:
    """registered 자산 전체를 OpenSearch 로 재동기화한다(복구 도구 — PG 읽기 전용 → OS 쓰기).

    ``_id=asset_id`` upsert 라 재실행은 멱등(중복 없음). 기본은 비파괴(없으면 생성·있으면 upsert);
    스키마 변경 시에만 ``recreate=True``. ``bulk_fn`` 은 색인 seam(기본 opensearch-py ``helpers.bulk``)
    으로, 단위 테스트가 가짜를 주입해 OS 없이 액션을 검증한다. 반환: ``(인덱스상태, 색인 건수, 오류 목록)``.
    ``nori_user_words``(인덱스 analyzer 사전)·``noise_patterns``(파일명 정제) 는 settings 단일 출처를
    IO 층(복구 도구)이 주입한다 — 미지정 시 순수 함수 기본값(026 기본 사전·빈 패턴).
    """
    if bulk_fn is None:
        from opensearchpy import helpers

        bulk_fn = helpers.bulk

    status = ensure_index(
        client, index, recreate=recreate, dim=dim, nori_user_words=nori_user_words
    )
    actions = _bulk_actions(
        index, iter_asset_docs(conn, channel, noise_patterns=noise_patterns)
    )
    ok, errors = bulk_fn(client, actions, stats_only=False, raise_on_error=False)
    client.indices.refresh(index=index)
    return status, ok, list(errors)


def resolve_channel(channel: str | None) -> str:
    """미지정이면 운영 활성 임베딩 채널(018)로 해소한다(적재·검색과 같은 채널을 색인)."""
    from src.config.settings import active_embed_channel

    return channel if channel is not None else active_embed_channel()
