"""PostgreSQL 자산 데이터를 OpenSearch 색인 문서로 옮긴다.

**흐름에서의 위치**: 적재가 끝난 자산을 검색 가능하게 만드는 마지막 단계다. PG 는 읽기만 하고
(원본은 건드리지 않는다) OpenSearch 에만 쓴다.

설계
    - **자산 1건 = OpenSearch 문서 1개**. 임베딩은 활성 채널 청크들의 **평균 풀링**(`avg(embedding)`)
      한 벡터를 `knn_vector` 로 색인한다 — 평균이 최댓값보다 검색 품질이 좋았고,
      자산당 단일 벡터라 색인·질의가 단순하다.
    - **하이브리드 한 인덱스**: 텍스트(summary·keywords·file_name)는 한국어 형태소 분석기 `nori` 로
      BM25(``labels`` 는 keyword 정확매칭 필드), 임베딩은 `knn_vector`(코사인). 메타(modality·
      domain_label·filter_kw 등)는 keyword/date 필터. ``status``·``channel``·``chunk_count`` 는 색인하지
      않는다(인덱스가 활성 채널 하나만 담는 전제).

**순수 함수**(`build_index_body`·`asset_to_doc`·`parse_vector`)는 DB·OpenSearch 없이 돌아가
단위 테스트로 전부 검증된다. **IO 함수**는 opensearch-py·psycopg 를 모듈 상단이 아니라 **함수
안에서 지연 import** 한다 — 그 라이브러리가 없는 환경에서도 순수 함수만 쓰는 코드가 이 모듈을
import 할 수 있어야 하기 때문이다.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.config.filename_util import basename_of
from src.config.search_constants import NORI_USER_WORDS_DEFAULT
from src.search.filter_index_fields import build_filter_index_fields

# ── 색인 대상 조회 SQL (읽기 전용 — PG 는 건드리지 않는다) ──
# 자산 하나를 한 행으로 만든다: 메타 + **청크 임베딩의 평균**(자산당 1벡터).
# 평균 집계는 pgvector 0.5 이상에서만 되므로 복구 도구가 시작 전에 버전을 확인한다.
# 임베딩이 없는 자산은 INNER JOIN 에서 자연히 빠진다 — 색인해도 벡터 검색이 안 되기 때문이다.
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
# 전체 재동기화용. asset_id 로 정렬해 순서를 고정한다(같은 데이터면 같은 순서로 색인).
_SYNC_SQL = _ASSET_SELECT + "WHERE a.status = 'registered'\nORDER BY a.asset_id\n"
# 단건 색인용. 전체 재동기화와 **같은 조건**(registered 만)을 쓰는 것이 중요하다 — 두 경로의
# 기준이 어긋나면 아직 준비되지 않은 자산이 증분 경로로만 색인돼 검색에 새어 나온다.
# 파라미터 순서는 (channel, asset_id) — 서브쿼리의 channel 이 먼저 바인딩된다.
_ASSET_ONE_SQL = _ASSET_SELECT + "WHERE a.asset_id = %s AND a.status = 'registered'\n"


def parse_vector(value: Any) -> list[float]:
    """pgvector 가 돌려준 값을 float 리스트로 정규화한다(순수).

    드라이버·쿼리에 따라 리스트로도, ``'[0.1,0.2,...]'`` 문자열로도 온다.

    Args:
        value: 리스트/튜플 또는 대괄호 문자열.

    Returns:
        float 리스트.

    Raises:
        ValueError: 숫자로 바꿀 수 없는 원소가 섞여 있을 때(형식이 깨진 값을 조용히 넘기지 않는다).
    """
    if isinstance(value, list | tuple):
        return [float(x) for x in value]
    s = str(value).strip()
    inner = s[1:-1] if s.startswith("[") and s.endswith("]") else s
    return [float(x) for x in inner.split(",") if x.strip()]


# 파일명에서 잡음(유튜브 ID 같은 것)을 걸러내기 위한 판정 상수.
# 영숫자 토큰만 의심하고, 한글 등 비-ASCII 가 섞이면 자연어로 보고 **항상 보존**한다 —
# 외래어 명사는 잡음이 아니라 검색 신호이기 때문이다.
_ALNUM_TOKEN_RE = re.compile(r"^[A-Za-z0-9-]+$")
_VOWELS = frozenset("aeiouAEIOU")
_ID_MIN_LEN = 8          # 길이≥8 (짧은 영숫자 'Qi2'·'xyz' 는 보수적으로 보존)
_ID_VOWEL_RATIO = 0.25   # 모음 비율<25% 면 ID스러움(영문 자연어는 모음이 더 많다)


def _looks_like_natural_word(token: str) -> bool:
    """표기 규칙성으로 '일반 영단어/고유명사'와 무작위 ID 를 가른다(사전 없이·순수).

    사전을 두지 않고, **숫자 없음 + 규칙적 대소문자**(전부 소문자/전부 대문자/첫글자만 대문자)면
    자연어로 본다 — 'Maintenance'·'SAMSUNG'·'galaxy' 는 보존되고, 숫자 혼입·불규칙 대소문자 교차
    ('HAi1OZD1OMM' 등)는 ID 로 분류된다. 하이픈 포함은 토큰성이 약해 자연어로 보지 않는다.

    Args:
        token: 판정할 토큰(빈 문자열은 넘기지 말 것 — 호출부가 걸러 준다).

    Returns:
        자연어로 보이면 True.
    """
    if any(c.isdigit() for c in token) or "-" in token:
        return False
    if token.islower() or token.isupper():
        return True
    return token[0].isupper() and token[1:].islower()


def _is_id_like_token(token: str) -> bool:
    """토큰이 유튜브 ID 같은 '식별자성' 잡음인지 판정한다(보수적·결정적).

    제거 조건(AND): 순수 영숫자([A-Za-z0-9-]) · 길이≥8 · (모음 비율<25% **또는**
    (대문자·소문자·숫자 중 2종 이상 혼합 and 사전식 단어 아님)). 한글 등 비-ASCII 토큰은
    첫 정규식에서 탈락해 **항상 보존**된다(외래어 명사 = 검색 신호이지 잡음이 아님).

    Args:
        token: 판정할 파일명 토큰.

    Returns:
        잡음 ID 로 보이면 True(호출부가 제거). **판정이 애매하면 False**(보존) 쪽으로 기운다 —
        검색 신호를 잃는 쪽보다 잡음이 조금 남는 쪽이 안전하기 때문이다.
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
    """파일명에서 ID스러운(유튜브 ID 등) 잡음 토큰을 보수적으로 제거한다(순수·결정적).

    절차: 확장자 분리(stem 만 — 확장자는 검색 신호가 아님) → ``_``/공백으로 토큰화 →
    ① 설정 잡음 패턴(``noise_patterns`` regex)에 매칭되는 토큰 제거(수집원별 규약 — 코드 수정 없이
    새 명명 규약 대응) ② ``_is_id_like_token`` 잡음 토큰 제거 → 남은 토큰을 공백으로 결합.
    한글 토큰은 항상 보존되고, 모든 토큰이 잡음이면 빈 문자열을 돌려준다(파일명 신호 0·안전).

    임베딩 입력을 만들 때의 파일명 처리와 규칙을 맞춰, 어떤 명명 규약의 파일이 들어와도
    파일명 노이즈가 BM25·임베딩을 오염시키지 않게 한다.

    Args:
        name: 원본 파일명(경로 아님). 빈 값이면 빈 문자열을 돌려준다.
        noise_patterns: 수집원별 잡음 정규식들. **코드를 고치지 않고 설정으로** 새 명명 규약에
            대응하려는 확장점이다. 여기 매칭되면 ID 판정 전에 먼저 버린다.

    Returns:
        공백으로 이어붙인 정제 파일명. 모든 토큰이 잡음이면 **빈 문자열**(파일명 신호를 0으로 둔다).
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


def build_index_body(
    *,
    dim: int = FIX_EMBEDDING_DIMENSION,
    nori_user_words: Iterable[str] | None = None,
) -> dict[str, Any]:
    """자산 인덱스 settings+mappings(순수). 커스텀 nori 한국어 분석기 + knn_vector(코사인).

    ``index.knn=true`` 로 kNN 검색을 켜고, 텍스트 필드는 ``analyzer='nori_user'`` 를 쓴다 — 이는
    ``nori_tokenizer`` + ``user_dictionary_rules``(외래어 고유명사 목록)로 만든 **커스텀** analyzer 다.
    내장 'nori' analyzer 는 user_dictionary 를 받지 못해(설정 불가) 외래어가 분해되므로, 사전을 받는
    커스텀 토크나이저를 반드시 정의한다. ``nori_user_words`` 미지정 시 설정과 공유하는 기본 목록을 쓴다.
    임베딩은 HNSW + 코사인. 차원은 단일 출처 상수를 따른다.

    Args:
        dim: 벡터 차원. 기본값이 정본이며 **바꾸면 기존 색인과 호환되지 않는다**(재색인 필요).
        nori_user_words: 분해를 막을 외래어·고유명사 목록. ``None`` 이면 설정 기본 목록을 쓴다.
            빈 리스트를 명시하면 사전 없이 만든다(테스트용).

    Returns:
        인덱스 생성에 그대로 넘길 ``{settings, mappings}`` dict(순수 데이터 — 생성은 호출부가 한다).
    """
    words = list(nori_user_words) if nori_user_words is not None else list(NORI_USER_WORDS_DEFAULT)
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
                # 065 자기주제 — 자산 자기주제 정본(fetch_asset_topic)을 색인한 값(관계-이웃 투영 은퇴).
                # 패싯·정확필터용 keyword(terms)만 색인한다. BM25 관련도 보강(topics_text)은
                # 스코프 철회(FR-504·SC-04 제거) — 랭킹 회귀 위험 대비 이득이 낮아 keyword 필터만 남긴다.
                "topics": {"type": "keyword"},
                "subtopics": {"type": "keyword"},
                # 059 FR-102 — (topic_ko, subtopic_ko) 짝을 "topic>subtopic" 한 문자열로 보존한
                # keyword. 평면 topics/subtopics 는 부모-자식 짝을 잃어 프론트 트리가 교차곱으로
                # 오배치되는 문제를 해소한다(표시·패싯 전용·랭킹 미반영). topics/subtopics 옆 동형 keyword.
                "topic_pairs": {"type": "keyword"},
                # 073 aboutness 개체 — 적재시 LLM 1회로 확정한 "이 자산이 무엇에 관한 것인가" 명사
                # 1~3개(ext_meta['about']). 검색 OR-증거 필터(about_or_filter)의 매칭 소스(랭킹 미반영).
                "about": {"type": "keyword"},
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
    """labels 항목을 색인용 문자열 리스트로 평탄화한다(순수).

    과거엔 ``str(label)`` 직렬화라 ``{'label':'텍스트','score':0.51}`` dict 가 통째로
    ``"{'label': '텍스트', 'score': 0.519}"`` 로 색인돼 'label'·'score'·숫자가 BM25 를 오염시키고
    labels 정확매칭을 무력화했다. dict 면 ``label`` 문자열만, str 은 그대로 — ``vlm_text_for_embedding``
    의 처리와 **동형**(임베딩 입력과 색인 입력의 labels 표현을 일치).

    Args:
        raw: ext_meta 의 labels 값. 리스트가 아니면 빈 결과로 안전 처리한다(스키마 위반 방어).

    Returns:
        라벨 문자열 리스트. 빈 문자열·공백뿐인 항목·알 수 없는 타입은 제외한다.
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


# 주제 필드 조립(056 FR-202) — asset_to_doc(전체문서)·update_asset_topics(부분문서) 단일 출처.
# 065: 입력은 자기주제 정본(fetch_asset_topic) [{topic_ko, subtopic_ko, ...}] 이며 여기서는
# 입력 순서를 보존한 채 dedup·빈값 스킵만 한다(재정렬 없음 → 순수·결정적, 헌법 3조).


def _dedup_in_order(values: Iterable[Any]) -> list[str]:
    """빈 값을 빼고 **첫 등장 순서를 보존한 채** 중복을 제거한다(keyword 필드용·결정적).

    정렬하지 않는 이유: 입력(자기주제 정본)의 순서가 곧 의미 순서이고, 재정렬하면 같은 자산의
    색인 결과가 입력 순서에 따라 달라 보이기 때문이다.

    Args:
        values: 임의의 값들(None·빈 문자열 섞여도 된다).

    Returns:
        문자열 리스트(중복·빈 값 제거, 순서 보존).
    """
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if not v:
            continue
        s = str(v)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _topic_pair(topic_ko: Any, subtopic_ko: Any) -> str:
    """(topic_ko, subtopic_ko) → ``"topic>subtopic"`` 짝 문자열(순수·결정적).

    subtopic 이 None/"" 이면 ``topic_ko`` 단독으로 돌려준다(짝 없는 주제도 트리 루트로 표시).
    구분자 ``>`` 는 정규화 라벨에 나타나지 않는 문자다(라벨은 한 어절·``·`` 만 허용) → 충돌 0·프론트
    파싱 계약(C2). **프론트 파싱 계약: 첫 ``>`` 로만 분할한다**(``split('>', 1)``) — topic 층은
    대주제는 닫힌 어휘라(``>`` 를 포함하지 않는다) 항상 첫 토큰으로 정확히 복원되고, 만에 하나 세부주제
    라벨에 ``>`` 가 섞여도 부모 오배치(교차곱)는 나지 않는다(subtopic 표기만 그대로 보존).
    Args:
        topic_ko: 대주제 라벨.
        subtopic_ko: 세부주제 라벨. 비면 대주제 단독 문자열이 된다.

    Returns:
        ``"대주제>세부주제"`` 또는 ``"대주제"``. **대주제가 비면 빈 문자열** — 호출부의
        ``_dedup_in_order`` 가 이를 스킵한다(평면 ``topics`` 처리와 동형).
    """
    tk = str(topic_ko) if topic_ko else ""
    if not tk:
        return ""
    sk = str(subtopic_ko) if subtopic_ko else ""
    return f"{tk}>{sk}" if sk else tk


def _topics_doc_fields(topics: list[dict[str, Any]]) -> dict[str, Any]:
    """주제 리스트 → OS 문서 주제 필드(순수·결정적).

    - ``topics``      = dedup 된 ``topic_ko`` (keyword·패싯/필터)
    - ``subtopics``   = dedup 된 ``subtopic_ko``(None/"" 스킵·keyword)
    - ``topic_pairs`` = dedup 된 ``"topic_ko>subtopic_ko"`` 짝(subtopic 없으면 topic 단독·keyword)

    ``topic_pairs`` 는 평면 ``topics``/``subtopics`` 가 잃어버리는 **부모-자식 짝**을
    한 문자열로 보존해, 프론트가 topic→subtopic 트리를 교차곱 오배치 없이 그리게 한다. 짝은 입력
    (자기주제 정본 조회 결과)에 이미 있으므로 그대로 나르며(생성 아님),
    ``_dedup_in_order`` 로 순서 보존·중복 제거만 한다. **평면 ``topics``/``subtopics`` 반환값·로직은 불변**(짝 필드는
    표시·패싯 전용·랭킹 미반영 — 검색 무회귀, C5).

    주제는 keyword terms 필터·패싯으로만 검색에 반영한다(BM25 관련도에는 넣지 않는다 — 랭킹
    회귀 위험 대비 이득이 낮아 철회했다).

    Args:
        topics: 자기주제 정본 리스트(``{topic_ko, subtopic_ko, ...}``).

    Returns:
        ``{topics, subtopics, topic_pairs}`` 3필드. 입력이 비면 각 값이 빈 리스트다.
    """
    return {
        "topics": _dedup_in_order(t.get("topic_ko") for t in topics),
        "subtopics": _dedup_in_order(t.get("subtopic_ko") for t in topics),
        "topic_pairs": _dedup_in_order(
            _topic_pair(t.get("topic_ko"), t.get("subtopic_ko")) for t in topics
        ),
    }


def asset_to_doc(
    row: dict[str, Any],
    channel: str,
    *,
    noise_patterns: Iterable[str] = (),
    topics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """PG 행(asset+metadata+평균임베딩) → OpenSearch 문서(순수·결정적).

    요약·키워드는 검색 시점에 한 필드처럼 함께 조회되므로 합본 필드를 따로 색인하지 않는다.
    ``file_name`` 은 별도 필드로 두고 가중치를 낮게 준다(파일명 노이즈가 랭킹을 뒤집지 않게).
    ``channel`` 인자는 resync SQL 파라미터와 call-site 호환용 — **문서 필드로는 저장하지 않는다**
    (단일 active channel 인덱스 전제). ``status``·``chunk_count`` 도 색인 제외(registered 만 sync).
    ``file_name`` 필드 자체는 ``clean_file_name`` 으로 ID스러운 잡음 토큰을 정제한 값이다.
    ``labels`` 는 ``_flatten_labels`` 로 dict→label 문자열만 추출한다. ext_meta 가 None/
    비-리스트(스키마 위반)여도 빈 값으로 안전 처리한다. ``noise_patterns`` 는 settings 정제 패턴(IO 층 주입).

    ``topics`` 는 자기주제 정본 조회 결과다. **주어지고 비어있지
    않으면** ``topics``/``subtopics``/``topic_pairs`` 3필드를 수록하고, ``None``/빈
    리스트면 이 필드들을 **넣지 않는다**(주제 미부여 자산·하위호환 — 기존 문서 형상 불변). 이 경로가 전체문서 색인마다
    자산 자기주제를 함께 실어, 재수집·재색인이 색인된 주제를 지우지 않게 한다.

    Args:
        row: PG 조회 행(asset + metadata + 평균 임베딩).
        channel: 임베딩 채널. **문서에는 저장하지 않는다** — 인덱스가 활성 채널 하나만 담는 전제라
            호출부 시그니처 호환용으로만 받는다.
        noise_patterns: 파일명 정제용 잡음 정규식(설정에서 주입).
        topics: 자기주제 정본 리스트. ``None``·빈 리스트면 주제 3필드를 **아예 넣지 않는다**.

    Returns:
        색인용 문서 dict. **0-노름 임베딩이면 ``embedding`` 필드를 생략**한다 — 코사인 kNN 이
        거부하는 값이라, 색인 자체를 실패시키는 대신 그 자산을 BM25 로만 찾히게 둔다.
    """
    _ = channel  # resync SQL·call-site 호환 — 문서 필드 아님(단일 active channel 인덱스).
    ext = row.get("ext_meta") or {}
    file_name = clean_file_name(
        basename_of(str(row.get("fs_path") or "")), noise_patterns=noise_patterns
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
        # 073: aboutness 개체(ext_meta['about']·적재시 확정). 미추출 자산은 빈 리스트 — 검색 필터의
        # amatch 만 비활성(kmatch·fail-safe 는 동작)이라 백필 전에도 안전.
        "about": [str(a) for a in (ext.get("about") or [])],
    }
    # 영벡터(퇴화 임베딩 — 빈 STT 등)는 cosinesimil knn 이 거부하므로 embedding 필드를 **생략**한다.
    # 해당 자산은 텍스트(BM25)로만 검색되고 벡터 검색 대상에서만 빠진다(색인 실패 대신 우아한 처리).
    vec = parse_vector(row["emb"])
    if any(x != 0.0 for x in vec):
        doc["embedding"] = vec
    # 자기주제 수록(065) — 비어있지 않을 때만 3필드(topics·subtopics·topic_pairs) 추가(None/[] → 생략·하위호환).
    if topics:
        doc.update(_topics_doc_fields(topics))
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

    개발 환경 무인증(http) 기준이다.

    Args:
        url: 접속 URL. ``None`` 이면 설정값(기본 ``http://localhost:9200``)을 쓴다.

    Returns:
        OpenSearch 클라이언트(타임아웃 60초·압축 on).
    """
    from opensearchpy import OpenSearch

    from src.config.settings import get_current_settings

    if url is None:
        url = get_current_settings().opensearch.url
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

    Args:
        client: OpenSearch 클라이언트.
        index: 인덱스 이름.
        recreate: **True 면 기존 인덱스를 지우고 다시 만든다** — 색인된 문서가 전부 사라지는
            파괴적 동작이라 명시적으로 켤 때만 수행한다.
        dim: 벡터 차원(기본 = 정본 1536D).
        nori_user_words: 분해 방지 외래어 목록. ``None`` 이면 기본 목록.

    Returns:
        ``'created'``(신규) · ``'recreated'``(삭제 후 재생성) · ``'exists'``(그대로 둠).
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
    """단건 행을 dict 로 조회한다(읽기 전용). psycopg 는 함수 안에서 지연 import 한다.

    Args:
        conn: DB 커넥션.
        sql: 실행할 SELECT.
        params: 바인딩 파라미터 튜플.

    Returns:
        행 dict. 결과가 없으면 ``None``.
    """
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def check_pgvector_version(conn: Any, *, minimum: tuple[int, int] = (0, 5)) -> str:
    """pgvector 확장 버전이 최소 요구(기본 0.5)를 만족하는지 선검사한다(읽기 전용 1쿼리).

    동기화 SELECT 가 자산당 청크 임베딩을 ``avg(embedding)`` 으로 평균 풀링하는데, vector 타입
    집계(avg/sum)는 pgvector **0.5.0** 에서 추가됐다(그 이전엔 집계 함수 자체가 없다). 미설치/구버전
    환경에서 동기화가 런타임에 모호한 SQL 오류로 깨지는 대신, 복구 도구(run_opensearch_resync)
    시작 시 한 번 선검사해 원인이 분명한 오류로 막는다.

    Args:
        conn: DB 커넥션(읽기만 한다).
        minimum: 요구 최소 버전 ``(major, minor)``.

    Returns:
        확인된 확장 버전 문자열.

    Raises:
        RuntimeError: 확장 미설치이거나 버전이 낮을 때. **버전 문자열을 못 읽으면 ``(0,0)`` 으로
            보고 차단한다** — 모호하게 통과시키느니 명확히 거부하는 쪽을 택했다.
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


def _default_topics_fn(conn: Any, asset_id: Any) -> list[dict[str, Any]]:
    """자산의 자기주제 정본을 읽는다(색인 경로가 공유하는 기본 seam).

    주제는 관계 이웃에서 빌려오는 게 아니라 **자산이 자기 내용에서 확정한 정본**을 읽는다.
    ``fetch_asset_topic`` 은 ``psycopg`` 를 모듈 상단에서
    당기므로 **호출 시 지연 import** 한다 — 플래그 off(미도입) 환경의 순수 함수 게이트가 이 무거운
    의존 없이 opensearch_sync 를 import 할 수 있게 하기 위함이다.

    Args:
        conn: DB 커넥션.
        asset_id: 조회할 자산.

    Returns:
        ``[{topic_ko, subtopic_ko, topic_en, subtopic_en, weight}]``. **빈 리스트면 주제 미부여**.
    """
    from src.topic.asset_topic_query import fetch_asset_topic

    return fetch_asset_topic(conn, asset_id=str(asset_id))


# 색인 경로에 topic 을 잇는 seam(065 T402). 기본은 fetch_asset_topic(자기주제 정본 읽기).
# 단위 테스트/특수 경로는 topics_fn 을 주입해 실 DB 없이 색인 문서 조립을 검증한다(bulk_fn·sync_fn 동형).
TopicsFn = Callable[[Any, Any], list[dict[str, Any]]]


def iter_asset_docs(
    conn: Any,
    channel: str,
    *,
    noise_patterns: Iterable[str] = (),
    topics_fn: TopicsFn = _default_topics_fn,
) -> Iterator[dict[str, Any]]:
    """PG 에서 registered 자산을 읽어 OpenSearch 문서를 yield(읽기 전용).

    ``noise_patterns`` 는 파일명 정제용 settings 패턴(IO 층이 주입 — 순수 ``asset_to_doc`` 으로 전달).
    ``topics_fn`` 은 자산별 자기주제를 읽어 문서에 함께 싣는다 — 전체 재색인이 색인된 주제를
    지우지 않게 하는 장치다.

    구현 주의(실 DB): 바깥 커서를 순회하며 자산마다 ``topics_fn`` 이 conn 에 **중첩 커서**로 topic 을
    조회한다. psycopg3 기본(client-side) 커서는 ``execute`` 시 결과를 클라이언트로 내려받아 버퍼링하므로,
    순회 중 같은 conn 에 다른 커서로 조회해도 안전하다(server-side named 커서가 아님).

    Args:
        conn: DB 커넥션(읽기 전용).
        channel: 평균 임베딩을 뽑을 채널.
        noise_patterns: 파일명 정제용 잡음 정규식.
        topics_fn: 자산별 주제를 읽는 함수 **주입 seam**. 테스트는 가짜 함수를 넣어 실 DB 없이
            문서 조립만 검증한다.

    Yields:
        색인용 문서 dict(registered 자산만).
    """
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_SYNC_SQL, (channel,))
        for row in cur:
            topics = topics_fn(conn, row["asset_id"])
            yield asset_to_doc(row, channel, noise_patterns=noise_patterns, topics=topics)


def index_asset(
    client: Any,
    conn: Any,
    asset_id: str,
    *,
    index: str,
    channel: str,
    noise_patterns: Iterable[str] = (),
    topics_fn: TopicsFn = _default_topics_fn,
) -> dict[str, Any] | None:
    """자산 1건을 OpenSearch 에 색인한다(증분 훅의 정상 경로 — PG 읽기 전용 → OS 쓰기).

    그 자산의 (메타 + 활성 채널 평균 임베딩) 1행을 조회해 ``asset_to_doc`` 로 문서를 만들고
    ``client.index(_id=asset_id)`` 로 **upsert**(재실행 멱등) 한다. 자산/임베딩이 없으면(INNER
    JOIN 제외) 색인하지 않고 ``None`` 을 반환한다(no-op). 반환: 색인한 문서 또는 ``None``.

    ``topics_fn``(065·기본 ``fetch_asset_topic``) 으로 그 자산의 **자기주제 정본**을 읽어
    전체문서에 함께 싣는다 — run_ingest 증분 훅이 재수집 자산을 이 경로로 재색인해도 앞서
    색인된 topics 를 지우지 않는다(C5·SC-03). ``_fetch_one`` 커서는 ``with`` 종료로 닫힌 뒤
    ``topics_fn`` 이 conn 에 조회하므로 커서 충돌이 없다.

    **OpenSearch 에 쓴다**(PG 는 읽기만).

    Args:
        client: OpenSearch 클라이언트.
        conn: DB 커넥션.
        asset_id: 색인할 자산.
        index: 대상 인덱스.
        channel: 평균 임베딩 채널.
        noise_patterns: 파일명 정제 패턴.
        topics_fn: 주제 조회 함수 주입 seam.

    Returns:
        색인한 문서 dict. **자산이 없거나 임베딩이 없으면 ``None``**(아무 것도 하지 않는다).
        같은 asset_id 로 다시 부르면 덮어쓰므로 재실행에 안전하다.
    """
    row = _fetch_one(conn, _ASSET_ONE_SQL, (channel, asset_id))
    if row is None:
        return None
    topics = topics_fn(conn, asset_id)
    doc = asset_to_doc(row, channel, noise_patterns=noise_patterns, topics=topics)
    client.index(index=index, id=str(asset_id), body=doc)
    return doc


def update_asset_topics(
    client: Any, index: str, asset_id: Any, topics: list[dict[str, Any]]
) -> None:
    """자산 문서의 **주제 3필드만**(topics·subtopics·topic_pairs) 부분 갱신한다(056 FR-203·059 — 전체 재색인 아님).

    G5 재색인 훅(관계 배치 꼬리·검토 승인 커밋 후)이 관계 변화를 반영할 때 쓰는 seam이다. OS
    ``update`` API 의 부분 문서(``body={"doc": {...}}``)로 ``topics``/``subtopics``/``topic_pairs``(059)
    3필드를 덮어쓴다 — ``asset_to_doc`` 과 동일한 ``_topics_doc_fields`` 로 조립해 두 경로의 주제 표현을 일치시킨다.
    ``topics`` 가 비면 두 필드를 **빈 값으로 갱신**해 강등/제거된 stale 주제를 지운다(SC-02). ``asset_to_doc``
    은 관계 없는 자산에서 필드를 생략하지만, 여기서는 이미 색인된 문서의 주제를 갱신·삭제해야 하므로
    비어도 필드를 실어 보낸다(전체문서 색인과 의도적으로 다른 대칭).

    **OpenSearch 에 쓴다**(부분 갱신).

    Args:
        client: OpenSearch 클라이언트.
        index: 대상 인덱스.
        asset_id: 갱신할 자산.
        topics: 새 주제 목록. **빈 리스트를 주면 기존 주제를 지운다** — 강등·제거된 주제가
            색인에 남지 않게 하는 의도적 동작이다.
    """
    client.update(index=index, id=str(asset_id), body={"doc": _topics_doc_fields(topics)})


def update_asset_about(client: Any, index: str, asset_id: Any, about: list[str]) -> None:
    """자산 문서의 ``about`` 필드만 부분 갱신한다(073 백필용 — 전체 재색인 아님).

    ``update_asset_topics`` 와 같은 방식이다.

    **OpenSearch 에 쓴다**(부분 갱신).

    Args:
        client: OpenSearch 클라이언트.
        index: 대상 인덱스.
        asset_id: 갱신할 자산.
        about: 새 개체 목록. **빈 리스트도 그대로 실어 보내** 예전 값을 지운다.
    """
    client.update(index=index, id=str(asset_id), body={"doc": {"about": [str(a) for a in about]}})


def ensure_about_mapping(client: Any, index: str) -> None:
    """기존 인덱스에 ``about``(keyword) 매핑을 추가한다(073 — put_mapping 은 멱등·재색인 불요).

    새 인덱스는 ``build_index_body`` 가 이미 포함하므로 이 함수는 **백필이 구 인덱스에 1회** 호출한다.

    **OpenSearch 매핑을 바꾼다**(문서 재색인은 필요 없다).

    Args:
        client: OpenSearch 클라이언트.
        index: 대상 인덱스.
    """
    client.indices.put_mapping(index=index, body={"properties": {"about": {"type": "keyword"}}})


def _bulk_actions(index: str, docs: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """문서들을 bulk 액션으로 감싼다.

    ``_id`` 를 ``asset_id`` 로 고정하므로 재실행하면 **덮어쓰기**가 되어 중복 문서가 생기지 않는다.

    Args:
        index: 대상 인덱스.
        docs: 색인할 문서들.

    Yields:
        bulk API 액션 dict.
    """
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
    topics_fn: TopicsFn = _default_topics_fn,
) -> tuple[str, int, list[Any]]:
    """registered 자산 전체를 OpenSearch 로 재동기화한다(복구 도구 — PG 읽기 전용 → OS 쓰기).

    ``_id=asset_id`` upsert 라 재실행은 멱등(중복 없음). 기본은 비파괴(없으면 생성·있으면 upsert);
    스키마 변경 시에만 ``recreate=True``. ``bulk_fn`` 은 색인 seam(기본 opensearch-py ``helpers.bulk``)
    으로, 단위 테스트가 가짜를 주입해 OS 없이 액션을 검증한다. 반환: ``(인덱스상태, 색인 건수, 오류 목록)``.
    ``nori_user_words``(인덱스 analyzer 사전)·``noise_patterns``(파일명 정제) 는 settings 단일 출처를
    IO 층(복구 도구)이 주입한다 — 미지정 시 순수 함수 기본값을 쓴다.

    **OpenSearch 에 쓴다**(PG 는 읽기만).

    Args:
        client: OpenSearch 클라이언트.
        conn: DB 커넥션.
        index: 대상 인덱스.
        channel: 평균 임베딩 채널.
        recreate: **True 면 인덱스를 지우고 다시 만든다**(파괴적 — 스키마를 바꿀 때만).
        dim: 벡터 차원.
        bulk_fn: 색인 함수 주입 seam. ``None`` 이면 opensearch-py ``helpers.bulk`` 를 지연 로드한다.
        nori_user_words: 분해 방지 외래어 목록.
        noise_patterns: 파일명 정제 패턴.
        topics_fn: 자산별 주제 조회 함수 주입 seam. 전체 재색인이 주제를 함께 실어야 색인된
            주제가 지워지지 않는다.

    Returns:
        ``(인덱스 상태, 색인 건수, 오류 목록)``. 상태는 ``created``·``recreated``·``exists``.
    """
    if bulk_fn is None:
        from opensearchpy import helpers

        bulk_fn = helpers.bulk

    status = ensure_index(
        client, index, recreate=recreate, dim=dim, nori_user_words=nori_user_words
    )
    actions = _bulk_actions(
        index,
        iter_asset_docs(conn, channel, noise_patterns=noise_patterns, topics_fn=topics_fn),
    )
    ok, errors = bulk_fn(client, actions, stats_only=False, raise_on_error=False)
    client.indices.refresh(index=index)
    return status, ok, list(errors)


def resolve_channel(channel: str | None) -> str:
    """색인에 쓸 임베딩 채널을 정한다 — 적재·검색과 **같은 채널**이어야 비교가 성립한다.

    Args:
        channel: 명시 채널. ``None`` 이면 운영 활성 채널로 해소한다.

    Returns:
        해소된 채널 이름.
    """
    from src.config.settings import active_embed_channel

    return channel if channel is not None else active_embed_channel()
