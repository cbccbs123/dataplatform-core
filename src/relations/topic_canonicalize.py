"""관계 topic 정규화 seam — 자유기입 라벨을 성장하는 정본 레지스트리로 수렴(spec 058 G2).

왜 이 seam 인가 (spec 058 §접근 C1·C2)
    관계 생성(``run_relations``)은 LLM 이 ``topic_ko`` 를 자유 기입해 동의어(요리/음식/식품)·
    계층·모달리티 난립을 낳는다. 정본 어휘 집합(``topic_registry``)과 해소 캐시(``topic_alias``)를
    **데이터**로 두고, persist 직전에 라벨을 정본으로 해소한다.

해소 파이프라인(retrieve-then-judge — 리랭커와 동형)
    ① ``lookup_alias`` 정확일치(캐시) → 반환(**LLM 0**·결정적).
    ② 미스 → ``knn_topic_candidates`` 임베딩 kNN 후보(pgvector ``<=>``·결정적 정렬).
    ③ ``judge_topic`` LLM 판정(후보 K개만 주입·``src.llm.client`` 단일 seam·temp=0):
       - 매칭 → ``topic_alias`` 에 raw→canonical **동결** → 반환(재실행 캐시 히트).
       - NEW → ``register_topic``(라벨 임베딩 저장) + self-alias 동결 → 신규 정본 반환.

헌법·불변식
    - **결정성(3조)**: 재사용=데이터 룩업, LLM 판정 결과는 alias 에 동결(재실행 LLM 0·SC-05).
      kNN 정렬 타이브레이커 = **거리 asc → topic_ko asc**(같은 입력 같은 순서).
    - **LLM 단일 seam(2조)**: ``judge_topic`` 만 ``complete_json``(temp=0·client 주입)을 쓴다.
    - **임베딩 불변식(034 교훈·plan Global Constraints)**: ``register_topic`` 은 항상 **비어있지 않은
      (0-노름 아님) 라벨 임베딩**만 저장한다(0-노름이면 ValueError). NULL/0-노름 topic 은 kNN
      불가시 → 동의어 재난립이므로 앱에서 금지.
    - **학습 0(2조)**: 임베딩은 추론만, 규칙은 결정적. 임베딩 seam 은 검색 질의 임베딩부를
      **재사용**한다(037 이후 공유·1536D ``st_bge``).
"""
from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.database.ids import uuid7_str

# 임베딩 채널: 라벨 문자열을 검색·적재와 같은 벡터 공간(BGE-M3·1536D)에서 임베딩해 kNN 후보 회수.
_EMBED_CHANNEL = "st_bge"


def _embed_label(raw_ko: str) -> list[float]:
    """라벨 문자열 → 1536D 임베딩(검색 질의 임베딩 seam 재사용).

    ``src.search.query_embed.embed_query_for_media_search`` + ``model_for_channel('st_bge')``
    를 **재사용**한다 — 임베딩 로직을 복제하지 않는다(037 이후 적재·검색 공유부). 무거운 임베더
    의존은 함수 안에서 지연 import 해 모듈 기동을 가볍게 유지하고, 단위 테스트가 이 함수를
    patch 로 대체할 수 있게 한다.
    """
    from src.config.settings import model_for_channel
    from src.search.query_embed import embed_query_for_media_search

    return embed_query_for_media_search(raw_ko, model_name=model_for_channel(_EMBED_CHANNEL))


def _l2_norm(vec: list[float]) -> float:
    """벡터 L2 노름(0-노름 판정용). numpy 의존 없이 순수 계산."""
    return sum(x * x for x in vec) ** 0.5


def lookup_alias(conn, raw_ko: str) -> str | None:
    """``topic_alias`` 정확일치 canonical_ko(캐시 룩업). 없으면 None.

    조회행 계약(graph_query 관례): canonical_ko 는 ``str()`` 로 강제.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT canonical_ko FROM topic_alias WHERE raw_ko = %s",
            (raw_ko,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    canonical = row["canonical_ko"]
    return str(canonical) if canonical is not None else None


def _lookup_topic_en(conn, topic_ko: str) -> str | None:
    """정본 ``topic_ko`` 의 registry ``topic_en``(정본 영문·FR-204). 없으면 None."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT topic_en FROM topic_registry WHERE topic_ko = %s",
            (topic_ko,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    en = row["topic_en"]
    return str(en) if en is not None else None


def knn_topic_candidates(conn, raw_ko: str, k: int = 5) -> list[str]:
    """raw_ko 임베딩 → ``topic_registry.embedding`` pgvector 코사인 상위 k topic_ko.

    - ``<=>`` 는 pgvector 코사인 거리(0=동일). 결정적 정렬 = **거리 asc → topic_ko asc**
      (동거리 타이브레이커가 없으면 PG 실행계획에 따라 순서가 흔들려 헌법 3조 재현성을 깬다).
    - 034 교훈: NULL/0-노름 registry 임베딩은 코사인이 NaN 이라 후보를 오염시키므로 제외.
    - 레지스트리가 비면 ``[]``(→ judge 는 LLM 없이 NEW). topic_ko 는 ``str()`` 로 강제.
    """
    vec = _embed_label(raw_ko)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT topic_ko
            FROM topic_registry
            WHERE embedding IS NOT NULL
              AND vector_norm(embedding) > 0
            ORDER BY embedding <=> %s::vector({FIX_EMBEDDING_DIMENSION}), topic_ko ASC
            LIMIT %s
            """,
            (vec, k),
        )
        rows = cur.fetchall()
    return [str(r["topic_ko"]) for r in rows]


def register_topic(conn, topic_ko: str, topic_en: str, *, source: str = "auto") -> None:
    """정본 topic 등록 — 라벨 임베딩 계산·``topic_registry`` INSERT(멱등).

    - **임베딩 불변식(plan)**: 비어있지 않은(0-노름 아님) 임베딩만 저장한다. 0-노름이면
      ``ValueError`` 로 차단(034 교훈: 0-노름 벡터는 NaN 코사인 → kNN 불가시 → 동의어 재난립).
    - ``ON CONFLICT (topic_ko) DO NOTHING`` — 이미 있으면 무시(멱등·동시성 안전).
    - topic_id 는 앱 발급 UUIDv7(런타임 테이블 관례). 벡터는 ``::vector(1536)`` 캐스트.
    """
    vec = _embed_label(topic_ko)
    if _l2_norm(vec) <= 0.0:
        raise ValueError(
            f"0-노름 임베딩은 등록 불가(kNN 불가시 → 동의어 재난립): topic_ko={topic_ko!r}"
        )
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO topic_registry (topic_id, topic_ko, topic_en, embedding, source)
            VALUES (%s, %s, %s, %s::vector({FIX_EMBEDDING_DIMENSION}), %s)
            ON CONFLICT (topic_ko) DO NOTHING
            """,
            (uuid7_str(), topic_ko, topic_en, vec, source),
        )


def _freeze_alias(conn, raw_ko: str, canonical_ko: str, decided_by: str) -> None:
    """해소 결과를 ``topic_alias`` 에 동결(결정성 캐시). 이미 있으면 무시(멱등).

    ``ON CONFLICT (raw_ko) DO NOTHING`` — 같은 raw 재판정 없이 최초 결정을 유지(재실행 결정적).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO topic_alias (raw_ko, canonical_ko, decided_by)
            VALUES (%s, %s, %s)
            ON CONFLICT (raw_ko) DO NOTHING
            """,
            (raw_ko, canonical_ko, decided_by),
        )


# LLM 판정 프롬프트 — 후보 K개만 주입(전체 레지스트리 주입 금지·프롬프트 비대화 방지·FR-203).
_JUDGE_PROMPT = """너는 한국어 주제(topic) 라벨을 정본 어휘로 수렴시키는 판정기다.
아래 "원본 라벨"이 "후보 정본 라벨" 중 하나와 **같은 주제**를 가리키면 그 후보 라벨을,
어느 후보와도 같은 주제가 아니면 "NEW" 를 고른다.

규칙:
- 반드시 JSON 객체 하나만 출력한다. 코드블록·설명 문장 금지.
- 형식: {{"match": "<후보 라벨 중 하나>"}} 또는 {{"match": "NEW"}}.
- 후보 목록에 없는 라벨을 지어내지 않는다. 애매하면 "NEW".

원본 라벨: {raw_ko}
후보 정본 라벨: {candidates}

출력: {{"match": "..."}}"""


def judge_topic(raw_ko: str, candidates: list[str], *, client=None) -> str | None:
    """LLM 재사용/신규 판정 — 매칭 topic_ko 또는 None(NEW).

    - 후보가 비면 **LLM 호출 없이** None(NEW). 레지스트리 포화 전(초기)엔 호출 급감.
    - ``src.llm.client.complete_json`` 단일 seam·temp=0·``client=`` 주입. 후보 K개만 프롬프트에.
    - LLM 이 후보 밖 라벨을 지어내면 안전하게 None(오병합 방지·결정성).
    """
    if not candidates:
        return None
    from src.llm.client import complete_json

    prompt = _JUDGE_PROMPT.format(raw_ko=raw_ko, candidates=candidates)
    out = complete_json(prompt, client=client)
    match = out.get("match")
    if isinstance(match, str) and match in candidates:
        return match
    return None


def canonicalize_topic(
    conn, raw_ko: str, raw_en: str | None = None, *, client=None
) -> dict[str, Any]:
    """자유기입 topic 라벨 → 정본 ``{canonical_ko, canonical_en, decided_by}``.

    1. 빈/None raw_ko → ``{raw_ko, raw_en or "general", "passthrough"}``(정규화 안 함).
    2. ``lookup_alias`` 히트 → canonical + registry en + ``decided_by="exact"``(**judge 미호출**).
    3. 미스 → ``knn_topic_candidates`` → ``judge_topic``:
       - 매칭 C → alias raw→C 동결(``decided_by="llm"``) → C + registry en.
       - None(NEW) → ``register_topic``(임베딩 저장) + self-alias 동결 → raw_ko + (raw_en or "general").
    """
    # 1) 빈/None → passthrough(정규화 안 함·하위호환)
    if not raw_ko or not str(raw_ko).strip():
        return {
            "canonical_ko": raw_ko,
            "canonical_en": raw_en or "general",
            "decided_by": "passthrough",
        }

    # 2) alias 정확일치(캐시) → LLM 0
    hit = lookup_alias(conn, raw_ko)
    if hit is not None:
        return {
            "canonical_ko": hit,
            "canonical_en": _lookup_topic_en(conn, hit),
            "decided_by": "exact",
        }

    # 3) 미스 → 임베딩 kNN 후보 → LLM 판정
    candidates = knn_topic_candidates(conn, raw_ko)
    match = judge_topic(raw_ko, candidates, client=client)
    if match is not None:
        # 재사용 판정 → alias 동결(재실행 결정적)
        _freeze_alias(conn, raw_ko, match, "llm")
        return {
            "canonical_ko": match,
            "canonical_en": _lookup_topic_en(conn, match),
            "decided_by": "llm",
        }

    # NEW → registry 등록(임베딩 저장) + self-alias 동결(다음 호출부터 캐시 히트)
    topic_en = raw_en or "general"
    register_topic(conn, raw_ko, topic_en, source="auto")
    _freeze_alias(conn, raw_ko, raw_ko, "llm")
    return {
        "canonical_ko": raw_ko,
        "canonical_en": topic_en,
        "decided_by": "llm",
    }
