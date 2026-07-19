"""관계 topic 정규화 seam — 닫힌 topic 분류체계 + 열린 subtopic(부모 스코프)로 수렴(spec 058 v2).

왜 이 seam 인가 (spec 058 v2 §접근 C1~C3·ADR 2026-07-07)
    관계 생성(``run_relations``)은 LLM 이 ``topic_ko``/``subtopic_ko`` 를 자유 기입해 동의어
    (요리/음식/식품)·계층 불일치·subtopic 충돌(김밥이 요리·식품 양쪽)·모달리티 누수를 낳았다.
    v1 의 "열린 어휘 + 쌍별 동의어 병합"은 실측 3연속 실패(광역 흡수↔충돌 잔존의 진동)로 폐기하고
    **2층 구조**로 전환한다:
      - **topic 층 = 닫힌 27+미분류**(``taxonomy_seed.json`` 정본·``source='taxonomy'``·parent NULL).
        신규 topic 을 만들지 않는다(고정 대분류) — 자유 라벨을 목록 중 하나로 **분류**할 뿐.
      - **subtopic 층 = 열린 성장 + 부모 스코프**(``parent_topic`` = 부모 topic_ko). 동의어 정리를
        부모 안에서만 수행해 동음이의(교통>사고 ≠ 사회>사고)를 보존하고, 오병합의 폭발 반경을
        부모 버킷 안에 가둔다.

topic 해소(``canonicalize_topic``) — 분류(classify)
    ① 빈/None → passthrough. ② 닫힌 정본 집합 정확일치 → 그대로. ③ alias 캐시(parent NULL) 히트
    → 정본. ④ 미스 → ``classify_topic`` LLM 분류(후보=닫힌 27+미분류 전체·temp=0): 목록 중 하나로
    분류, 애매하면 ``미분류``(+제안 라벨 로그) → alias 동결(``decided_by='classify'``). **신규 등록 없음**.

subtopic 해소(``canonicalize_subtopic``) — 부모 스코프 retrieve-then-judge
    ⓪ 빈/None·모달리티어·부모 범주명 → None(C7). ① (부모, raw) alias 정확일치 → 정본. ② 미스 →
    같은 부모 스코프 kNN. ③ 동의어-한정 ``judge_topic`` 재사용(후보 same-parent 만). ④ NEW →
    ``register_topic``(부모 스코프)·매칭=정본 → alias 동결(부모 스코프).

헌법·불변식
    - **결정성(3조)**: 재사용=데이터 룩업, LLM 결과는 alias 에 동결(재실행 LLM 0·SC-04v2).
      kNN 정렬 타이브레이커 = **거리 asc → topic_ko asc**(같은 입력 같은 순서).
    - **LLM 단일 seam(2조)**: ``classify_topic``/``judge_topic`` 만 ``complete_json``(temp=0·client 주입).
    - **임베딩 불변식(034 교훈·plan Global Constraints)**: ``register_topic`` 은 항상 **비어있지 않은
      (0-노름 아님) 라벨 임베딩**만 저장한다(0-노름이면 ValueError). NULL/0-노름 topic 은 kNN
      불가시 → 동의어 재난립이므로 앱에서 금지.
    - **subtopic 스코프 불변식(migration-reviewer v297 🟡·plan)**: subtopic 층 쓰기는 반드시 **같은
      ``parent_topic`` 스코프로 register → 그 스코프로 freeze** 순서를 지킨다(v297 FK 완화를 앱
      불변식으로 보증하므로). ``canonicalize_subtopic`` 은 (부모, raw) 스코프로만 registry/alias 기입.
    - **학습 0(2조)**: 임베딩은 추론만, 규칙은 결정적. 임베딩 seam 은 검색 질의 임베딩부를
      **재사용**한다(037 이후 공유·1536D ``st_bge``).
"""
from __future__ import annotations

import logging
from typing import Any

from psycopg.rows import dict_row

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.database.ids import uuid7_str
from src.relations.schema import normalize_subtopic_ko

logger = logging.getLogger(__name__)

# 임베딩 채널: 라벨 문자열을 **활성 임베딩 채널**(018 단일 출처·`active_embed_channel`)로 임베딩해 kNN
# 후보 회수. 062 이후 st_api(온프레미스 API bge-m3)로 전환 가능 — 시드(register_topic)와 런타임
# canonicalize 가 같은 활성 채널을 쓰므로 topic 임베딩 공간은 self-consistent(적재/검색 채널과도 일관).
# 전환 시 topic_registry 재시드 필요(시드 임베딩도 활성 채널로 재생성).

# 모달리티 블랙리스트(FR-302) — 매체어(텍스트/오디오/영상/이미지 + en)는 '주제'가 아니라 자산의
# 매체 형태이므로 하위주제(subtopic) 자격이 없다. **단일 출처**(plan §계약): 다른 모듈이 필요하면
# 여기서 import 해 공유한다(schema.py 중복 정의 금지). 한글은 대소문자 무관, 영문은 소문자로 비교.
_MODALITY_BLACKLIST: frozenset[str] = frozenset(
    {"텍스트", "오디오", "영상", "이미지", "text", "audio", "video", "image"}
)


def _embed_label(raw_ko: str) -> list[float]:
    """라벨 문자열 → 1536D 임베딩(검색 질의 임베딩 seam 재사용).

    ``src.search.query_embed.embed_query_for_media_search`` 를 **활성 채널**(`active_embed_channel`)로
    호출해 재사용한다 — 임베딩 로직을 복제하지 않는다(037 이후 적재·검색 공유부). ``channel`` 을 넘겨
    백엔드까지 활성 채널을 따른다(st_api면 API·그외 로컬·062). 무거운 임베더 의존은 함수 안에서 지연
    import 해 모듈 기동을 가볍게 유지하고, 단위 테스트가 이 함수를 patch 로 대체할 수 있게 한다.
    """
    from src.config.settings import active_embed_channel, model_for_channel
    from src.search.query_embed import embed_query_for_media_search

    channel = active_embed_channel()
    return embed_query_for_media_search(
        raw_ko, model_name=model_for_channel(channel), channel=channel
    )


def _l2_norm(vec: list[float]) -> float:
    """벡터 L2 노름(0-노름 판정용). numpy 의존 없이 순수 계산."""
    return sum(x * x for x in vec) ** 0.5


def lookup_alias(conn, raw_ko: str, *, parent_topic: str | None = None) -> str | None:
    """``topic_alias`` 정확일치 canonical_ko(캐시 룩업). 없으면 None.

    **부모 스코프(v297)**: ``parent_topic`` None=topic 층(parent NULL) / 값=subtopic 층(부모 스코프).
    v297 이후 raw_ko 는 층별 부분 유니크라 스코프를 함께 조여야 subtopic 오히트를 막는다(동음이의 보존).
    조회행 계약(graph_query 관례): canonical_ko 는 ``str()`` 로 강제.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        if parent_topic is None:
            cur.execute(
                "SELECT canonical_ko FROM topic_alias WHERE raw_ko = %s AND parent_topic IS NULL",
                (raw_ko,),
            )
        else:
            cur.execute(
                "SELECT canonical_ko FROM topic_alias WHERE raw_ko = %s AND parent_topic = %s",
                (raw_ko, parent_topic),
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


def _fetch_canonical_topics(conn) -> dict[str, str | None]:
    """닫힌 정본 topic 집합 조회 → ``{topic_ko: topic_en}`` (v2 topic 층).

    - **스코프**: ``parent_topic IS NULL`` 이고 ``source='taxonomy'`` 인 행만 = 닫힌 27+미분류 분류체계
      (``taxonomy_seed.json`` 시드본). subtopic 층·auto 등록 잔재를 배제한다.
    - **결정적 정렬**: ``ORDER BY topic_ko`` — 분류 프롬프트에 넣는 후보 목록 순서를 재실행마다 고정
      (헌법 3조 재현성). dict 는 삽입 순서를 보존하므로 호출부가 그대로 후보 순서로 쓴다.
    - 레지스트리 미시드면 ``{}``(→ canonicalize_topic 은 분류하지 않고 원본 유지·동작 보존).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT topic_ko, topic_en
            FROM topic_registry
            WHERE parent_topic IS NULL AND source = 'taxonomy'
            ORDER BY topic_ko ASC
            """
        )
        rows = cur.fetchall()
    return {
        str(r["topic_ko"]): (str(r["topic_en"]) if r["topic_en"] is not None else None)
        for r in rows
    }


def knn_topic_candidates(
    conn, raw_ko: str, k: int = 5, *, parent_topic: str | None = None
) -> list[str]:
    """raw_ko 임베딩 → ``topic_registry.embedding`` pgvector 코사인 상위 k topic_ko.

    - **부모 스코프(v297)**: ``parent_topic`` None=topic 층(parent NULL) / 값=subtopic 층(같은 부모만).
      subtopic 후보를 같은 부모 안으로 한정해 오병합 폭발 반경을 버킷 안에 가둔다(C3).
    - ``<=>`` 는 pgvector 코사인 거리(0=동일). 결정적 정렬 = **거리 asc → topic_ko asc**
      (동거리 타이브레이커가 없으면 PG 실행계획에 따라 순서가 흔들려 헌법 3조 재현성을 깬다).
    - 034 교훈: NULL/0-노름 registry 임베딩은 코사인이 NaN 이라 후보를 오염시키므로 제외.
    - 후보가 비면 ``[]``(→ judge 는 LLM 없이 NEW). topic_ko 는 ``str()`` 로 강제.
    """
    vec = _embed_label(raw_ko)
    scope_sql = "parent_topic IS NULL" if parent_topic is None else "parent_topic = %s"
    # 바인딩 순서 = SQL 텍스트상 %s 등장 순서: [WHERE parent(있으면)] → ORDER BY vec → LIMIT k.
    params: tuple = (vec, k) if parent_topic is None else (parent_topic, vec, k)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT topic_ko
            FROM topic_registry
            WHERE embedding IS NOT NULL
              AND vector_norm(embedding) > 0
              AND {scope_sql}
            ORDER BY embedding <=> %s::vector({FIX_EMBEDDING_DIMENSION}), topic_ko ASC
            LIMIT %s
            """,
            params,
        )
        rows = cur.fetchall()
    return [str(r["topic_ko"]) for r in rows]


def register_topic(
    conn,
    topic_ko: str,
    topic_en: str | None,
    *,
    source: str = "auto",
    parent_topic: str | None = None,
) -> None:
    """정본 topic/subtopic 등록 — 라벨 임베딩 계산·``topic_registry`` INSERT(멱등).

    - **임베딩 불변식(plan)**: 비어있지 않은(0-노름 아님) 임베딩만 저장한다. 0-노름이면
      ``ValueError`` 로 차단(034 교훈: 0-노름 벡터는 NaN 코사인 → kNN 불가시 → 동의어 재난립).
    - **부모 스코프(v297)**: ``parent_topic`` 이 None 이면 topic 층(닫힌 27+미분류), 값이 있으면
      subtopic 층(부모 스코프). ON CONFLICT 는 층별 **부분 유니크 인덱스**를 인덱스 술어(WHERE)로
      지정해 인퍼런스한다 — v297 이후 topic_ko 에 단일 유니크가 없으므로 술어가 필수다.
    - ``topic_en`` 은 None 허용(subtopic 층은 정본 en 을 추적하지 않음 → NULL 저장·후속 여지).
    - topic_id 는 앱 발급 UUIDv7(런타임 테이블 관례). 벡터는 ``::vector(1536)`` 캐스트.
    """
    vec = _embed_label(topic_ko)
    if _l2_norm(vec) <= 0.0:
        raise ValueError(
            f"0-노름 임베딩은 등록 불가(kNN 불가시 → 동의어 재난립): topic_ko={topic_ko!r}"
        )
    with conn.cursor() as cur:
        if parent_topic is None:
            # topic 층: 부분 유니크 인덱스 (topic_ko) WHERE parent_topic IS NULL.
            cur.execute(
                f"""
                INSERT INTO topic_registry
                    (topic_id, topic_ko, topic_en, embedding, source, parent_topic)
                VALUES (%s, %s, %s, %s::vector({FIX_EMBEDDING_DIMENSION}), %s, NULL)
                ON CONFLICT (topic_ko) WHERE parent_topic IS NULL DO NOTHING
                """,
                (uuid7_str(), topic_ko, topic_en, vec, source),
            )
        else:
            # subtopic 층: 부분 유니크 인덱스 (parent_topic, topic_ko) WHERE parent_topic IS NOT NULL.
            cur.execute(
                f"""
                INSERT INTO topic_registry
                    (topic_id, topic_ko, topic_en, embedding, source, parent_topic)
                VALUES (%s, %s, %s, %s::vector({FIX_EMBEDDING_DIMENSION}), %s, %s)
                ON CONFLICT (parent_topic, topic_ko) WHERE parent_topic IS NOT NULL DO NOTHING
                """,
                (uuid7_str(), topic_ko, topic_en, vec, source, parent_topic),
            )


def _freeze_alias(
    conn,
    raw_ko: str,
    canonical_ko: str,
    decided_by: str,
    *,
    parent_topic: str | None = None,
) -> None:
    """해소 결과를 ``topic_alias`` 에 동결(결정성 캐시). 이미 있으면 무시(멱등).

    **부모 스코프(v297)**: ``parent_topic`` None=topic 층 / 값=subtopic 층. ON CONFLICT 는 층별
    **부분 유니크 인덱스**를 인덱스 술어(WHERE)로 지정해 인퍼런스한다(raw_ko 단일 PK 는 v297 에서
    드롭됐으므로 술어 필수). 같은 스코프의 raw 재판정 없이 최초 결정을 유지(재실행 결정적).
    """
    with conn.cursor() as cur:
        if parent_topic is None:
            # topic 층: 부분 유니크 인덱스 (raw_ko) WHERE parent_topic IS NULL.
            cur.execute(
                """
                INSERT INTO topic_alias (raw_ko, canonical_ko, decided_by, parent_topic)
                VALUES (%s, %s, %s, NULL)
                ON CONFLICT (raw_ko) WHERE parent_topic IS NULL DO NOTHING
                """,
                (raw_ko, canonical_ko, decided_by),
            )
        else:
            # subtopic 층: 부분 유니크 인덱스 (parent_topic, raw_ko) WHERE parent_topic IS NOT NULL.
            cur.execute(
                """
                INSERT INTO topic_alias (raw_ko, canonical_ko, decided_by, parent_topic)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (parent_topic, raw_ko) WHERE parent_topic IS NOT NULL DO NOTHING
                """,
                (raw_ko, canonical_ko, decided_by, parent_topic),
            )


# topic 분류 프롬프트(v2·FR-201v2) — 자유 라벨을 닫힌 27+미분류 중 하나로 **분류**한다.
#
# judge(동의어 판정)와 다르다: judge 는 "원본과 후보가 서로 바꿔 써도 되는 같은 것인가"를 묻지만,
# classify 는 "이 라벨이 어느 대분류에 속하는가(is-a·소속)"를 묻는다. 닫힌 대분류는 상호 배타 설계라
# 상위분류 흡수(등산→스포츠·레저) 가 오히려 정답이다. 확신이 없으면 강제 배정하지 말고 '미분류'로 파킹
# (거버넌스 §4 가산 확장의 입력). 후보=닫힌 목록 전체를 주입(27+미분류는 프롬프트에 충분히 작다).
_CLASSIFY_PROMPT = """너는 한국어 주제(topic) 라벨을 **닫힌 분류체계**로 분류하는 분류기다.
아래 "분류할 라벨"을 "범주 목록"에 있는 범주 중 **정확히 하나**로 분류하라.

핵심 기준: "이 라벨은 어느 대분류에 속하는가?"(소속·is-a). 동의어 판정이 아니라 **분류**다.
- 라벨이 어느 범주의 한 종류·종목·장르·분야여도 그 범주로 분류한다(예: 등산 → 스포츠·레저).
- 어느 범주에도 자신 있게 넣기 어려우면 **"미분류"** 를 고른다(억지로 배정하지 마라).

규칙:
- 반드시 "범주 목록"에 있는 라벨 하나만 고른다. 목록에 없는 라벨을 지어내지 않는다.
- 확신이 없으면 "미분류".
- JSON 객체 하나만 출력한다. 코드블록·설명 문장 금지.
- 형식: {{"category": "<범주 목록 중 하나>"}}.

분류할 라벨(한글): {raw_ko}
분류할 라벨(영문): {raw_en}

범주 목록:
{categories}

출력: {{"category": "..."}}"""


def classify_topic(
    raw_ko: str, raw_en: str | None, categories: list[str], *, client=None
) -> str:
    """자유 topic 라벨을 닫힌 범주 목록 중 하나로 **분류**(동의어 판정 아님)·확신 없으면 '미분류'.

    - ``src.llm.client.complete_json`` 단일 seam·temp=0·``client=`` 주입. 후보=닫힌 목록 전체를 주입.
    - LLM 이 목록 밖 라벨을 지어내거나 응답이 누락되면 안전하게 ``"미분류"``(강제·오배정 방지·결정성).
    - 후보가 비면(레지스트리 미시드) 호출부(``canonicalize_topic``)가 분류 자체를 건너뛴다.
    """
    from src.llm.client import complete_json

    prompt = _CLASSIFY_PROMPT.format(
        raw_ko=raw_ko,
        raw_en=raw_en or "",
        categories="\n".join(f"- {c}" for c in categories),
    )
    out = complete_json(prompt, client=client)
    category = out.get("category")
    if isinstance(category, str) and category in categories:
        return category
    return "미분류"


# LLM 판정 프롬프트 — 후보 K개만 주입(전체 레지스트리 주입 금지·프롬프트 비대화 방지·FR-203).
#
# 동의어-한정(사용자 결정): judge 가 상위 카테고리·is-a·광역 부모까지 흡수하면(등산→스포츠·
# 사진→예술·물리학→과학) 정본이 지나치게 넓어져 "같은-주제" 탐색이 뭉개진다. 그래서 **원본과
# 후보가 서로 바꿔 써도 되는 같은 개념(동의어)일 때만** 매칭하고, 상위분류·종류(is-a)·부분-전체·
# 단순 연관은 전부 NEW 로 보낸다. 긍정/부정 예시를 프롬프트에 명시해 판정을 못박는다(temp=0·zero-shot).
_JUDGE_PROMPT = """너는 한국어 주제(topic) 라벨을 정본 어휘로 수렴시키는 판정기다.
아래 "원본 라벨"이 "후보 정본 라벨" 중 하나와 **같은 개념을 다른 말로 부르는 동의어**일 때만
그 후보 라벨을 고르고, 어느 후보와도 동의어가 아니면 "NEW" 를 고른다.

핵심 기준(딱 하나만 물어라): "원본과 후보는 **서로 바꿔 써도 되는 같은 것**인가?"
- 예(같은 것을 다른 말로 부름) → 그 후보를 고른다.
- 아니오(원본이 후보의 **상위 분류**이거나, 후보의 **한 종류·종목·장르·분야**(is-a)이거나,
  부분-전체·단순 연관 관계) → "NEW". 넓은 상위 카테고리로 빨아들이지 마라.

매칭 O — 동의어·동일 개념(서로 바꿔 써도 됨):
- 등산 == 산악, 관광 == 여행, 천문 == 천문학, 자연재해 == 재난, 수공예 == 공예.

매칭 X → "NEW" — 상위분류·종류(is-a)·수단·연관(같은 것이 아님):
- 등산 → 스포츠 (등산은 스포츠의 한 종목일 뿐, 스포츠 자체가 아님).
- 사진 → 예술 (사진은 예술의 한 장르, is-a).
- 물리학 → 과학 (물리학은 과학의 한 분야, is-a).
- 전기차 → 교통 (전기차는 교통 수단의 하나).
- 낚시 → 취미 (낚시는 취미의 한 종류).

규칙:
- 반드시 JSON 객체 하나만 출력한다. 코드블록·설명 문장 금지.
- 형식: {{"match": "<후보 라벨 중 하나>"}} 또는 {{"match": "NEW"}}.
- 후보 목록에 없는 라벨을 지어내지 않는다. 같은 것인지 확신이 없으면(상위분류·종류·연관이면) "NEW".

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
    """자유기입 topic 라벨 → 닫힌 정본 ``{canonical_ko, canonical_en, decided_by}`` (v2·FR-201v2).

    1. 빈/None raw_ko → passthrough(정규화 안 함·하위호환).
    2. 레지스트리 미시드(닫힌 집합 빈) → 원본 유지(동작 보존·G4 T401).
    3. 닫힌 정본 집합 정확일치 → 그대로(``decided_by="exact"``·LLM 0).
    4. alias 캐시(parent NULL) 히트 → 정본 + registry en(``decided_by="exact"``·LLM 0).
    5. 미스 → ``classify_topic`` LLM 분류(후보=닫힌 목록 전체·temp=0): 목록 중 하나로 분류, 애매하면
       ``미분류``(+제안 라벨 로그) → alias 동결(``decided_by="classify"``). **신규 topic 등록 없음**(고정 층).
    """
    # 1) 빈/None → passthrough(정규화 안 함·하위호환)
    if not raw_ko or not str(raw_ko).strip():
        return {
            "canonical_ko": raw_ko,
            "canonical_en": raw_en or "general",
            "decided_by": "passthrough",
        }

    # 닫힌 정본 집합(27+미분류·source taxonomy·parent NULL). {topic_ko: topic_en}·정렬됨.
    canonical = _fetch_canonical_topics(conn)

    # 2) 레지스트리 미시드 → 정본화 불가·원본 유지(동작 보존·플래그 시드 전 동치·G4 T401)
    if not canonical:
        return {
            "canonical_ko": raw_ko,
            "canonical_en": raw_en or "general",
            "decided_by": "passthrough",
        }

    # 3) 닫힌 정본 집합 정확일치 → 그대로(분류 불필요·LLM 0)
    if raw_ko in canonical:
        return {"canonical_ko": raw_ko, "canonical_en": canonical[raw_ko], "decided_by": "exact"}

    # 4) alias 캐시(parent NULL·topic 층) 히트 → 정본(재실행 결정적·LLM 0)
    hit = lookup_alias(conn, raw_ko)
    if hit is not None:
        return {"canonical_ko": hit, "canonical_en": canonical.get(hit), "decided_by": "exact"}

    # 5) 미스 → LLM 분류(닫힌 목록 중 하나·애매하면 미분류). dict 삽입순=정렬순이라 후보 순서 결정적.
    label = classify_topic(raw_ko, raw_en, list(canonical), client=client)
    _freeze_alias(conn, raw_ko, label, "classify")  # 결정 동결(재실행 캐시 히트·SC-04v2)
    if label == "미분류":
        # 미분류 파킹 = 가산 확장의 입력(거버넌스 §4). 제안 라벨(원본)을 로그로 남긴다(범주 추가 근거).
        logger.info(
            "topic 분류 미분류 폴백 — 제안 라벨: raw_ko=%r raw_en=%r", raw_ko, raw_en
        )
    return {"canonical_ko": label, "canonical_en": canonical.get(label), "decided_by": "classify"}


def canonicalize_subtopic(
    conn, topic_ko_canonical: str, raw_sub: str | None, *, client=None
) -> str | None:
    """subtopic 정규화 — 부모 스코프 해소(spec 058 v2·FR-202v2·C7).

    규칙(순서):
      0. ``raw_sub`` None/빈 → ``None`` · 모달리티 블랙리스트(매체어) → ``None`` ·
         정규화 라벨 == 부모 범주명(``topic_ko_canonical``) → ``None``(C7·redundant). registry 조회 전 차단.
      1. **(부모, raw) alias 정확일치**(부모 스코프 캐시) → 그 정본(LLM 0·결정적).
      2. 미스 → **같은 부모 스코프 kNN** 후보(``parent_topic=부모``).
      3. **동의어-한정 judge**(기존 ``judge_topic`` 재사용·후보 same-parent 만·temp=0): 매칭 or NEW.
      4. NEW → ``register_topic``(부모 스코프·라벨 임베딩)·매칭=그 정본 → **동일 부모 스코프로 alias 동결**.
         (스코프 불변식·plan: v297 FK 완화를 앱 불변식으로 보증 — register→freeze 를 같은 parent 로.)

    파라미터
        ``topic_ko_canonical``: 이미 정본화된 상위 topic(부모 스코프 키·범주명 비움 기준).
        ``client``: 동의어 judge 에 주입(캐시 미스에만 호출). 캐시 히트·조기 반환 경로는 LLM 0(결정성).
    """
    # 0a) None/빈/공백만 → 하위주제 없음
    if raw_sub is None or not str(raw_sub).strip():
        return None

    # 문자열 정규화(한 어절·기존 schema seam 재사용). 이후 판정은 정규화 라벨 기준.
    normalized = normalize_subtopic_ko(raw_sub)
    if not normalized:
        return None

    # 0b) 모달리티 블랙리스트(C7) — registry 조회 전에 차단(결정성·비용 0). 영문은 소문자로 비교.
    if normalized.lower() in _MODALITY_BLACKLIST:
        return None

    # 0c) 부모 범주명과 동일하면 비움(C7) — subtopic 이 부모 topic 을 반복하는 redundant 케이스.
    if normalized == topic_ko_canonical:
        return None

    # 1) (부모, raw) alias 정확일치(부모 스코프 캐시) → 정본(LLM 0)
    hit = lookup_alias(conn, normalized, parent_topic=topic_ko_canonical)
    if hit is not None:
        return hit

    # 2) 미스 → 같은 부모 스코프 kNN 후보(오병합 폭발 반경 = 부모 버킷·C3)
    #    = 부모 topic 의 기존 subtopic 후보 회수(동의어-한정 judge 에 넘긴다).
    candidates = knn_topic_candidates(conn, normalized, parent_topic=topic_ko_canonical)

    # 3) 동의어-한정 judge(후보 same-parent 만·strict·058 관계 경로 불변) → 매칭 or NEW.
    match = judge_topic(normalized, candidates, client=client)
    if match is not None:
        # 매칭(재사용) → 같은 부모 스코프로 alias 동결(재실행 캐시 히트·SC-04v2)
        _freeze_alias(conn, normalized, match, "llm", parent_topic=topic_ko_canonical)
        return match

    # 4) NEW → 부모 스코프로 register → 그 스코프로 freeze(스코프 불변식: register 가 freeze 보다 먼저).
    #    subtopic 은 정본 en 을 별도로 추적하지 않는다(subtopic_en 정본화는 후속 여지) → topic_en=None.
    register_topic(conn, normalized, None, source="auto", parent_topic=topic_ko_canonical)
    _freeze_alias(conn, normalized, normalized, "llm", parent_topic=topic_ko_canonical)
    return normalized
