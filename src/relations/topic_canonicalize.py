"""자유 기입 주제 라벨을 정본 어휘로 수렴시킨다 — 대주제는 닫힌 목록, 세부주제는 부모 안에서 성장.

**흐름에서의 위치**: 관계를 저장하기 직전(``graph_persist``)에 불린다. LLM 이 붙인 라벨을 그대로
저장하면 같은 뜻의 표기가 끝없이 늘어나므로(요리/음식/식품), 여기서 하나로 모은다.

2층 구조인 이유
    - **대주제는 목록에서 고르기만** 한다(새로 만들지 않는다). 목록이 고정이라 분류 결과가 흩어지지
      않는다.
    - **세부주제는 부모 안에서만** 정리한다. 그래야 동음이의가 보존되고(교통>사고 ≠ 사회>사고),
      잘못 합쳐지더라도 피해가 그 부모 안에 갇힌다.

해소 순서(둘 다 캐시 우선 — LLM 은 마지막 수단)
    대주제: 빈 값 통과 → 정본 목록에 정확히 있으면 그대로 → 캐시(alias) 적중 → 없으면 LLM 이
    목록 중 하나로 분류하고 그 결정을 캐시에 동결한다(애매하면 '미분류'로 파킹).
    세부주제: 빈 값·모달리티 단어·부모 이름과 같으면 버림 → 캐시 적중 → 같은 부모 안에서 벡터
    후보 검색 → 동의어인지 LLM 판정 → 새 라벨이면 등록. 결정은 모두 캐시에 동결한다.

지켜야 할 불변식
    - **결정을 캐시에 동결한다** — 그래야 다시 돌려도 LLM 없이 같은 결과가 나온다. 벡터 후보
      정렬에도 2차 키를 둬 동점 순서가 흔들리지 않게 한다.
    - **0-노름 임베딩은 저장하지 않는다**(``register_topic`` 이 예외로 막는다). 저장되면 그 라벨은
      벡터 검색에 영영 안 잡혀 같은 뜻의 라벨이 계속 새로 생긴다.
    - **세부주제는 register → freeze 를 같은 부모로** 수행한다. 순서·스코프가 어긋나면 부모가 다른
      행이 만들어져 이후 조회가 어긋난다.
    - LLM 은 분류·동의어 판정 두 곳에서만, 단일 seam 으로 부른다(temperature 0).
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
    호출해 재사용한다 — 임베딩 로직을 복제하지 않는다. ``channel`` 을 함께 넘겨 로컬/원격 어느 쪽을
    쓸지도 적재와 일치시킨다. 무거운 임베더 의존은 함수 안에서 지연
    import 해 모듈 기동을 가볍게 유지하고, 단위 테스트가 이 함수를 patch 로 대체할 수 있게 한다.

    Args:
        raw_ko: 임베딩할 라벨 문자열.

    Returns:
        1536차원 실수 리스트. 빈 문자열이면 0-노름 벡터가 나올 수 있으므로 저장 전에
        ``_l2_norm`` 으로 걸러야 한다(``register_topic`` 이 그 역할을 한다).
    """
    from src.config.settings import active_embed_channel, model_for_channel
    from src.search.query_embed import embed_query_for_media_search

    channel = active_embed_channel()
    return embed_query_for_media_search(
        raw_ko, model_name=model_for_channel(channel), channel=channel
    )


def _l2_norm(vec: list[float]) -> float:
    """벡터의 L2 노름(길이)을 구한다 — 0-노름 판정용. numpy 없이 순수 계산.

    Args:
        vec: 임베딩 벡터.

    Returns:
        노름 값. **0.0 이면 내용이 없는 벡터**라 코사인 유사도가 NaN 이 되어 kNN 에 쓸 수 없다.
    """
    return sum(x * x for x in vec) ** 0.5


def lookup_alias(conn, raw_ko: str, *, parent_topic: str | None = None) -> str | None:
    """동결된 정본 라벨을 캐시(``topic_alias``)에서 찾는다 — 있으면 LLM 을 부르지 않는다.

    raw_ko 는 층별로만 유일하므로 스코프를 함께 조여야 subtopic 오히트를 막는다(동음이의 보존 —
    교통>사고와 사회>사고를 섞지 않는다).

    Args:
        conn: DB 커넥션.
        raw_ko: 찾을 원본 라벨.
        parent_topic: ``None`` 이면 **대주제 층**(부모 없음), 값이 있으면 **그 부모 아래 세부주제
            층**만 본다. 같은 raw_ko 라도 층이 다르면 다른 행이다.

    Returns:
        동결된 정본 라벨. 캐시에 없으면 ``None``(호출자가 판정 경로로 넘어간다).
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
    """정본 ``topic_ko`` 에 대응하는 registry ``topic_en``(정본 영문)을 조회한다.

    ⚠️ **이름이 밑줄로 시작하지만 레포 밖 소비자가 있다** — 파이프 레포
    ``processing/classify/asset_topic.py`` 가 이 함수를 import 해 자산 주제의 영문 정본을 채운다
    (중복 구현을 막으려고 코어 정본을 공유하는 구조). 코어만 grep 하면 미사용으로 보이므로
    **죽은 코드로 오인해 삭제하지 말 것**.

    Args:
        topic_ko: 정본 한글 주제.

    Returns:
        정본 영문. 행이 없거나 ``topic_en`` 이 NULL 이면 ``None``.
    """
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
    """대주제 정본 목록을 읽는다 — 분류 프롬프트에 넣을 후보가 된다.

    - **시드본만 본다**(부모 없음 + 출처가 taxonomy). 실행 중 자동 등록된 세부주제가 대주제 후보로
      섞이면 닫힌 목록이라는 전제가 무너진다.
    - **정렬해서 읽는다** — 후보 순서가 매번 같아야 LLM 분류 결과도 재현된다(dict 는 넣은 순서를
      유지하므로 호출부가 그대로 쓴다).

    Args:
        conn: DB 커넥션.

    Returns:
        ``{topic_ko: topic_en}``. ``topic_en`` 은 NULL 일 수 있어 값이 ``None`` 인 항목이 나온다.
        시드가 안 됐으면 빈 dict.
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

    - 세부주제 후보를 **같은 부모 안으로 한정**해, 잘못 합쳐지더라도 피해가 그 부모에 갇히게 한다.
    - ``<=>`` 는 pgvector 코사인 거리(0=동일). 결정적 정렬 = **거리 asc → topic_ko asc**
      (동거리 타이브레이커가 없으면 PG 실행계획에 따라 순서가 흔들려 헌법 3조 재현성을 깬다).
    - 0-노름 임베딩은 코사인이 NaN 이라 후보 순위를 통째로 오염시키므로 SQL 단계에서 제외한다.

    Args:
        conn: DB 커넥션.
        raw_ko: 후보를 찾을 원본 라벨(이 함수가 임베딩한다).
        k: 가져올 최대 후보 수. 프롬프트에 실릴 양이라 크게 잡으면 LLM 판정이 흔들린다.
        parent_topic: ``None`` 이면 대주제 층, 값이 있으면 **그 부모 아래**에서만 찾는다.

    Returns:
        가까운 순서의 정본 라벨 목록. 후보가 없으면 ``[]`` — 이때 judge 는 LLM 을 부르지 않고
        곧바로 NEW 로 처리한다.
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
      ``ValueError`` 로 차단한다(0-노름은 코사인이 NaN → 벡터 검색에 안 잡힘 → 같은 뜻 라벨이 재난립).
    - ``parent_topic`` 이 층을 가른다(None=대주제 / 값=그 부모의 세부주제). 층마다 **부분 유니크
      인덱스**를 쓰므로 ON CONFLICT 에 인덱스 조건(WHERE)을 함께 적어야 인식된다.
    - ``topic_en`` 은 None 허용(subtopic 층은 정본 en 을 추적하지 않음 → NULL 저장·후속 여지).
    - topic_id 는 앱 발급 UUIDv7(런타임 테이블 관례). 벡터는 ``::vector(1536)`` 캐스트.

    **DB에 쓴다**(이미 있으면 아무 것도 하지 않는 멱등 INSERT). 라벨 임베딩을 위해 임베더를 호출한다.

    Args:
        conn: DB 커넥션.
        topic_ko: 등록할 한글 라벨(임베딩 대상이기도 하다).
        topic_en: 영문 라벨. ``None`` 허용 — 세부주제 층은 영문 정본을 추적하지 않아 NULL 로 둔다.
        source: 출처 표기. ``taxonomy``(시드 정본) / ``auto``(실행 중 자동 등록) 구분에 쓴다.
        parent_topic: ``None`` 이면 대주제 층, 값이 있으면 그 부모 아래 세부주제로 등록한다.

    Raises:
        ValueError: 라벨 임베딩이 0-노름일 때. 저장하면 그 행은 kNN 에서 영영 안 잡혀 같은 뜻의
            라벨이 계속 새로 생긴다 — 그래서 조용히 넘기지 않고 즉시 막는다.
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

    ``parent_topic`` 이 층을 가른다. 층마다 **부분 유니크 인덱스**라 ON CONFLICT 에 인덱스 조건을
    함께 적어야 한다(raw_ko 하나만으로는 유일하지 않다). 같은 스코프의 raw 재판정 없이 최초 결정을 유지(재실행 결정적).

    **DB에 쓴다**(이미 있으면 무시). 이 동결이 있어야 같은 라벨을 다시 만나도 LLM 을 부르지 않는다.

    Args:
        conn: DB 커넥션.
        raw_ko: 원본(자유 기입) 라벨.
        canonical_ko: 이 라벨이 수렴한 정본.
        decided_by: 누가 정했는지 — ``exact``(정확 일치)·``classify``(분류 LLM)·``llm``(동의어 판정).
        parent_topic: ``None`` 이면 대주제 층, 값이 있으면 그 부모 스코프로 동결한다.
            **``register_topic`` 과 같은 부모로** 불러야 한다(스코프 불변식).
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

    Args:
        raw_ko: 분류할 한글 라벨.
        raw_en: 영문 라벨(보조 힌트). ``None`` 이면 빈 문자열로 프롬프트에 들어간다.
        categories: 고를 수 있는 **닫힌 범주 목록**. 이 목록 밖 응답은 전부 버린다.
        client: **테스트용 LLM 클라이언트 주입 seam** — 미주입이면 운영 온프레미스 LLM 을 쓴다.

    Returns:
        고른 범주 문자열. LLM 이 목록 밖 값을 지어내거나 응답이 비면 ``"미분류"``
        (억지 배정 대신 파킹 — 나중에 범주를 늘릴 근거로 로그가 남는다).
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

    Args:
        raw_ko: 판정할 원본 라벨.
        candidates: 같은 스코프의 정본 후보들. **비면 LLM 을 부르지 않는다**.
        client: **테스트용 LLM 클라이언트 주입 seam** — 미주입이면 운영 LLM.

    Returns:
        동의어로 판정된 정본 라벨, 또는 ``None``(=NEW·새 라벨로 등록해야 함).
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
    2. 정본 목록이 비어 있으면(시드 전) → 원본 유지.
    3. 닫힌 정본 집합 정확일치 → 그대로(``decided_by="exact"``·LLM 0).
    4. alias 캐시(parent NULL) 히트 → 정본 + registry en(``decided_by="exact"``·LLM 0).
    5. 미스 → ``classify_topic`` LLM 분류(후보=닫힌 목록 전체·temp=0): 목록 중 하나로 분류, 애매하면
       ``미분류``(+제안 라벨 로그) → alias 동결(``decided_by="classify"``). **신규 topic 등록 없음**(고정 층).

    **DB에 쓸 수 있다** — 5번 경로에서 판정 결과를 alias 로 동결한다(1~4번은 읽기만).

    Args:
        conn: DB 커넥션.
        raw_ko: 정본화할 자유 기입 라벨.
        raw_en: 영문 라벨(분류 힌트·폴백값). ``None`` 이면 ``general`` 로 채운다.
        client: **테스트용 LLM 클라이언트 주입 seam** — 미주입이면 운영 LLM.

    Returns:
        ``{canonical_ko, canonical_en, decided_by}``. ``decided_by`` 는 ``passthrough``(빈 입력·
        레지스트리 미시드) · ``exact``(정확 일치·캐시 히트·LLM 0) · ``classify``(LLM 분류) 중 하나다.
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
    """세부주제를 부모 스코프 안에서 정본으로 바꾼다.

    규칙(순서):
      0. ``raw_sub`` None/빈 → ``None`` · 모달리티 블랙리스트(매체어) → ``None`` ·
         정규화 라벨 == 부모 범주명(``topic_ko_canonical``) → ``None``(C7·redundant). registry 조회 전 차단.
      1. **(부모, raw) alias 정확일치**(부모 스코프 캐시) → 그 정본(LLM 0·결정적).
      2. 미스 → **같은 부모 스코프 kNN** 후보(``parent_topic=부모``).
      3. **동의어-한정 judge**(기존 ``judge_topic`` 재사용·후보 same-parent 만·temp=0): 매칭 or NEW.
      4. NEW → ``register_topic``(부모 스코프·라벨 임베딩)·매칭=그 정본 → **동일 부모 스코프로 alias 동결**.
         (register 와 freeze 를 **같은 부모로** 해야 한다 — 어긋나면 조회가 안 맞는 행이 생긴다.)

    **DB에 쓸 수 있다** — 3·4번 경로에서 registry 등록·alias 동결이 일어난다(0~1번은 읽기만).

    Args:
        conn: DB 커넥션.
        topic_ko_canonical: **이미 정본화된** 상위 주제. 부모 스코프 키이자, 세부주제가 부모 이름을
            그대로 반복할 때 비우는 기준이 된다.
        raw_sub: 자유 기입 세부주제. ``None``·빈 값·모달리티 단어(사진·영상 등)·부모와 동일하면
            **registry 를 보기도 전에** ``None`` 을 돌려준다.
        client: **테스트용 LLM 클라이언트 주입 seam** — 미주입이면 운영 LLM. 캐시 히트·조기 반환
            경로에서는 애초에 호출되지 않는다.

    Returns:
        정본 세부주제 라벨, 또는 ``None``(세부주제 없음 — 화면에서는 '기타'로 표시된다).
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
