"""파일명·경로 **결정적 관계 후보**(레버 B) — 임베딩이 약한 관계를 경로 신호로 보강.

배경
    같은 시리즈(1부/2부), 원문↔요약·번역·전사 파생, 명시적 참조는 코사인 유사도가
    ``relation_min_sim`` 하한을 못 넘어 임베딩 후보에서 누락된다. 이 모듈은 **동일 디렉터리** +
    **파일명 stem 일치/근접**(C-1)인 registered 자산을 **결정적**(헌법 3조)으로 산출해
    임베딩 후보와 union 하기 위한 ``EmbeddingCandidate`` 목록을 만든다.

결정 근거(spec §Clarifications)
    C-1 stem 일치: 확장자 + 알려진 버전/파생 접미사를 정규식으로 **1회** 제거한 정규화 stem 의
        **정확 일치**(또는 확장자만 제거한 raw stem 의 정확 일치). 접미사 목록은 코드 상수.
    C-2 한도·정렬: 별도 한도 ``path_top_k``(기본 10). 근접도 DESC(정확 raw 일치 > 정규화 일치),
        asset_id ASC, LIMIT path_top_k. **동일 폴더만**(하위 디렉터리 제외).
    C-3 emb_score: path-only 후보는 ``emb_score=0.0``(결정적 sentinel). 임베딩 후보와의 union·
        우선 처리는 호출자(asset_entry)에서 한다.

설계 메모(테스트 가능성)
    stem 정규화·디렉터리 분리·근접도 판정은 **순수 함수**로 분리해 DB 없이 단위 검증한다.
    DB 의존부는 "소스 fs_path 조회 + 동일 디렉터리 registered 자산 조회"로 최소화하고,
    매칭·정렬·LIMIT 은 결정적 Python 으로 처리한다(SQL 의존 정렬 비결정성 회피).
"""
from __future__ import annotations

import posixpath
import re
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.relations.asset_candidates import EmbeddingCandidate

# C-1: 알려진 버전/파생 접미사. 정규식으로 stem 끝에서 **1회** 제거한다(확장 가능한 코드 상수).
#   _vN(_v1, _v2 ...) · _N부(_1부, _2부 ...) · _summary/_요약 · _번역 · _전사 · _copy/_사본.
#   결정성을 위해 IGNORECASE 만 쓰고 로케일 의존(casefold 등) 정규화는 피한다.
_SUFFIX_PATTERN = re.compile(
    r"(?:"
    r"_v\d+"            # 버전: _v1, _v2 ...
    r"|_\d+부"          # 연작: _1부, _2부 ...
    r"|_summary"        # 요약(영문)
    r"|_요약"           # 요약(국문)
    r"|_번역"           # 번역
    r"|_전사"           # 전사(transcript)
    r"|_copy"           # 사본(영문)
    r"|_사본"           # 사본(국문)
    r")$",
    re.IGNORECASE,
)


def split_dir_and_name(fs_path: str) -> tuple[str, str]:
    """``fs_path`` 를 (디렉터리, basename) 으로 분리(POSIX 경로 규칙, 결정적).

    posixpath 를 쓰는 이유: 결정성·로케일 비의존을 위해 실행 OS(os.sep)에 의존하지 않는다.

    Args:
        fs_path: 자산 경로 문자열. 구분자는 ``/`` 로 간주한다.

    Returns:
        ``(디렉터리, 파일명)``. 디렉터리가 없으면 앞이 빈 문자열이다.
    """
    d = posixpath.dirname(fs_path)
    name = posixpath.basename(fs_path)
    return d, name


def _raw_stem(filename: str) -> str:
    """확장자만 제거한 stem(소문자 정규화). 접미사는 남긴다 — 정확 raw 일치 판정용.

    Args:
        filename: 디렉터리를 뺀 파일명.

    Returns:
        소문자 stem. ``report_v1.pdf`` → ``report_v1``.
    """
    stem = posixpath.splitext(filename)[0]
    return stem.lower()


def normalize_stem(filename: str) -> str:
    """C-1: 확장자 + 알려진 버전/파생 접미사를 **1회** 제거한 정규화 stem.

    예) ``manual_v1.pdf`` / ``manual_v2.pdf`` → ``manual`` , ``report_summary.txt`` → ``report`` ,
        ``강의_1부.mp4`` → ``강의`` , ``a_summary_copy.txt`` → ``a_summary``(1회만 제거).
    소문자 정규화로 로케일 비의존·결정적(헌법 3조).

    Args:
        filename: 디렉터리를 뺀 파일명.

    Returns:
        정규화 stem. 접미사가 없으면 ``_raw_stem`` 과 같은 값이다.
    """
    stem = _raw_stem(filename)
    return _SUFFIX_PATTERN.sub("", stem)


def proximity_rank(
    *, src_raw: str, src_norm: str, cand_raw: str, cand_norm: str
) -> int | None:
    """후보와 소스의 stem 근접도. 정확 raw 일치=2 > 정규화 일치=1 > 불일치=None.

    raw(확장자만 제거)가 같으면 같은 파일명(확장자만 다른 변종)으로 보고 최고 근접도(2).
    raw 는 다르지만 접미사 제거 후 정규화 stem 이 같으면 파생/연작으로 보고 근접도 1.
    소스 stem 이 비면(예 ``_v1.txt`` 처럼 접미사만 있는 파일) 의미 있는 매칭이 아니므로 제외한다 —
    빈 stem 끼리 ''로 일치하는 오탐(서로 무관한 접미사-only 파일)을 막는다.

    Args:
        src_raw: 소스의 raw stem(확장자만 제거).
        src_norm: 소스의 정규화 stem(접미사까지 제거).
        cand_raw: 후보의 raw stem.
        cand_norm: 후보의 정규화 stem.

    Returns:
        근접도 ``2``(raw 일치) · ``1``(정규화 일치) · ``None``(후보 아님 — 호출자가 버린다).
    """
    if src_raw and cand_raw == src_raw:
        return 2
    if src_norm and cand_norm == src_norm:
        return 1
    return None


def find_path_signal_candidates(
    conn: Connection[Any],
    *,
    source_asset_id: str,
    limit: int,
) -> list[EmbeddingCandidate]:
    """같은 폴더에 있고 파일명 stem 이 맞는 자산을 관계 후보로 산출한다(임베딩 보완).

    **조회 전용**(DB 쓰기 없음). 반환 형태는 임베딩 후보와 **동일한 계약**(``EmbeddingCandidate``)이라
    호출자가 두 후보를 그대로 합칠 수 있다. path-only 라 ``emb_score`` 는 실측값이 아닌 ``0.0``
    sentinel 이다. 자기 자신과 무내용(자기주제 미부여) 자산은 임베딩 후보와 같은 기준으로 배제한다.

    Args:
        source_asset_id: 기준 자산. 이 자산의 폴더·파일명이 매칭 기준이 된다.
        limit: 반환할 최대 후보 수. **임베딩 후보의 ``top_k`` 와는 별도 한도**이며, 파일이 많은
            폴더에서 후보가 폭주하는 것을 막는 상한이다.

    Returns:
        후보 리스트(근접도 내림차순 → asset_id 오름차순으로 결정적 정렬). 소스 경로를 못 찾거나
        맞는 파일이 없으면 빈 리스트.

    정렬·LIMIT(C-2)
        근접도 DESC(정확 raw 일치 > 정규화 일치), 동률은 asset_id ASC, 상위 ``limit`` 만.
        결정성을 위해 매칭·정렬·LIMIT 을 Python 에서 처리한다(SQL ORDER BY 의 정규화 stem 표현
        복잡성·동률 비결정성을 피함). 동일 폴더 폭주는 이 limit 으로 차단된다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT fs_path FROM asset WHERE asset_id = %s LIMIT 1",
            (source_asset_id,),
        )
        src = cur.fetchone()
        if src is None or not src.get("fs_path"):
            return []
        src_path = str(src["fs_path"])
        src_dir, src_name = split_dir_and_name(src_path)
        src_raw = _raw_stem(src_name)
        src_norm = normalize_stem(src_name)

        # 동일 디렉터리 registered 자산 후보. self 는 SQL 에서 1차 제외.
        # LIKE 로 디렉터리 prefix 를 1차 필터링하되(인덱스 활용·행 수 축소),
        # 하위 디렉터리 배제는 Python 에서 디렉터리 **정확 일치**로 재확인한다.
        # LIKE 와일드카드/escape 문자는 결정성·오탐 방지를 위해 디렉터리 prefix 만 바인딩한다.
        dir_prefix = src_dir if src_dir.endswith("/") else src_dir + "/"
        cur.execute(
            """
            SELECT a.asset_id, a.fs_path, a.modality,
                   COALESCE(m.ext_meta->>'summary', '') AS summary,
                   COALESCE(m.ext_meta->>'keywords', '') AS keywords,
                   -- 066 FR-201(2026-07-15 B4): 후보 자기주제 동반 — 임베딩 후보(asset_candidates)와
                   --   동일 계약. 종전 path 후보는 topic 없이 프롬프트에 null 로 실렸다.
                   t.topic_ko    AS topic_ko,
                   t.subtopic_ko AS subtopic_ko
            FROM asset a
            LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id
            LEFT JOIN asset_topic t ON t.asset_id = a.asset_id
            WHERE a.status = 'registered'
              AND a.asset_id <> %s
              AND a.fs_path LIKE %s
              -- 066 FR-101(2026-07-15 B4): 미부여(무내용) 자산 배제 — 임베딩 후보의 EXISTS 와 동일.
              --   종전엔 path 신호 경로만 이 배제가 없어 무내용 자산이 관계 후보로 재유입 가능했다.
              AND EXISTS (SELECT 1 FROM asset_topic at WHERE at.asset_id = a.asset_id)
            """,
            (source_asset_id, like_escape(dir_prefix) + "%"),
        )
        rows = cur.fetchall()

    # 매칭·근접도·정렬·LIMIT — 결정적 Python.
    matched: list[tuple[int, str, dict[str, Any]]] = []
    for r in rows:
        aid = str(r["asset_id"])
        if aid == source_asset_id:  # 방어: SQL 제외 외에 한 번 더(결정성·self loop 차단)
            continue
        cand_path = str(r["fs_path"])
        cand_dir, cand_name = split_dir_and_name(cand_path)
        if cand_dir != src_dir:  # 하위 디렉터리 등 동일 폴더 아님 → 제외(C-2)
            continue
        rank = proximity_rank(
            src_raw=src_raw,
            src_norm=src_norm,
            cand_raw=_raw_stem(cand_name),
            cand_norm=normalize_stem(cand_name),
        )
        if rank is None:  # stem 불일치 → 후보 아님
            continue
        matched.append((rank, aid, r))

    # 정렬: 근접도 DESC(-rank 로 오름차순화), asset_id ASC. 결정적·안정 정렬.
    matched.sort(key=lambda t: (-t[0], t[1]))

    out: list[EmbeddingCandidate] = []
    for _rank, aid, r in matched[:limit]:
        # 주제는 방어적으로 None 을 허용한다(EXISTS 로 이미 걸렀지만 LEFT JOIN 이라 형식상 가능).
        topic_ko = r.get("topic_ko")
        subtopic_ko = r.get("subtopic_ko")
        out.append(
            {
                "id": aid,
                "file_uri": str(r["fs_path"]),
                "media_type": str(r["modality"]),
                # C-3: path-only 후보는 결정적 sentinel 0.0. union 시 임베딩 실측값 우선.
                "emb_score": 0.0,
                "summary": str(r["summary"] or ""),
                "keywords": str(r["keywords"] or ""),
                "topic_ko": str(topic_ko) if topic_ko is not None else None,
                "subtopic_ko": str(subtopic_ko) if subtopic_ko is not None else None,
            }
        )
    return out


def like_escape(value: str) -> str:
    """LIKE 패턴의 메타문자(``%`` ``_`` ``\\``)를 이스케이프해 입력을 **리터럴**로 매칭시킨다.

    디렉터리 경로에 ``_`` 가 흔하므로(예: ``my_docs``) 와일드카드로 오인되지 않게 escape 한다.
    PostgreSQL 기본 escape 문자는 백슬래시다. 관계 검토 화면의 검색어 등 다른 모듈도 이 함수를
    공유한다(같은 규칙을 여러 곳에 복사하지 않기 위한 단일 출처).

    Args:
        value: 사용자·경로에서 온 원본 문자열.

    Returns:
        LIKE 에 안전하게 넣을 수 있는 문자열. 호출자가 뒤에 ``%`` 를 붙여 prefix 매칭에 쓴다.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
