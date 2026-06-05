"""[샘플·개발용] 하이브리드 검색 테스트 API (FastAPI).

⚠️ **개발/테스트 편의용 샘플이다 — 개발계획(specs/README)에 등재하지 않는다.**
정식 HTTP/포탈 레이어는 단계 E(spec 010/013)에서 별도로 설계한다. 이 파일은 그 전에
`search_service.search_hybrid` 를 브라우저·curl 로 빠르게 두들겨 보기 위한 일회성 도구다.
프로덕션 사용 금지(인증·레이트리밋·에러 표준화 없음).

실행(conda 환경):
    conda activate AuroraFS && uvicorn src.app.sample_search_api:app --port 8000
    # 또는: conda run -n AuroraFS python -m src.app.sample_search_api
예) curl "http://localhost:8000/search?q=무선%20충전기&modalities=text,image&fusion=alpha"
    curl "http://localhost:8000/search?q=회식&no_cutoff=true"   # 컷오프 끄고 약한 매칭까지
    curl "http://localhost:8000/search?q=회식&compact=true"     # 한눈에(순위·모달리티·점수·파일명·요약)
    curl "http://localhost:8000/search?q=회식&compact=true&skip_llm=true"  # LLM 스킵(~2.3s↓)+축약

부트스트랩은 run_search 와 동일: load_dotenv(.env.{env}) → init_settings → 첫 요청 시 LLM/DB 지연 초기화.
환경은 SAMPLE_API_ENV(기본 dev). 실 PostgreSQL + 온프레미스 LLM(질의 구조화) 필요.
"""
from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Query

_ENV = os.getenv("SAMPLE_API_ENV", "dev")
_VALID_MODALITIES = ("text", "image", "video", "audio")

# 결과 버킷 키 → 사람이 읽기 쉬운 모달리티 라벨(compact 뷰 표시용).
_BUCKET_TO_MODALITY = {
    "text_documents": "text",
    "audio": "audio",
    "image": "image",
    "video": "video",
}


def _finite(value: object) -> float:
    """None/NaN/inf/비수치 → 0.0 인 유한 실수(점수 정화)."""
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def _basename(uri: str) -> str:
    """file_uri(전체 경로/URI)에서 파일명만 뽑는다(결정적·순수)."""
    if not uri:
        return ""
    tail = uri.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail.split("?", 1)[0].split("#", 1)[0] or tail


def _clip_text(text: str, max_chars: int) -> str:
    """요약을 한 줄로 정규화하고 max_chars 초과분은 …로 자른다(한눈에 보기용)."""
    one_line = " ".join(text.split())
    if max_chars > 0 and len(one_line) > max_chars:
        return one_line[: max_chars - 1].rstrip() + "…"
    return one_line


def _compact_view(result: dict[str, Any], limit: int, summary_max: int) -> dict[str, Any]:
    """모달리티 버킷 결과를 한눈에 보기 좋은 단일 랭킹으로 축약한다.

    각 행은 순위·모달리티·합산 점수(similarity)·파일명·요약만 남긴다. 점수 내림차순,
    동점은 asset id 오름차순으로 정렬(결정적·헌법 3조). 전 모달리티를 합쳐 상위 ``limit`` 건만.
    """
    flat: list[tuple[float, str, dict[str, Any]]] = []
    for bucket, rows in (result.get("results") or {}).items():
        modality = _BUCKET_TO_MODALITY.get(bucket, bucket)
        for r in rows:
            score = round(_finite(r.get("similarity")), 4)
            iid = str(r.get("id", ""))
            flat.append(
                (
                    score,
                    iid,
                    {
                        "모달리티": modality,
                        "점수": score,
                        "파일명": _basename(str(r.get("file_uri", ""))),
                        "요약": _clip_text(str(r.get("summary", "")), summary_max),
                    },
                )
            )
    flat.sort(key=lambda t: (-t[0], t[1]))
    top = [{"순위": i, **row} for i, (_s, _id, row) in enumerate(flat[:limit], start=1)]
    return {"query": result.get("query", ""), "건수": len(top), "결과": top}


# 같은 소스(영상)로 묶을 때 쓰는 관계 종류. 같은 영상의 모달리티 간 근접중복/파생만 — 주제(same_domain)는
# 다른 영상을 잇는 관계라 묶음에서 제외(메가블롭 방지, 2026-06-05 진단으로 확정).
_SAME_SOURCE_KINDS = ("duplicate_near", "derived_from")


def _stem_name(fn: str) -> str:
    """파일명에서 확장자/파생 접미사를 떼 '소스 영상' 이름만(.ko.txt/.stt.txt 포함 처리)."""
    for suf in (".ko.txt", ".stt.txt"):
        if fn.endswith(suf):
            return fn[: -len(suf)]
    return fn.rsplit(".", 1)[0] if "." in fn else fn


def _fetch_same_source_edges(asset_ids: list[str]) -> list[tuple[str, str]]:
    """히트 자산들 사이의 '같은 소스' 묶음 엣지(active + duplicate_near/derived_from)를 (src,dst) 로.

    종류를 same-source 2종으로 한정해 주제(same_domain) 병합·메가블롭을 차단한다. status='active' 만.
    asset_id 가 UUID 컬럼이라 ::text 캐스팅 후 문자열 리스트와 비교(어댑터 이슈 회피).
    """
    ids = [a for a in asset_ids if a]
    if not ids:
        return []
    from psycopg.rows import dict_row

    from src.database.postgres_util import PostgresUtil

    sql = """
        SELECT sn.asset_id::text AS src, dn.asset_id::text AS dst
        FROM graph_edge ge
        JOIN node sn ON sn.node_id = ge.src_node AND sn.node_kind = 'asset'
        JOIN node dn ON dn.node_id = ge.dst_node AND dn.node_kind = 'asset'
        JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
        WHERE ge.status = 'active'
          AND rk.kind_code = ANY(%s)
          AND sn.asset_id::text = ANY(%s)
          AND dn.asset_id::text = ANY(%s)
    """

    def _run(conn):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (list(_SAME_SOURCE_KINDS), ids, ids))
            return cur.fetchall()

    db = PostgresUtil()
    with db:
        rows = db.execute_in_transaction(_run, idempotent=True)
    return [(r["src"], r["dst"]) for r in rows]


def _group_by_relation_view(result: dict[str, Any], summary_max: int) -> dict[str, Any]:
    """검색 결과를 '같은 소스 영상' 단위로 묶어 보여준다(graph_edge 관계 기반).

    같은 영상의 여러 모달리티(이미지·영상·오디오·텍스트)가 한 묶음으로 접힌다. manifest.json 류
    허브는 제외. 묶음점수=구성원 max, 정렬은 점수 내림차순(동점은 대표 asset_id, 결정적·헌법 3조).
    주의: 컷오프로 잘린 모달리티는 묶음에 안 들어온다(검색에 걸린 것만; 형제 끌어오기는 미적용).
    """
    # 1) 자산별 표시정보 평탄화(manifest 허브 제외, 동일 asset 은 최고점만).
    info: dict[str, dict[str, Any]] = {}
    for bucket, rows in (result.get("results") or {}).items():
        modality = _BUCKET_TO_MODALITY.get(bucket, bucket)
        for r in rows:
            aid = str(r.get("id") or "")
            fn = _basename(str(r.get("file_uri", "")))
            if not aid or fn == "manifest.json":
                continue
            score = round(_finite(r.get("similarity")), 4)
            prev = info.get(aid)
            if prev is None or score > prev["점수"]:
                info[aid] = {
                    "asset_id": aid,
                    "모달리티": modality,
                    "점수": score,
                    "파일명": fn,
                    "요약": _clip_text(str(r.get("summary", "")), summary_max),
                }

    # 2) 같은 소스 엣지로 union-find.
    parent = {a: a for a in info}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in _fetch_same_source_edges(list(info)):
        if a in parent and b in parent:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

    # 3) 묶음 구성 — 대표=최고점 멤버, 묶음점수=max.
    groups: dict[str, list[dict[str, Any]]] = {}
    for aid in info:
        groups.setdefault(_find(aid), []).append(info[aid])

    bundles = []
    for members in groups.values():
        members.sort(key=lambda m: (-m["점수"], m["asset_id"]))
        rep = members[0]
        bundles.append(
            {
                "_rep_id": rep["asset_id"],
                "묶음점수": rep["점수"],
                "대표": _stem_name(rep["파일명"]),
                "요약": rep["요약"],
                "구성": [
                    {"모달리티": m["모달리티"], "점수": m["점수"], "파일명": m["파일명"]}
                    for m in members
                ],
            }
        )
    bundles.sort(key=lambda bz: (-bz["묶음점수"], bz["_rep_id"]))
    out = [
        {"순위": i, **{k: v for k, v in b.items() if k != "_rep_id"}}
        for i, b in enumerate(bundles, start=1)
    ]
    return {"query": result.get("query", ""), "묶음수": len(out), "묶음": out}


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # run_search 와 동일 순서: .env.{env} 로드 → init_settings(필수 env 검증). 1회만.
    from src.config.settings import init_settings

    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / f".env.{_ENV}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(_ENV)
    yield


app = FastAPI(title="샘플 하이브리드 검색 API (개발용)", lifespan=_lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """헬스 체크(설정 초기화 여부 확인용)."""
    return {"status": "ok", "env": _ENV}


@app.get("/search")
def search(
    q: str = Query(..., description="검색 질의(한국어)"),
    modalities: str | None = Query(None, description="콤마 구분: text,image,video,audio (미지정=전체)"),
    limit: int = Query(20, ge=1, le=200, description="버킷당 최대 결과 수"),
    alpha: float = Query(0.75, ge=0.0, le=1.0, description="텍스트 하이브리드 임베딩 가중"),
    fusion: str = Query("alpha", pattern="^(alpha|rrf)$", description="융합 방식(alpha 기본 / rrf 프로토타입)"),
    no_cutoff: bool = Query(False, description="true 면 모달리티별 적합도 컷오프를 무시(약한 매칭까지 노출)"),
    skip_llm: bool = Query(False, description="true 면 LLM 질의 구조화를 생략(structured 직접 주입) — ~2.3s 절감, 단 동의어/영문 확장 없음"),
    compact: bool = Query(False, description="true 면 한눈에 보기용 축약(전 모달리티 합쳐 점수순 top-K; 순위·모달리티·점수·파일명·요약만)"),
    group_by_relation: bool = Query(False, description="true 면 관계(graph_edge active duplicate_near/derived_from)로 같은 소스 영상의 모달리티를 한 묶음으로 접어 반환(compact 보다 우선)"),
    summary_chars: int = Query(160, ge=0, le=2000, description="compact/묶음 요약 최대 글자수(0=자르지 않음)"),
) -> dict[str, Any]:
    """`search_service.search_hybrid` 를 그대로 호출해 모달리티 버킷 결과를 JSON 으로 반환한다.

    - ``fusion``: alpha(기본)/rrf 선택 — 단, grouped 출력은 cap 재정렬로 rrf 가 보이지 않을 수 있다(설계 §8 한계).
    - ``no_cutoff``: 디버깅용 — 컷오프(min_scores)를 빈 dict 로 줘 약한 후보까지 노출.
    - ``compact``: 전 모달리티를 합쳐 합산 점수(similarity) 내림차순 top-K(=limit)로, 순위·모달리티·
      점수·파일명·요약만 남긴 단일 리스트를 반환한다(브라우저·터미널에서 한눈에 보기용).
    - ``group_by_relation``: 같은 영상의 모달리티들을 graph_edge(active duplicate_near/derived_from)로
      묶어 1건으로 접는다. 같은 영상이 4번 나오던 게 묶음 카드 1개가 된다(spec 010 프로토타입).
    질의 구조화(LLM)·임베딩·DB 는 첫 호출 시 지연 초기화된다.
    """
    from src.config.settings import get_current_settings
    from src.search.search_service import search_hybrid

    mods = None
    if modalities and modalities.strip():
        mods = [m.strip() for m in modalities.split(",") if m.strip()]
        unknown = [m for m in mods if m not in _VALID_MODALITIES]
        if unknown:
            return {"error": f"알 수 없는 modality: {unknown} (허용: {list(_VALID_MODALITIES)})"}

    # 기본은 설정의 모달리티별 컷오프, no_cutoff 면 빈 dict(=전부 0.0=비활성).
    min_scores: dict[str, float] = {} if no_cutoff else get_current_settings().search_min_scores

    # skip_llm: structured 를 직접 주입하면 search_hybrid 가 structure_user_query(LLM) 를 건너뛴다.
    # 동의어·영문 확장 없이 원 질의를 그대로 임베딩+FTS 에 태운다(테스트·지연 절감용).
    structured = None
    if skip_llm:
        structured = {
            "semantic_query": q,
            "semantic_query_en": "",
            "keywords": [q],
            "keywords_en": [],
            "date_start": "unknown",
            "date_end": "unknown",
        }

    result = search_hybrid(
        q,
        modalities=mods,
        limit_per_bucket=limit,
        text_hybrid_alpha=alpha,  # 버그 수정: 받은 alpha 를 실제로 전달(이전엔 무시됐음)
        fusion=fusion,
        structured=structured,
        min_scores=min_scores,
    )
    if group_by_relation:
        return _group_by_relation_view(result, summary_chars)
    if compact:
        return _compact_view(result, limit, summary_chars)
    return result


if __name__ == "__main__":
    # python -m src.app.sample_search_api 로도 띄울 수 있게(개발 편의).
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("SAMPLE_API_PORT", "8000")))
