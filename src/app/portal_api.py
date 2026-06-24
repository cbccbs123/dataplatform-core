"""일반 도메인 포탈 백엔드 API (FastAPI) — spec 010 P1 정식 진입점(plan D-6).

검색 → 상세 조회 → 단일/관계묶음 다운로드를 묶은 **정식 HTTP(JSON) 백엔드**다. 개발용
샘플(``sample_search_api``)이 증명한 FastAPI 패턴(`_lifespan`: load_dotenv→init_settings)을
정식 포탈 라우터로 승격한다. 환경은 ``PORTAL_API_ENV``(기본 dev).

엔드포인트(FR-015 계약)
    - ``GET /health``                       → ``{"status":"ok","env":...}``
    - ``GET /search``                       → ``{"query","results":{modality:[...]},"meta"}`` (모달리티별 그룹)
    - ``GET /assets/{asset_id}``            → AssetDetail (없음/의료/비registered → 404)
    - ``GET /assets/{asset_id}/download``   → 원본 스트리밍(Range 부분 요청 206 지원)
    - ``GET /assets/{asset_id}/bundle``     → 관계 ego-network zip

헌법 불변식
    - **FR-013(헌법 2조)**: 포탈은 신규 LLM 호출 0. 검색은 ``search_hybrid``(006 seam, 내부에서
      ``src/llm/client`` 경유)만 재사용한다. 본 파일은 LLM SDK·LLM seam 패키지를 직접
      import/호출하지 않는다(grep 가드로 검증).
    - **FR-014(헌법 7·10조)**: 검색은 ``exclude_domains={'medical'}``, 상세·다운로드·묶음 seed 는
      서비스 계층(``fetch_asset_detail``/``resolve_download_target``)의 노출 게이트로 의료를 배제한다.
    - **결정성(헌법 3조)**: 응답 순서는 ``group_ranked``(버킷 내 -round(sim,6)·asset_id)/
      ``graph_query`` 결정성에 위임한다(라우터는 추가 정렬을 하지 않는다).
    - **읽기 전용(헌법 6조)**: 스키마·쓰기 0. DB 접근은 조회 트랜잭션(idempotent)만.

인증·ext_meta 키 omit(spec 042 · 010 US4 흡수)
    JWT Bearer ``Depends(require_principal)`` — 검색·상세·다운로드·묶음.
    ``PORTAL_AUTH_DISABLED=1`` — dev bypass(anonymous → public). JWT → authorized(2-tier MVP).
    ``GET /me`` · ``POST /auth/token``(dev 전용, auth disabled 일 때만).
    Swagger ``/docs`` — ``Authorize`` 에 JWT **토큰만** 입력(Bearer 접두사 불필요).

ext_meta read 집행(042)
    ``project_ext_meta`` — clearance 미달 키 **응답에서 제거**(null 아님). ingest(039)는 DB 전량 유지.
"""
from __future__ import annotations

import mimetypes
import os
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from src.config.settings import get_current_settings

# 소비 서비스 함수들은 모듈 최상위에서 import 한다(테스트가 src.app.portal_api.<name> 으로 patch).
# search_hybrid 만 006 검색 seam(LLM 경유) — 그 외는 순수/조회 함수. 직접 LLM 호출은 없다.
from src.portal.asset_detail import fetch_asset_detail
from src.portal.auth import Principal, require_principal
from src.portal.auth.config import load_portal_auth_config
from src.portal.auth.dev_issuer import issue_dev_token
from src.portal.auth.schemas import DevTokenRequest
from src.portal.download import (
    build_bundle_zip,
    collect_bundle_assets,
    parse_range_header,
    resolve_download_target,
)
from src.portal.search_group import group_ranked
from src.registry.access_tier import project_ext_meta
from src.registry.ext_meta_field_registry import fetch_access_tiers
from src.search.search_filters import parse_search_filters
from src.search.search_service import search_hybrid

_ENV = os.getenv("PORTAL_API_ENV", "dev")
_VALID_MODALITIES = ("text", "image", "video", "audio")

# 의료(PHI) 배제 도메인 집합(FR-014). group_ranked 가 각 모달리티 버킷에서 이 도메인 행을 제거한다.
_EXCLUDE_DOMAINS = frozenset({"medical"})

# search_hybrid 의 버킷당 후보 풀 한도. 응답은 모달리티별 top-N(size)으로 자르지만, 2단계 시각
# 후보 풀·랭킹 품질을 위해 풀을 넉넉히 받은 뒤 group_ranked 에서 size 로 캡한다(의료 배제로 줄어도
# 충분한 잔여 확보). 전체 코퍼스 keyset 페이징은 006 재설계 후속.
_SEARCH_LIMIT_PER_BUCKET = 200

# 다운로드 스트리밍 청크 크기(64KiB) — 대용량 멀티모달 자산을 메모리에 다 올리지 않는다.
_STREAM_CHUNK = 64 * 1024


def _run_in_db(callback: Callable[[Any], Any]) -> Any:
    """PostgresUtil 조회 트랜잭션에서 ``callback(conn)`` 을 실행하는 단일 seam.

    상세/다운로드/묶음 핸들러의 DB 접근은 모두 이 함수를 거친다(테스트는 이 함수를 patch 로
    대체해 DB 없이 단위 검증). ``idempotent=True`` 조회 전용(쓰기 0, 헌법 6조).
    PostgresUtil 은 import 비용·풀 초기화를 늦추기 위해 함수 안에서 지연 import 한다.
    """
    from src.database.postgres_util import PostgresUtil

    db = PostgresUtil()
    with db:
        return db.execute_in_transaction(callback, idempotent=True)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # sample_search_api 와 동일 부트스트랩: .env.{env} 로드 → init_settings(필수 env 검증). 1회만.
    from src.config.settings import init_settings

    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / f".env.{_ENV}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(_ENV)
    yield


app = FastAPI(title="일반 도메인 포탈 API (010 P1)", lifespan=_lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """헬스 체크(부트스트랩·라우팅 확인용)."""
    return {"status": "ok", "env": _ENV}


@app.post("/auth/token")
def auth_token(body: DevTokenRequest) -> dict[str, str]:
    """dev JWT 발급 — ``PORTAL_AUTH_DISABLED=1`` 일 때만. 로컬 스모크·Swagger Authorize 용.

    운영(``PORTAL_AUTH_DISABLED=0``)에서는 404 — IdP 연동 전 dev 엔드포인트 노출 방지.
    """
    if not load_portal_auth_config().auth_disabled:
        raise HTTPException(status_code=404, detail="dev 토큰 발급 비활성")
    user_id = (body.user_id or body.username or "dev-user").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="username 또는 user_id 필요")
    return {"access_token": issue_dev_token(user_id=user_id), "token_type": "bearer"}


@app.get("/me")
def me(principal: Annotated[Principal, Depends(require_principal)]) -> dict[str, str]:
    """현재 principal(user_id·clearance)."""
    return {"user_id": principal.user_id, "clearance": principal.clearance}


def _project_grouped_search(
    conn: Any,
    grouped: dict[str, list[dict[str, Any]]],
    *,
    clearance: str,
) -> dict[str, list[dict[str, Any]]]:
    """검색 hit ``summary`` 에 tier 기반 키 omit (042).

    ``summary`` 를 mini ext_meta 로 ``project_ext_meta`` 에 넘김 — 미달 시 행에서 ``summary`` 키 제거.
    OpenSearch 색인은 변경 없음(API 응답 단계만).
    """
    tiers_cache: dict[str, dict[str, str]] = {}
    out: dict[str, list[dict[str, Any]]] = {}
    for modality, rows in grouped.items():
        projected: list[dict[str, Any]] = []
        for row in rows:
            domain = str(row.get("domain_label") or "general")
            if domain not in tiers_cache:
                tiers_cache[domain] = fetch_access_tiers(conn, domain)
            summary = row.get("summary") or ""
            masked = project_ext_meta(
                {"summary": summary} if summary else {},
                tiers_cache[domain],
                domain=domain,
                clearance=clearance,
            )
            new_row = dict(row)
            if summary and "summary" not in masked:
                new_row.pop("summary", None)
            elif "summary" in masked:
                new_row["summary"] = masked["summary"]
            projected.append(new_row)
        out[modality] = projected
    return out


def _search_min_scores() -> dict[str, float] | None:
    """settings 의 모달리티별 적합도 하한(``SEARCH_MIN_SCORE_*``)을 검색에 적용한다.

    ``run_search``/``sample_search_api`` 와 동일하게 ``search_hybrid`` 에 floor 를 넘겨, 점수
    무관한 약한 후보가 결과에 그대로 노출되는 것을 막는다(010 포탈은 이 배선을 빠뜨렸었다).
    settings 미초기화(라우팅 단위 테스트·오설정)면 ``None``(필터 비활성=기존 동작)으로 보수
    폴백한다 — 운영 진입점은 lifespan 이 ``init_settings`` 하므로 항상 설정값을 따른다.
    """
    try:
        return get_current_settings().search_min_scores
    except RuntimeError:
        return None


def _parse_search_mode(mode: str) -> str:
    """검색 mode 파라미터 검증(044 — auto|keyword)."""
    m = (mode or "auto").strip().lower()
    if m not in ("auto", "keyword"):
        raise HTTPException(
            status_code=400,
            detail=f"알 수 없는 mode: {mode!r} (허용: auto, keyword)",
        )
    return m


def _parse_modalities(modalities: str | None) -> list[str] | None:
    """콤마 구분 모달리티 문자열을 검증된 리스트로 파싱한다(미지정=None=전체).

    알 수 없는 모달리티는 ``HTTPException(400)`` 으로 거부한다.
    """
    if not modalities or not modalities.strip():
        return None
    mods = [m.strip() for m in modalities.split(",") if m.strip()]
    unknown = [m for m in mods if m not in _VALID_MODALITIES]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"알 수 없는 modality: {unknown} (허용: {list(_VALID_MODALITIES)})",
        )
    return mods or None


@app.get("/search")
def search(
    q: str = Query(..., description="검색 질의(한국어)"),
    modalities: str | None = Query(
        None, description="콤마 구분: text,image,video,audio (미지정=전체)"
    ),
    size: int = Query(20, ge=1, le=100, description="모달리티별 최대 결과 수(top-N)"),
    mode: str = Query("auto", description="검색 모드: auto(기본) | keyword(단어 포함 문서)"),
    file_ext: list[str] | None = Query(None, description="파일 확장자 필터(반복 가능, 예: txt,pdf)"),
    source_dataset: list[str] | None = Query(
        None, description="출처 데이터셋 필터(반복 가능: data1~3, wikipedia, youtube, unknown)"
    ),
    created_from: str | None = Query(None, description="생성일 하한(YYYY-MM-DD 또는 ISO datetime, UTC)"),
    created_to: str | None = Query(None, description="생성일 상한(YYYY-MM-DD 또는 ISO datetime, UTC)"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """006 하이브리드 검색을 **모달리티별 그룹**으로 반환한다(FR-001/002/003).

    내부: ``search_hybrid``(신규 LLM 호출 0, 006 seam) → ``group_ranked``(모달리티별 독립 랭킹·
    의료 배제, FR-014). 모달리티 간 점수 척도가 비교 불가라 단일 랭킹으로 합치지 않고 섹션별로
    제공한다 — 포탈은 어차피 text/image/video/audio 로 분류해 보여주면 되고, 이로써 점수가
    구조적으로 높은 영상이 다른 모달리티를 침범하는 문제를 피한다. 섹션별 top-N(``size``), 페이징
    없음(전체 코퍼스 keyset 페이징은 006 재설계 후속).
    """
    mods = _parse_modalities(modalities)
    search_mode = _parse_search_mode(mode)
    try:
        search_filters = parse_search_filters(
            file_ext=file_ext,
            source_dataset=source_dataset,
            created_from=created_from,
            created_to=created_to,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"필터 파라미터 형식 오류: {exc}") from exc

    # FR-013: 검색은 006 seam 만 호출(신규 LLM 호출 추가 없음). min_scores 로 모달리티별 적합도
    # 하한을 적용해 약한 후보를 거른다(settings 의 SEARCH_MIN_SCORE_*; 미초기화면 None=필터 비활성).
    result = search_hybrid(
        q,
        modalities=mods,
        limit_per_bucket=_SEARCH_LIMIT_PER_BUCKET,
        min_scores=_search_min_scores(),
        search_mode=search_mode,
        search_filters=search_filters,
    )

    # FR-014: 버킷별 의료 배제 + 모달리티별 독립 랭킹·top-N. results 는 {modality: [rows]}.
    grouped = group_ranked(result, limit_per_modality=size, exclude_domains=_EXCLUDE_DOMAINS)
    grouped = _run_in_db(
        lambda conn: _project_grouped_search(
            conn, grouped, clearance=principal.clearance
        )
    )
    counts = {modality: len(rows) for modality, rows in grouped.items()}

    meta: dict[str, Any] = {
        "query": q,
        "modalities": mods,
        "size": size,
        "counts": counts,
    }
    search_plan = (result.get("meta") or {}).get("search_plan")
    if search_plan is not None:
        meta["search_plan"] = search_plan
    if search_filters is not None:
        meta["filters"] = {
            "file_ext": list(search_filters.file_exts),
            "source_dataset": list(search_filters.source_datasets),
            "created_from": search_filters.created_from.isoformat()
            if search_filters.created_from is not None
            else None,
            "created_to": search_filters.created_to.isoformat()
            if search_filters.created_to is not None
            else None,
        }

    return {
        "query": q,
        "results": grouped,
        "meta": meta,
    }


@app.get("/assets/{asset_id}")
def asset_detail(
    asset_id: str,
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """자산 1건 상세(메타·임베딩 채널 요약·관계 미니뷰)를 반환한다(FR-004/005/006).

    노출 게이트(FR-014)는 ``fetch_asset_detail`` 이 책임진다 — 없음/비registered/의료면 None →
    404(자산 데이터 노출 0).
    """
    detail = _run_in_db(
        lambda conn: fetch_asset_detail(
            conn, asset_id=asset_id, clearance=principal.clearance
        )
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="자산을 찾을 수 없거나 노출 대상이 아님")
    return detail


def _guess_content_type(file_name: str, modality: str | None) -> str:
    """파일명 확장자 → MIME, 실패 시 모달리티 기반 폴백(최종 octet-stream)."""
    ctype, _ = mimetypes.guess_type(file_name)
    if ctype:
        return ctype
    fallback = {"text": "text/plain; charset=utf-8"}
    return fallback.get(modality or "", "application/octet-stream")


def _content_disposition(file_name: str) -> str:
    """RFC 6266 attachment 헤더(ASCII filename + UTF-8 filename* 병기).

    ASCII fallback 에서 큰따옴표·제어문자(CR/LF)·비-ASCII 를 제거해 헤더 분리/인젝션을 막는다
    (UTF-8 ``filename*`` 측은 ``quote`` 로 안전). file_name 은 basename 이라 현실 위험은 낮으나 위생.
    """
    ascii_safe = "".join(c for c in file_name if c.isascii() and c.isprintable() and c != '"')
    return f'attachment; filename="{ascii_safe}"; filename*=UTF-8\'\'{quote(file_name)}'


def _file_iterator(path: str, start: int, end: int) -> Iterator[bytes]:
    """``[start, end]`` (둘 다 포함) 구간을 청크 단위로 읽어 흘려보낸다(메모리 절약·스트리밍)."""
    remaining = end - start + 1
    with open(path, "rb") as fh:
        fh.seek(start)
        while remaining > 0:
            chunk = fh.read(min(_STREAM_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@app.get("/assets/{asset_id}/download")
def download(
    asset_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> StreamingResponse:
    """단일 자산 원본을 스트리밍한다 — HTTP ``Range`` 부분 요청(206) 지원(FR-007/009).

    절차
        1. ``resolve_download_target`` 로 노출 게이트(registered·비의료) 통과 확인 → None → 404.
        2. 원본 파일 존재 확인 → 없거나 접근 불가면 410(FR-009, 자산 데이터 노출 0).
        3. ``Range`` 헤더가 있으면 ``parse_range_header`` 로 구간 산출 → 206 + ``Content-Range``;
           범위 위반(ValueError) → 416. 헤더 없으면 200 전체.
    바이트 산출은 디스크 실제 크기 기준(무결성). ``Accept-Ranges: bytes`` 항상 고지.
    """
    target = _run_in_db(lambda conn: resolve_download_target(conn, asset_id=asset_id))
    if target is None:
        raise HTTPException(status_code=404, detail="다운로드 대상을 찾을 수 없거나 노출 대상이 아님")

    fs_path = target.get("fs_path")
    if not fs_path or not os.path.isfile(fs_path):
        # FR-009: DB 엔 있으나 원본이 사라짐/접근 불가 → 자산 노출 없이 410 Gone.
        raise HTTPException(status_code=410, detail="원본 파일이 존재하지 않거나 접근할 수 없음")

    file_size = os.path.getsize(fs_path)
    file_name = target.get("file_name") or os.path.basename(fs_path)
    content_type = _guess_content_type(file_name, target.get("modality"))

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": _content_disposition(file_name),
    }

    range_value = request.headers.get("range")
    try:
        rng = parse_range_header(range_value, file_size)
    except ValueError as exc:
        # 범위 위반(416 Range Not Satisfiable) — Content-Range 로 전체 크기 고지.
        raise HTTPException(
            status_code=416,
            detail=f"요청 범위 충족 불가: {exc}",
            headers={"Content-Range": f"bytes */{file_size}"},
        ) from exc

    if rng is None:
        start, end, status_code = 0, file_size - 1, 200
    else:
        start, end = rng
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(
        _file_iterator(fs_path, start, end),
        status_code=status_code,
        media_type=content_type,
        headers=headers,
    )


@app.get("/assets/{asset_id}/bundle")
def bundle(
    asset_id: str,
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> Response:
    """seed 자산 기준 관계 ego-network(seed + 1-hop active 이웃)를 zip 으로 묶어 내려준다(FR-008).

    seed 는 ``resolve_download_target`` 로 게이팅한다 — None(없음/의료/비registered) → 404. 이로써
    의료/비registered seed 의 묶음 진입을 차단한다.

    한계(MVP 범위): **이웃 자산의 의료 필터는 적용하지 않는다.** 현재 medical 자산이 존재하지
    않아(plan §0.1) 실효가 없고, 이웃별 도메인 게이트(RBAC·노출 정책)는 후속(013/RBAC, 010-follow)
    에서 ``collect_bundle_assets`` 수준에 추가한다.
    """

    def _work(conn: Any) -> list[dict[str, Any]] | None:
        # seed 게이트: 노출 불가(의료/비registered/없음) seed → None 신호 → 404.
        if resolve_download_target(conn, asset_id=asset_id) is None:
            return None
        return collect_bundle_assets(conn, seed_asset_id=asset_id)

    targets = _run_in_db(_work)
    if targets is None:
        raise HTTPException(status_code=404, detail="묶음 seed 를 찾을 수 없거나 노출 대상이 아님")

    # build_bundle_zip 은 파일 IO(원본 읽기) — DB 트랜잭션 밖에서 수행. 누락 파일은 부분 zip + manifest.
    zip_bytes = build_bundle_zip(targets)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": _content_disposition(f"bundle_{asset_id}.zip")},
    )


if __name__ == "__main__":
    # python -m src.app.portal_api 로도 띄울 수 있게(개발 편의).
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORTAL_API_PORT", "8001")))
