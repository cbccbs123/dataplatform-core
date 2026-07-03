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
    - **읽기 전용(헌법 6조) + append-only 감사(013 FR-012)**: 자산 데이터·스키마는 쓰기 0. 단 접근 이력
      (``access_log``)은 미들웨어가 append-only 로 적재한다(``_run_in_db_write``·best-effort).
      계보·검색·상세·다운로드 조회는 idempotent 트랜잭션이며, 신규 LLM 호출 0(SQL 만).

인증·ext_meta 키 omit(spec 042 · 010 US4 흡수)
    JWT Bearer ``Depends(require_principal)`` — 검색·상세·다운로드·묶음.
    ``PORTAL_AUTH_DISABLED=1`` — dev bypass(anonymous → public). JWT → authorized(2-tier MVP).
    ``GET /me`` · ``POST /auth/token``(dev 전용, auth disabled 일 때만).
    Swagger ``/docs`` — ``Authorize`` 에 JWT **토큰만** 입력(Bearer 접두사 불필요).

ext_meta read 집행(042)
    ``project_ext_meta`` — clearance 미달 키 **응답에서 제거**(null 아님). ingest(039)는 DB 전량 유지.
"""
from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.config.settings import get_current_settings
from src.portal._timeline_util import TIMELINE_INTERVALS

# 소비 서비스 함수들은 모듈 최상위에서 import 한다(테스트가 src.app.portal_api.<name> 으로 patch).
# search_hybrid 만 006 검색 seam(LLM 경유) — 그 외는 순수/조회 함수. 직접 LLM 호출은 없다.
from src.portal.access_log import (
    access_log_overview,
    access_log_stats,
    access_log_timeline,
    derive_access_action,
    query_access_logs,
    record_access,
)
from src.portal.asset_detail import fetch_asset_detail
from src.portal.asset_stats import (
    _RELATION_SCOPES,
    _SNAPSHOT_BUCKETS,
    asset_stats,
    asset_timeline,
    build_modality_overview,
    modality_detail,
    query_assets,
)
from src.portal.auth import Principal, authenticate_token, require_principal
from src.portal.auth.config import load_portal_auth_config
from src.portal.auth.dev_issuer import issue_dev_token
from src.portal.auth.schemas import DevTokenRequest
from src.portal.dashboard import build_dashboard_summary
from src.portal.download import (
    build_bundle_zip,
    collect_bundle_assets,
    parse_range_header,
    resolve_download_target,
)
from src.portal.lineage_query import (
    lineage_timeline,
    query_asset_lineage,
    query_lineage_feed,
    relation_proposed_summary,
)
from src.portal.search_group import group_ranked
from src.portal.thumbnail import THUMBNAILABLE_MODALITIES, cached_thumbnail
from src.registry.access_tier import project_ext_meta
from src.registry.ext_meta_field_registry import fetch_access_tiers

# 052 HITL 관계 검토 — review.py 검증된 단일 트랜잭션 로직을 HTTP 로 올리는 thin 레이어.
# 신규 3함수(list/bulk/revise) + 기존 promote 재사용. status 화이트리스트는 _REVIEW_STATUSES 공유.
from src.relations.review import (
    _REVIEW_STATUSES,
    bulk_review,
    list_edges_for_review,
    list_relation_kinds,
    promote_relation_kind,
    revise_edge,
)

# 056 포털 주제 표면 — 자산상세 same-topic·주제 브라우즈·검색 패싯(FR-501/502/503). 전부 순수 조회
# (graph_query seam·active-only·의료 제외)로 **신규 LLM 호출 0**(FR-505). 모듈 상단 import 로 두어
# 단위 테스트가 src.app.portal_api.<name> 을 patch 할 수 있게 한다(다른 조회 seam 과 동형).
from src.relations.topic_query import (
    assets_in_topic,
    find_topic_neighbor_groups,
    list_topics,
    project_asset_topics,
)
from src.search.search_filters import parse_search_filters
from src.search.search_service import search_hybrid

# 056 재색인 훅 — 검토 결정(승인/반려/정정) 커밋 후 관계 주제 변화를 OS 에 반영한다.
# 모듈 상단 import 로 두어 단위 테스트가 src.app.portal_api.reindex_asset_topics 를 patch 할 수 있게 한다.
from src.search.topic_reindex import reindex_asset_topics

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

# access_log 적재 실패(best-effort)·미들웨어 진단용 모듈 로거(서비스 응답엔 영향 없음).
_LOG = logging.getLogger("meta_extract.portal_api")


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


def _run_in_db_write(callback: Callable[[Any], Any]) -> Any:
    """access_log append-only 감사 write 용 트랜잭션(``idempotent=False``·commit).

    조회 seam(``_run_in_db``)과 분리한 별도 write seam — 자산 데이터·스키마는 무변경,
    오직 ``access_log`` 한 행 INSERT 만 한다(append-only·013 FR-012 감사 무결성). 미들웨어가 best-effort
    로만 호출하며(테스트는 이 함수를 patch), PostgresUtil 은 함수 안에서 지연 import 한다.
    """
    from src.database.postgres_util import PostgresUtil

    db = PostgresUtil()
    with db:
        return db.execute_in_transaction(callback, idempotent=False)


def _parse_dt(value: str | None) -> datetime | None:
    """``YYYY-MM-DD`` 또는 ISO datetime 문자열을 ``datetime`` 으로 파싱한다.

    빈 값은 ``None``(필터 비활성). 형식 오류는 ``HTTPException(422)`` 로 거부한다.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"날짜 형식 오류: {value!r}") from exc


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
    # graceful shutdown: 남은 fire-and-forget 감사 기록 태스크를 드레인한다(best-effort·013 FR-012).
    # _PENDING_TASKS 는 모듈 하단(app 정의 후)에 선언 — 종료 시점엔 모듈 로드 완료라 참조 가능.
    if _PENDING_TASKS:
        await asyncio.gather(*_PENDING_TASKS, return_exceptions=True)


app = FastAPI(title="일반 도메인 포탈 API (010 P1)", lifespan=_lifespan)


def _user_id_from_request(request: Request) -> str:
    """best-effort: ``Authorization: Bearer <token>`` → user_id. 없거나 검증 실패면 ``anonymous``.

    기록(감사) 용 식별이라 인증 실패가 응답을 막아선 안 된다 — 어떤 예외든 삼키고 anonymous 로.
    실제 접근 인가는 라우트의 ``require_principal`` 이 이미 책임진다(여기선 기록 라벨링만).
    """
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        try:
            return authenticate_token(auth[7:].strip()).user_id
        except Exception:  # noqa: BLE001 — 기록용 best-effort, 인증 실패가 응답을 막지 않음
            return "anonymous"
    return "anonymous"


def _record_access_safe(method: str, path: str, status_code: int, user_id: str) -> None:
    """데이터 접근(성공 응답)을 ``access_log`` 에 1행 적재. 비대상·오류 응답은 무시(best-effort).

    4xx/5xx 응답은 기록하지 않고, ``derive_access_action`` 이 데이터 라우트로 판정한 GET 만
    append-only 로 적재한다(검색·상세·다운로드·묶음). 그 외(감사 뷰·health 등)는 None → skip.
    """
    if status_code >= 400:
        return
    derived = derive_access_action(method, path)
    if derived is None:
        return
    action, asset_id = derived
    _run_in_db_write(
        lambda conn: record_access(conn, action=action, user_id=user_id, asset_id=asset_id)
    )


# fire-and-forget 기록 태스크 강참조 보관(GC 로 중도 소멸 방지). 완료 시 자동 제거.
_PENDING_TASKS: set[asyncio.Task] = set()


async def _record_access_bg(method: str, path: str, status_code: int, user_id: str) -> None:
    """동기 DB write 를 스레드풀에서 수행하는 비차단 기록 태스크. 어떤 예외도 삼킨다(best-effort)."""
    try:
        await run_in_threadpool(_record_access_safe, method, path, status_code, user_id)
    except Exception:  # noqa: BLE001 — 감사 기록 실패가 서비스에 전파되면 안 됨(best-effort·D2)
        _LOG.warning("access_log 기록 실패(무시): %s %s", method, path)


@app.middleware("http")
async def _access_log_middleware(request: Request, call_next: Callable) -> Any:
    """데이터 접근 이력을 append-only 로 적재한다(013 US3·FR-008).

    기록을 **응답 critical path 에서 분리**(fire-and-forget)한다 — 응답을 먼저 반환하고 기록은
    ``create_task`` 로 뒤에서 수행한다. 동기 DB write 를 await 하면 DB 지연/풀 고갈 시 모든 데이터
    응답이 지연되므로(best-effort 감사가 서비스 지연을 유발), await 하지 않는다(D2). 기록 실패·지연은
    응답 상태·지연 어디에도 영향이 없다. 응답 객체는 변경 없이 그대로 반환.
    """
    response = await call_next(request)
    try:
        user_id = _user_id_from_request(request)
        task = asyncio.create_task(
            _record_access_bg(request.method, request.url.path, response.status_code, user_id)
        )
        _PENDING_TASKS.add(task)
        task.add_done_callback(_PENDING_TASKS.discard)
    except Exception:  # noqa: BLE001 — 기록 스케줄 실패조차 응답을 깨면 안 됨(best-effort)
        _LOG.warning("access_log 기록 스케줄 실패(무시): %s %s", request.method, request.url.path)
    return response


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


@app.get("/admin/assets/{asset_id}/lineage")
def asset_lineage(
    asset_id: str,
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """자산 처리 이력(계보)을 발생 시각순으로 반환한다(013 US2·FR-004).

    조회 전용(``_run_in_db`` idempotent) — ``query_asset_lineage`` 가 ``asset_lineage`` 를
    시간순으로 끌어온다. 자산 데이터·스키마 쓰기 0·신규 LLM 0.

    상태 무관(운영상 ``failed``/``deferred`` 자산의 계보가 디버깅에 필요)이되 **의료(PHI)는 제외**
    — ``query_asset_lineage`` 가 asset 조인으로 medical 도메인 계보를 노출하지 않는다(헌법 10조·FR-014).
    미존재/의료/이력 없음은 빈 ``activities`` 로 200 반환(의도).
    """
    activities = _run_in_db(lambda conn: query_asset_lineage(conn, asset_id))
    return {"asset_id": asset_id, "activities": activities}


@app.get("/admin/access-logs")
def access_logs(
    user: str | None = Query(None, description="사용자 id 필터"),
    action: str | None = Query(None, description="동작 필터(search/asset_view/download/bundle)"),
    from_: str | None = Query(None, alias="from", description="기간 하한(YYYY-MM-DD 또는 ISO)"),
    to: str | None = Query(None, alias="to", description="기간 상한"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """API 접근 이력 조회(필터·페이징·013 US3·FR-009). 조회 전용·결정적·LLM 0.

    접근 정책(013 D4): **인증된 사용자 누구나**(``require_principal``·현 2-tier MVP). 감사 데이터는
    clearance 별 마스킹 없이 전사 노출 — admin/operator 한정은 RBAC 도입 시(향후 포탈) 조인다(의도적 개방).
    """
    since, until = _parse_dt(from_), _parse_dt(to)
    return _run_in_db(
        lambda conn: query_access_logs(
            conn, user_id=user, action=action, since=since, until=until,
            limit=limit, offset=offset,
        )
    )


@app.get("/admin/access-logs/stats")
def access_logs_stats(
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None, alias="to"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """접근 이력 기본 집계(총계·action별·user별·013 FR-009a). 조회 전용·결정적·LLM 0."""
    since, until = _parse_dt(from_), _parse_dt(to)
    return _run_in_db(lambda conn: access_log_stats(conn, since=since, until=until))


@app.get("/admin/access-logs/timeline")
def access_logs_timeline(
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None, alias="to"),
    interval: str = Query("day", description="버킷 단위: day(기본) | hour | month(054)"),
    action: str | None = Query(None, description="단일 api 필터(search/asset_view/download/bundle)"),
    group_by: str | None = Query(None, description="멀티시리즈 분할: action | user_id(미지정=단일)"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """접근 이력 시계열(버킷별 호출 수·그래프용·013 FR-009c). 조회 전용·결정적·LLM 0.

    ``group_by=action``(또는 user_id)이면 멀티시리즈 1회 응답(시리즈별 막대). 미지정이면 단일 시리즈.
    """
    if interval not in TIMELINE_INTERVALS:
        raise HTTPException(status_code=422,
                            detail=f"interval 은 {'|'.join(TIMELINE_INTERVALS)} 만 허용: {interval!r}")
    if group_by is not None and group_by not in ("action", "user_id"):
        raise HTTPException(status_code=422, detail=f"group_by 는 action|user_id 만 허용: {group_by!r}")
    since, until = _parse_dt(from_), _parse_dt(to)
    return _run_in_db(
        lambda conn: access_log_timeline(
            conn, since=since, until=until, action=action, interval=interval, group_by=group_by))


@app.get("/admin/access-logs/overview")
def access_logs_overview_endpoint(
    from_: str | None = Query(None, alias="from", description="기간 하한(YYYY-MM-DD 또는 ISO)"),
    to: str | None = Query(None, alias="to", description="기간 상한(exclusive)"),
    action: str | None = Query(None, description="추이 드릴다운 action(총계/action별 KPI 는 기간 전체)"),
    interval: str = Query("day", description="추이 버킷 단위: day(기본) | hour | month"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """접근 이력 overview 1회 응답(057 FR-301) — ``{total, by_action, timeline}``. 조회 전용·결정적·LLM 0.

    프론트가 stats+list+timeline 3회 순차 호출하던 것을 stats+timeline **1회**로 묶는다(``access_log_overview``·
    list 는 별도 페이징 유지). ``total``/``by_action`` 은 기간 전체 KPI, ``timeline`` 은 action 별 멀티시리즈
    (``action`` 지정 시 그 action 단일 시리즈로 드릴다운). interval 화이트리스트 위반은 422.
    """
    if interval not in TIMELINE_INTERVALS:
        raise HTTPException(status_code=422,
                            detail=f"interval 은 {'|'.join(TIMELINE_INTERVALS)} 만 허용: {interval!r}")
    since, until = _parse_dt(from_), _parse_dt(to)
    return _run_in_db(
        lambda conn: access_log_overview(
            conn, since=since, until=until, action=action, interval=interval))


@app.get("/admin/lineage")
def lineage_feed(
    from_: str | None = Query(None, alias="from", description="기간 하한(YYYY-MM-DD 또는 ISO)"),
    to: str | None = Query(None, alias="to", description="기간 상한"),
    activity: str | None = Query(None, description="활동명 필터(예: ingest.registered.v1)"),
    modality: str | None = Query(None, description="자산 모달리티 필터(text/image/video/audio 등)"),
    status: str | None = Query(None, description="자산 FSM 단계 필터(registered/failed 등)"),
    file_ext: str | None = Query(None, description="자산 파일 확장자 필터(예: txt, pdf, mp4)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """기간 내 전 자산 계보 피드(시간역순·페이징·013 FR-009b). 조회 전용·결정적·LLM 0·의료 제외.

    대시보드 슬라이스: 기간(from/to)·활동(activity)·자산 차원(modality·status·file_ext) 필터.
    드릴다운은 ``GET /admin/assets/{id}/lineage``·자산 상세는 ``GET /assets/{id}`` 합성.
    """
    since, until = _parse_dt(from_), _parse_dt(to)
    return _run_in_db(
        lambda conn: query_lineage_feed(
            conn, since=since, until=until, activity=activity, modality=modality,
            status=status, file_ext=file_ext, limit=limit, offset=offset))

# 055: GET /admin/lineage/stats 엔드포인트 제거(양쪽 프론트 미사용). lineage_stats 함수는
# /admin/dashboard/summary(dashboard.py)가 사용하므로 유지 — 엔드포인트만 잉여였다.


@app.get("/admin/lineage/timeline")
def lineage_timeline_endpoint(
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None, alias="to"),
    activity: str | None = Query(None, description="활동명 필터"),
    interval: str = Query("day", description="버킷 단위: day(기본) | hour | month(054)"),
    group_by: str | None = Query(None, description="멀티시리즈 분할: activity | modality | status(미지정=단일)"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """계보 시계열(누적 막대 차트 1회·access timeline 과 대칭). 의료 제외·결정적·LLM 0.

    ``group_by``(activity/modality/status) 주면 멀티시리즈, 미지정이면 단일 시리즈
    (access-logs/timeline 과 기본값 일관). 차트는 group_by=activity 를 명시해 호출.
    """
    if interval not in TIMELINE_INTERVALS:
        raise HTTPException(status_code=422,
                            detail=f"interval 은 {'|'.join(TIMELINE_INTERVALS)} 만 허용: {interval!r}")
    if group_by is not None and group_by not in ("activity", "modality", "status"):
        raise HTTPException(status_code=422, detail=f"group_by 는 activity|modality|status 만: {group_by!r}")
    since, until = _parse_dt(from_), _parse_dt(to)
    return _run_in_db(
        lambda conn: lineage_timeline(
            conn, since=since, until=until, activity=activity, interval=interval, group_by=group_by))


@app.get("/admin/asset-stats")
def asset_stats_endpoint(
    from_: str | None = Query(None, alias="from", description="생성일 하한(YYYY-MM-DD 또는 ISO)"),
    to: str | None = Query(None, alias="to", description="생성일 상한(exclusive)"),
    snapshot_buckets: bool = Query(
        False, description="운영 5버킷 집계(by_snapshot_bucket) 동반(계보 현황 화면·054·FR-201/202)"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """전체 자산 집계(FSM status·modality·domain·file_ext·date별·총계·013 FR-009e). 의료 제외·결정적·LLM 0.

    관리자 API(`/admin/*`) — 계보·접근이력·대시보드는 전부 `/admin` 프리픽스(D12). 사용자용
    검색·상세·다운로드는 루트 유지. ``from``/``to``(생성일·to exclusive·보완 v6) 지정 시
    by_file_ext 포함 6개 집계가 그 기간으로 스코프(기간별 파일 포맷 통계).

    ``snapshot_buckets=true``(054·계보 현황) 지정 시 응답에 ``by_snapshot_bucket``(운영 5버킷
    count·``sum==total``·FR-201/202)이 추가된다. 미지정(기본 False)이면 기존 응답이 완전히 불변이다.
    """
    since, until = _parse_dt(from_), _parse_dt(to)
    return _run_in_db(
        lambda conn: asset_stats(
            conn, since=since, until=until, snapshot_buckets=snapshot_buckets))


@app.get("/admin/assets")
def assets_list(
    status: str | None = Query(None, description="FSM 단계 필터(received/registered/failed 등)"),
    modality: str | None = Query(None, description="모달리티 필터(text/image/video/audio 등)"),
    domain: str | None = Query(None, description="도메인 필터(general/review; medical 은 제외됨)"),
    file_ext: str | None = Query(None, description="파일 확장자 필터(예: txt, pdf, mp4)"),
    created_from: str | None = Query(None, description="생성일 하한(YYYY-MM-DD 또는 ISO)"),
    created_to: str | None = Query(None, description="생성일 상한"),
    snapshot_bucket: str | None = Query(
        None, description="운영 스냅샷 버킷 필터(processing/deferred/registered/failed/"
                          "relation_proposed·054). 지정 시 status 대신 버킷으로 롤업 필터(C3)"),
    relation_scope: str = Query(
        "period", description="relation_proposed/registered 관계 제안 판별 스코프: "
                             "period(기본·자산 created 기간) | alltime(전 기간)"),
    with_content: bool = Query(False, description="행마다 요약·키워드 동반(모달리티 상세·보완 v6)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """자산 목록(FSM·modality·domain·file_ext·날짜 필터·페이징·013 FR-009f). 의료 제외·created_at 역순·LLM 0.

    ``with_content=true`` 면 행마다 요약(summary)·키워드(keywords) 동반(자산을 안 열고 내용 파악·v6).

    ``snapshot_bucket``(054·계보 현황·FR-103) 지정 시 FSM status 를 운영 5버킷으로 롤업해 필터한다
    (``status`` 는 무시·C3). 버킷/스코프 화이트리스트 검증(400)은 이 API 계층 책임이며(f-string
    인젝션 방지), 실제 술어 조립은 ``query_assets`` 가 한다. 둘 다 미지정 시 기존 동작 불변(하위호환).
    """
    # 화이트리스트 검증(API 계층 책임·f-string 인젝션 차단·400). snapshot_bucket=None 이면 검증 생략.
    if snapshot_bucket is not None and snapshot_bucket not in _SNAPSHOT_BUCKETS:
        raise HTTPException(
            status_code=400,
            detail=f"알 수 없는 snapshot_bucket: {snapshot_bucket!r} (허용: {list(_SNAPSHOT_BUCKETS)})")
    if relation_scope not in _RELATION_SCOPES:
        raise HTTPException(
            status_code=400,
            detail=f"알 수 없는 relation_scope: {relation_scope!r} (허용: {list(_RELATION_SCOPES)})")
    cfrom, cto = _parse_dt(created_from), _parse_dt(created_to)
    return _run_in_db(
        lambda conn: query_assets(
            conn, status=status, modality=modality, domain=domain, file_ext=file_ext,
            created_from=cfrom, created_to=cto, snapshot_bucket=snapshot_bucket,
            relation_scope=relation_scope, limit=limit, offset=offset,
            with_content=with_content))


@app.get("/admin/assets/modality/{modality}")
def modality_detail_endpoint(
    modality: str,
    from_: str | None = Query(None, alias="from", description="생성일 하한"),
    to: str | None = Query(None, alias="to", description="생성일 상한(exclusive)"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """모달리티 드릴다운 집계(보완 v6) — 해당 모달리티의 확장자·상태·일자별 분포 + 총계. 의료 제외·결정적·LLM 0.

    개요(`/admin/asset-stats`)의 한 모달리티를 파고들 때(예: video 안 mp4/mov 분포·일자 추이).
    ``from``/``to`` 로 개요 기간 필터와 일관 스코프. 콘텐츠 목록은
    `GET /admin/assets?modality=...&with_content=true` 로 합성.
    """
    since, until = _parse_dt(from_), _parse_dt(to)
    return _run_in_db(lambda conn: modality_detail(conn, modality, since=since, until=until))


@app.get("/admin/assets/modality/{modality}/overview")
def modality_overview_endpoint(
    modality: str,
    from_: str | None = Query(None, alias="from", description="생성일 하한(YYYY-MM-DD 또는 ISO)"),
    to: str | None = Query(None, alias="to", description="생성일 상한(exclusive)"),
    interval: str = Query("day", description="추이 버킷 단위: day(기본) | hour | month"),
    limit: int = Query(50, ge=1, le=200),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """모달리티 현황 BFF 1회 응답(057 FR-302) — ``{detail, timeline, first_page}``. 조회 전용·의료 제외·LLM 0.

    프론트 모달리티 상세가 stats+timeline+first-page 를 3~4회 순차 호출하던 것을 ``build_modality_overview``
    로 **한 트랜잭션**에 묶는다. timeline 은 interval=month 를 지원해 프론트 일→월 롤업(FR-303)도 제거한다.
    interval 화이트리스트 위반은 422. **라우트 순서(C8)**: 리터럴 3세그 경로라 catch-all 1세그
    ``/admin/assets/{asset_id}`` 보다 위(구체 경로)에 두어 새지 않게 한다.
    """
    if interval not in TIMELINE_INTERVALS:
        raise HTTPException(status_code=422,
                            detail=f"interval 은 {'|'.join(TIMELINE_INTERVALS)} 만 허용: {interval!r}")
    since, until = _parse_dt(from_), _parse_dt(to)
    return _run_in_db(
        lambda conn: build_modality_overview(
            conn, modality, since=since, until=until, interval=interval, limit=limit))


@app.get("/admin/asset-timeline")
def asset_timeline_endpoint(
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None, alias="to"),
    interval: str = Query("day", description="버킷 단위: day(기본) | hour | month(054)"),
    group_by: str | None = Query(
        None, description="멀티시리즈 분할: modality | status | domain | file_ext(미지정=단일)"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """자산 생성 일자 추이(보완 v6·계보 timeline 과 대칭). 의료 제외·결정적·LLM 0.

    ``group_by``(modality/status/domain/file_ext) 주면 멀티시리즈(예: 어느 날 어떤 포맷이 몇 개 생성),
    미지정이면 단일.
    """
    if interval not in TIMELINE_INTERVALS:
        raise HTTPException(status_code=422,
                            detail=f"interval 은 {'|'.join(TIMELINE_INTERVALS)} 만 허용: {interval!r}")
    if group_by is not None and group_by not in ("modality", "status", "domain", "file_ext"):
        raise HTTPException(status_code=422,
                            detail=f"group_by 는 modality|status|domain|file_ext 만: {group_by!r}")
    since, until = _parse_dt(from_), _parse_dt(to)
    return _run_in_db(
        lambda conn: asset_timeline(
            conn, since=since, until=until, interval=interval, group_by=group_by))


# 054·FR-301: 계보 현황 목록에서 자산 1건으로 드릴다운(관리자 관점). 사용자용 루트
# ``GET /assets/{id}`` 와 동일한 ``fetch_asset_detail`` 노출 게이트(없음/비registered/의료 → 404)를
# 재사용한다(자산 데이터 노출 0·헌법 10조). **라우트 순서(C8)**: 리터럴/2세그 라우트
# (``/admin/assets/modality/{modality}``·``/admin/assets/{asset_id}/lineage``)를 위에서 먼저 선언해야
# 이 catch-all 1세그 경로가 그것들을 가리지 않는다 — 그래서 이 라우트를 두 라우트보다 **뒤**에 둔다.
@app.get("/admin/assets/{asset_id}")
def admin_asset_detail(
    asset_id: str,
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """관리자 자산 1건 상세(계보 현황 드릴다운·FR-301). 노출 게이트·의료 제외는 ``fetch_asset_detail``
    책임(없음/비registered/의료 → None → 404). 조회 전용·LLM 0. clearance 로 ext_meta tier omit(042)."""
    detail = _run_in_db(
        lambda conn: fetch_asset_detail(
            conn, asset_id=asset_id, clearance=principal.clearance))
    if detail is None:
        raise HTTPException(status_code=404, detail="자산을 찾을 수 없거나 노출 대상이 아님")
    return detail


@app.get("/admin/dashboard/summary")
def dashboard_summary_endpoint(
    months: int = Query(6, ge=1, le=24, description="월별 시계열 창(개월·기본 6·최대 24)"),
    monthly_interval: str = Query(
        "day", description="월별 슬라이스 버킷 단위: day(기본·하위호환) | month(057 FR-303·프론트 롤업 제거)"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """운영 대시보드 집계 1회 응답(013 운영 후속·052 번들) — access·lineage·asset 3도메인 ×
    전체/오늘/월별/시간별. 조회 전용·의료 제외·결정적·LLM 0·마이그레이션 0.

    프론트 대시보드가 상세 로더 조합으로 9~11회 호출하던 것을 **단일 응답**으로 대체한다(HTTP
    왕복·커넥션 풀 churn·모달리티 월별의 자산 전수 스캔 N+1 제거). 검증된 순수 조회 함수 6종을
    ``build_dashboard_summary`` 가 **한 트랜잭션**에서 조합한다(``src/portal/dashboard.py``).
    오늘·최근 months개월 윈도우는 서버 ``now``(UTC) 기준 — 대시보드의 시각 상대 창이다(결정성 무관).

    ``monthly_interval``(057 FR-303) — 월별 슬라이스 버킷 단위. 기본 ``day``(하위호환·기존 동작 불변),
    ``month`` 면 월 버킷으로 내려 프론트 일→월 롤업을 제거한다. 월 범위엔 day|month 만 유효(hour 부적합) →
    그 외 422.
    """
    if monthly_interval not in ("day", "month"):
        raise HTTPException(status_code=422,
                            detail=f"monthly_interval 은 day|month 만 허용: {monthly_interval!r}")
    now = datetime.now(timezone.utc)
    return _run_in_db(
        lambda conn: build_dashboard_summary(
            conn, now=now, months=months, monthly_interval=monthly_interval))


@app.get("/admin/relations/proposed-summary")
def relations_proposed_summary_endpoint(
    from_: str | None = Query(None, alias="from", description="발생일 하한(YYYY-MM-DD 또는 ISO)"),
    to: str | None = Query(None, alias="to", description="발생일 상한(exclusive)"),
    interval: str = Query("day", description="버킷 단위: day(기본) | hour | month"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """관계 제안(relations.proposed) distinct 자산 수 + 발생 추이 1회 응답(057 FR-204). 조회 전용·의료 제외·LLM 0.

    admin 관계-제안 화면이 ``getLineageFeed(limit:200)`` 원시 피드를 프론트에서 distinct/버킷팅하던
    것을 서버로 이관한다 — 200 초과 시 과소집계되던 **실버그**를 ``COUNT(DISTINCT)`` 전기간 집계로
    바로잡는다(``relation_proposed_summary``·lineage occurred_at 기준). 판별 activity 는 054 스냅샷
    카운트와 단일 출처 공유. interval 화이트리스트 위반은 422(다른 timeline 엔드포인트와 동일).
    """
    if interval not in TIMELINE_INTERVALS:
        raise HTTPException(status_code=422,
                            detail=f"interval 은 {'|'.join(TIMELINE_INTERVALS)} 만 허용: {interval!r}")
    since, until = _parse_dt(from_), _parse_dt(to)
    return _run_in_db(
        lambda conn: relation_proposed_summary(conn, since=since, until=until, interval=interval))


# ── 052 HITL 관계 검토 API (CLI→HTTP thin 레이어) ──────────────────────────────
# review.py 의 검증된 단일 트랜잭션 로직을 포탈로 노출한다. RBAC = require_principal(C2·현
# 2-tier MVP·인증된 누구나·reviewer = principal.user_id). write 3종은 결정+감사를 한 write
# 트랜잭션에 묶고, 감사 실패는 savepoint 로 결정을 보존한다(best-effort·FR-502).


class RelationDecisionRequest(BaseModel):
    """일괄 승인/반려 요청 — UI 체크박스로 고른 edge_id 목록(C3·명시 목록만)."""

    edge_ids: list[str]


class RelationReviseRequest(BaseModel):
    """결정 정정 요청 — 사람 전용 status 전이(C4)."""

    edge_id: str
    to_status: str


def _record_relation_audit(conn: Any, *, action: str, reviewer: str, detail: dict) -> None:
    """결정과 **같은 write 트랜잭션**에 감사(access_log)를 기록한다(FR-203/502·D5).

    ``psycopg`` 의 중첩 ``conn.transaction()`` 은 SAVEPOINT 다 — 감사 INSERT 가 실패해도
    savepoint 만 롤백돼 바깥 결정 트랜잭션(approve/reject/revise/promote 갱신)은 보존된다
    (감사 best-effort·결정 무손상·013 일관). ``detail`` 은 jsonb 로 edge_id/kind_code 를 담고,
    ``access_log.asset_id`` 는 관계에 부적합하므로 NULL 로 둔다(자산 단위 아님).
    미들웨어 ``derive_access_action`` 은 GET 데이터 라우트만 판정하므로 이 POST 는 이중 기록 없다.
    """
    try:
        with conn.transaction():
            record_access(conn, action=action, user_id=reviewer, detail=detail)
    except Exception:  # noqa: BLE001 — 감사 실패가 결정 트랜잭션을 깨지 않음(best-effort·FR-502)
        _LOG.warning("relation 감사 기록 실패(무시): %s %s", action, detail)


@app.get("/admin/relations")
def relations_list(
    status: str = Query("proposed", description="검토 상태: proposed(큐) | active(승인) | rejected(비승인)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    q: str | None = Query(
        None, max_length=200,
        description="통합 텍스트 검색(edge_id·asset_id·파일명·reason·topic·최대 200자·FR-702)"),
    asset_id: str | None = Query(None, description="양끝 중 하나 정확 일치"),
    kind_code: str | None = Query(None, description="관계종류 코드 정확 일치"),
    modality: str | None = Query(None, description="양끝 중 하나 모달리티"),
    min_confidence: float | None = Query(None, description="신뢰도 하한(≥·0~1)"),
    max_confidence: float | None = Query(None, description="신뢰도 상한(≤·0~1)"),
    reviewed_by: str | None = Query(None, description="검토자 정확 일치"),
    from_: str | None = Query(None, alias="from", description="기간 시작(inclusive·ISO)"),
    to: str | None = Query(None, description="기간 끝(exclusive·ISO)"),
    date_on: str | None = Query(None, description="기간 대상 컬럼: created | reviewed"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """관계 검토 큐/내역을 status별 페이징 조회한다(FR-101/102/103 + G7 검색·필터·기간). 조회 전용·LLM 0.

    ``status`` 화이트리스트(``_REVIEW_STATUSES``) 위반은 400. 각 항목은 양끝 자산(asset_id·
    파일명·모달리티)·kind_code·confidence·reason·topic·reviewed_by/at·created_at 를 담아
    "무엇을 승인하는지" 식별 가능하게 한다. 의료(PHI) 자산 엣지는 제외(헌법 10조·review.py SQL).

    G7 선택 필터(전부 생략 시 현행 동작·하위 호환·SC-011):
    - ``min>max`` 또는 conf∉[0,1] → **400**(FR-704) · ``date_on∉{created,reviewed}`` → **400**(FR-752)
    - ``from``/``to`` 는 ``_parse_dt``(형식 오류 422·013 관례) · 파싱 후 ``from>to`` → **400**(FR-751)
    - 빈/공백 ``q`` 는 무시(None·팀 결정·FR-702)
    - ``date_col`` 결정: date_on 명시 시 매핑, 생략 시 status별 자동
      (proposed→created_at·active/rejected→reviewed_at·FR-752)
    """
    if status not in _REVIEW_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"알 수 없는 status: {status!r} (허용: {list(_REVIEW_STATUSES)})",
        )
    # confidence 범위 검증(FR-704) — 0~1 밖·min>max 는 400(의미 없는 요청).
    for name, val in (("min_confidence", min_confidence), ("max_confidence", max_confidence)):
        if val is not None and not (0.0 <= val <= 1.0):
            raise HTTPException(status_code=400, detail=f"{name} 는 0~1 범위여야 함: {val}")
    if (min_confidence is not None and max_confidence is not None
            and min_confidence > max_confidence):
        raise HTTPException(
            status_code=400,
            detail=f"min_confidence({min_confidence}) > max_confidence({max_confidence})")

    # date_on → date_col 매핑(FR-752). 생략 시 status별 자동(proposed=제안일·검토됨=검토일).
    _DATE_ON_MAP = {"created": "created_at", "reviewed": "reviewed_at"}
    if date_on is not None:
        if date_on not in _DATE_ON_MAP:
            raise HTTPException(
                status_code=400,
                detail=f"알 수 없는 date_on: {date_on!r} (허용: {list(_DATE_ON_MAP)})")
        date_col = _DATE_ON_MAP[date_on]
    else:
        date_col = "created_at" if status == "proposed" else "reviewed_at"

    # 기간 파싱(형식 오류 422·013 _parse_dt) 후 from>to 는 400(FR-751).
    since, until = _parse_dt(from_), _parse_dt(to)
    if since is not None and until is not None and since > until:
        raise HTTPException(status_code=400, detail=f"from({from_}) > to({to})")

    # 빈/공백 q 는 무시(FR-702·필터 비활성).
    q_clean = q.strip() if q else None
    q_clean = q_clean or None

    return _run_in_db(
        lambda conn: list_edges_for_review(
            conn, status=status, limit=limit, offset=offset,
            q=q_clean, asset_id=asset_id, kind_code=kind_code, modality=modality,
            min_confidence=min_confidence, max_confidence=max_confidence,
            reviewed_by=reviewed_by, since=since, until=until, date_col=date_col)
    )


# graph_edge → node 역조인으로 엣지 양끝 자산 id 를 해소(056 재색인 훅용·graph_query 조인 스타일).
_EDGE_ENDPOINT_SQL = """
SELECT sn.asset_id AS src_asset, dn.asset_id AS dst_asset
FROM graph_edge ge
JOIN node sn ON sn.node_id = ge.src_node AND sn.node_kind = 'asset'
JOIN node dn ON dn.node_id = ge.dst_node AND dn.node_kind = 'asset'
WHERE ge.edge_id = ANY(%s)
"""


def _resolve_edge_endpoint_assets(edge_ids: list[str]) -> list[str]:
    """엣지 id 들의 **양끝 자산 id(str)** 를 중복 없이 조회한다(읽기 seam·056 재색인 훅용).

    조회 전용(``_run_in_db``·읽기 트랜잭션·헌법 6조). 값은 ``str()``·입력 순서 보존 dedup.
    """
    if not edge_ids:
        return []

    def _work(conn: Any) -> list[str]:
        with conn.cursor() as cur:
            cur.execute(_EDGE_ENDPOINT_SQL, (edge_ids,))
            rows = cur.fetchall()
        seen: dict[str, None] = {}
        for r in rows:
            for aid in (r[0], r[1]):  # 기본 커서(튜플 행) — src_asset·dst_asset 순
                if aid is not None:
                    seen[str(aid)] = None
        return list(seen.keys())

    return _run_in_db(_work)


def _reindex_review_topics(results: list[dict[str, Any]]) -> None:
    """검토 결정(승인/반려/정정)으로 status 가 바뀐 엣지 양끝 자산의 OS topics 를 재색인한다.

    (056 FR-301~304) — ``_run_in_db_write`` 커밋 **후**에 호출한다. 관계 결정 트랜잭션 **밖**이라
    재색인 실패가 결정을 롤백하거나 HTTP 응답/승인을 바꾸지 않는다(FR-304·승인 무손상). ``ok=True``
    엣지만 대상으로 그 양끝 asset_id 를 해소(graph_edge→node)해 ``reindex_asset_topics`` 에 넘긴다
    (그 함수가 각 자산의 active 이웃까지 함께 재색인). OS 동기화 off(020/038 게이트)면 스킵한다.

    **격리**: 설정 미초기화·OS 미도달·재색인 오류 등 **어떤 예외도 삼킨다**(warning 로그).
    """
    edge_ids = [r["edge_id"] for r in results if r.get("ok")]
    if not edge_ids:
        return  # 실제 status 변경(ok=True)이 없으면 재색인 대상 없음
    try:
        if not getattr(get_current_settings(), "opensearch_sync_enabled", False):
            return  # OS 동기화 off — 재색인 스킵(020/038 게이트와 정합)
        asset_ids = _resolve_edge_endpoint_assets(edge_ids)
        if not asset_ids:
            return
        # reindex 는 PG 읽기(이웃·투영)에 db.transaction() 을 쓴다 — _run_in_db 관례대로 새 풀 1개.
        from src.database.postgres_util import PostgresUtil

        db = PostgresUtil()
        with db:
            stats = reindex_asset_topics(db, asset_ids=asset_ids)
        _LOG.info("topic reindex(검토 결정): %s (edges=%d)", stats, len(edge_ids))
    except Exception as exc:  # noqa: BLE001 — 재색인 실패가 결정·응답을 깨지 않는다(FR-304)
        _LOG.warning("topic reindex(검토 결정) 실패(무시): %s", exc)


def _bulk_decide(action: str, edge_ids: list[str], reviewer: str) -> dict[str, Any]:
    """일괄 승인/반려 공통 — 결정+감사를 한 write 트랜잭션에서 수행(FR-203/502).

    빈 목록은 400(의미 없는 요청·오작동 방지). ``bulk_review`` per-id 결과를 받아
    ``ok=True`` 건만 ``relation.{action}`` 감사를 같은 트랜잭션에 남긴다(ok=False 는 미기록).
    """
    if not edge_ids:
        raise HTTPException(status_code=400, detail="edge_ids 는 1개 이상이어야 함")

    def _work(conn: Any) -> dict[str, Any]:
        results = bulk_review(conn, edge_ids=edge_ids, reviewer=reviewer, action=action)
        for r in results:
            if r["ok"]:
                _record_relation_audit(
                    conn, action=f"relation.{action}", reviewer=reviewer,
                    detail={"edge_id": r["edge_id"]})
        return {"results": results}

    out = _run_in_db_write(_work)
    # 커밋 후 재색인(056 FR-301·트랜잭션 밖·best-effort) — ok=True 엣지 양끝 자산 OS topics 갱신.
    _reindex_review_topics(out["results"])
    return out


@app.post("/admin/relations/approve")
def relations_approve(
    body: RelationDecisionRequest,
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """proposed 엣지 일괄 승인(→active)·per-id 결과·감사(FR-201/203/502·US2). LLM 0.

    reviewer = ``principal.user_id``(JWT sub·FR-501). 기존 ``approve_edge``(proposed 가드)를
    건별 재사용하므로 이미 결정된 엣지는 ``ok=False`` 로 반환(예외 아님)한다.
    """
    return _bulk_decide("approve", body.edge_ids, principal.user_id)


@app.post("/admin/relations/reject")
def relations_reject(
    body: RelationDecisionRequest,
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """proposed 엣지 일괄 반려(→rejected)·per-id 결과·감사(FR-202/203/502·US2). LLM 0.

    소프트 반려(행 보존·status 전이만) — 이후 LLM 재제안이 status 를 덮지 않아 rejected 가
    보존된다(``reject_edge`` 계약). reviewer = ``principal.user_id``.
    """
    return _bulk_decide("reject", body.edge_ids, principal.user_id)


@app.post("/admin/relations/revise")
def relations_revise(
    body: RelationReviseRequest,
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """결정 정정(사람 전용·proposed 가드 없음)·감사(FR-301/302/502·US4). LLM 0.

    ``to_status`` 화이트리스트(``_REVIEW_STATUSES``) 위반은 400. active↔rejected·→proposed
    전 방향 전이를 허용해 오결정을 되돌린다(C4). LLM ``sync_graph_edges`` 는 여전히 status
    미갱신이라 사람↔LLM 경계는 보존(FR-302). reviewer = ``principal.user_id``.
    """
    if body.to_status not in _REVIEW_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"알 수 없는 to_status: {body.to_status!r} (허용: {list(_REVIEW_STATUSES)})",
        )
    reviewer = principal.user_id

    def _work(conn: Any) -> dict[str, Any]:
        ok = revise_edge(conn, edge_id=body.edge_id, reviewer=reviewer, to_status=body.to_status)
        if ok:
            _record_relation_audit(
                conn, action="relation.revise", reviewer=reviewer,
                detail={"edge_id": body.edge_id, "to_status": body.to_status})
        # 055 FR-201: approve/reject 와 동일 봉투 {results:[{edge_id,ok}]} 로 통일(단건도 배열).
        return {"results": [{"edge_id": body.edge_id, "ok": ok}]}

    out = _run_in_db_write(_work)
    # 커밋 후 재색인(056 FR-301·트랜잭션 밖·best-effort) — 정정으로 status 가 바뀌면(ok=True)
    # active↔rejected 전이가 양끝 자산 주제 투영을 바꾸므로 그 자산 OS topics 를 갱신한다(SC-02).
    _reindex_review_topics(out["results"])
    return out


@app.post("/admin/relation-kinds/{kind_code}/promote")
def relation_kind_promote(
    kind_code: str,
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """inactive relation_kind 를 active 로 승격(어휘 거버넌스)·감사(FR-401/502·US5). LLM 0.

    기존 ``promote_relation_kind``(inactive 가드·멱등) 재사용 — 이미 active 면 ``ok=False``.
    reviewer(``principal.user_id``)는 감사에만 남는다(relation_kind 에 reviewed_by 컬럼 없음).
    """
    reviewer = principal.user_id

    def _work(conn: Any) -> dict[str, Any]:
        ok = promote_relation_kind(conn, kind_code=kind_code, reviewer=reviewer)
        if ok:
            _record_relation_audit(
                conn, action="relation.kind_promote", reviewer=reviewer,
                detail={"kind_code": kind_code})
        return {"kind_code": kind_code, "ok": ok}

    return _run_in_db_write(_work)


# relation_kind status 화이트리스트(필터 드롭다운 GET·FR-801). 관계 어휘 두 상태만 노출한다.
_RELATION_KIND_STATUSES = ("active", "inactive")


@app.get("/admin/relation-kinds")
def relation_kinds_list(
    status: str | None = Query(None, description="관계종류 상태: active | inactive(생략=전체)"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """관계종류 목록을 조회한다(FR-801·필터 드롭다운용·조회 전용·LLM 0).

    ``{rows:[{kind_code, kind_name_ko, status}], total}`` 를 kind_code 오름차순(결정적)으로
    반환한다. ``status`` 화이트리스트(``_RELATION_KIND_STATUSES``) 위반은 400. relation_kind
    테이블 재사용(마이그레이션 0). RBAC = ``require_principal``(현 2-tier MVP·인증된 누구나).
    """
    if status is not None and status not in _RELATION_KIND_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"알 수 없는 status: {status!r} (허용: {list(_RELATION_KIND_STATUSES)})",
        )
    return _run_in_db(lambda conn: list_relation_kinds(conn, status=status))


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


def _search_topic_facet(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """검색 결과(의료 배제·top-N)의 자산들이 공유하는 주제 패싯을 집계한다(056 FR-503·US3).

    각 결과 행에 이미 실린 **색인 topics/subtopics**(os_hit_to_row → search_group._shape 통과)로
    ``topic_ko`` 별 distinct 결과-자산 수와, 그 아래 ``subtopic_ko`` 별 결과-자산 수(nested)를 센다.
    topic 필터(``?topic=``/``?subtopic=`` → ``terms{topics}``/``terms{subtopics}``)와 **동일 소스**라
    패싯이 약속한 (하위)주제를 클릭했을 때 결과가 일치한다 — **subtopic 카운트도 결과-스코프**다
    (코퍼스 `/topics` 카탈로그 전체 수가 아님). 라이브 투영·자산당 DB 호출(N+1) 없이 행 값만 집계.
    프론트는 이 패싯을 클릭해 로드된 결과를 같은 topics/subtopics 로 **클라 필터(재검색 없음)** →
    컷오프 무관·**패싯 수 == 표시 수**. 반환 형태:
    ``[{topic_ko, asset_count, subtopics:[{subtopic_ko, asset_count}]}]``.

    ⚠️ OS 색인 topics/subtopics 는 flat 리스트라 topic↔subtopic 정확 pairing이 없다 — nested 집계는
    "한 결과 자산이 topic T 와 subtopic S 를 **함께** 보유" 기준(근사)이다. 단 이는 클릭 시 클라 좁히기
    (topic AND subtopic 필터)와 정확히 동일하므로 카운트=표시 일관은 보장된다(정밀 pairing은 색인에
    (topic,subtopic) 쌍 저장 필요·별도 작업).
    **신규 LLM 0·조회 0**·결정적 정렬(asset_count desc → topic_ko/subtopic_ko asc·헌법 3조).
    """
    topic_assets: dict[str, set[str]] = {}
    topic_subs: dict[str, dict[str, set[str]]] = {}  # topic_ko → {subtopic_ko → {asset_id}}
    for rows in grouped.values():
        for r in rows:
            aid = str(r.get("asset_id") or "")
            if not aid:
                continue
            tks = [str(t) for t in (r.get("topics") or []) if t]
            sks = [str(s) for s in (r.get("subtopics") or []) if s]
            for tk in tks:
                topic_assets.setdefault(tk, set()).add(aid)
                sub_map = topic_subs.setdefault(tk, {})
                for sk in sks:  # 근사 pairing: 같은 행의 subtopic 을 이 topic 아래로 귀속
                    sub_map.setdefault(sk, set()).add(aid)
    facet = []
    for tk, assets in topic_assets.items():
        subs = [
            {"subtopic_ko": sk, "asset_count": len(a)}
            for sk, a in topic_subs.get(tk, {}).items()
        ]
        subs.sort(key=lambda s: (-s["asset_count"], s["subtopic_ko"]))
        facet.append({"topic_ko": tk, "asset_count": len(assets), "subtopics": subs})
    # 결정성(헌법 3조): 자산 수 내림차순 → topic_ko 오름차순.
    facet.sort(key=lambda f: (-f["asset_count"], f["topic_ko"]))
    return facet


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
    topic: str | None = Query(None, description="주제(topic) 정확 일치 필터(056·keyword terms)"),
    subtopic: str | None = Query(None, description="세부주제(subtopic) 정확 일치 필터(056·keyword terms)"),
    must_include: list[str] | None = Query(
        None, description="반드시 포함 텀(반복 가능·BM25 must·전체 코퍼스 기준·057 FR-202)"
    ),
    must_exclude: list[str] | None = Query(
        None, description="반드시 제외 텀(반복 가능·BM25 must_not·전체 코퍼스 기준·057 FR-202)"
    ),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """006 하이브리드 검색을 **모달리티별 그룹**으로 반환한다(FR-001/002/003 + 056 FR-503).

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
            topic=topic,
            subtopic=subtopic,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"필터 파라미터 형식 오류: {exc}") from exc

    # 057 FR-202: 반복 쿼리 파라미터(must_include/must_exclude)를 strip·빈문자열 제거·순서 보존으로
    # 정규화한다(미지정 None → []). 빈 리스트면 OS 본문이 바이트 동일(하위호환·회귀 0).
    inc_terms = [s for s in ((t or "").strip() for t in (must_include or [])) if s]
    exc_terms = [s for s in ((t or "").strip() for t in (must_exclude or [])) if s]

    # FR-013: 검색은 006 seam 만 호출(신규 LLM 호출 추가 없음). min_scores 로 모달리티별 적합도
    # 하한을 적용해 약한 후보를 거른다(settings 의 SEARCH_MIN_SCORE_*; 미초기화면 None=필터 비활성).
    result = search_hybrid(
        q,
        modalities=mods,
        limit_per_bucket=_SEARCH_LIMIT_PER_BUCKET,
        min_scores=_search_min_scores(),
        search_mode=search_mode,
        search_filters=search_filters,
        # 057 FR-202: 서버 lexical 필터(전체 코퍼스 must/must_not) — 프론트 페이지-only 필터 대체.
        must_include=inc_terms,
        must_exclude=exc_terms,
    )

    # FR-014: 버킷별 의료 배제 + 모달리티별 독립 랭킹·top-N. results 는 {modality: [rows]}.
    grouped_raw = group_ranked(result, limit_per_modality=size, exclude_domains=_EXCLUDE_DOMAINS)

    # tier projection(042)과 주제 패싯(056 FR-503)을 **같은 읽기 트랜잭션**에서 계산한다(풀 1회).
    def _project_and_facet(conn: Any) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
        projected = _project_grouped_search(conn, grouped_raw, clearance=principal.clearance)
        facet = _search_topic_facet(projected)
        return projected, facet

    grouped, topic_facets = _run_in_db(_project_and_facet)
    counts = {modality: len(rows) for modality, rows in grouped.items()}

    meta: dict[str, Any] = {
        "query": q,
        "modalities": mods,
        "size": size,
        "counts": counts,
        # 056 FR-503(US3): 결과-스코프 주제 패싯 집계(topic_ko별 결과-자산 수). 주제 클릭 → topic= 필터.
        "topic_facets": topic_facets,
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
            "topic": search_filters.topic,
            "subtopic": search_filters.subtopic,
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
    """자산 1건 상세(메타·임베딩 채널 요약·관계 미니뷰)를 반환한다(FR-004/005/006 + 056 FR-501).

    노출 게이트(FR-014)는 ``fetch_asset_detail`` 이 책임진다 — 없음/비registered/의료면 None →
    404(자산 데이터 노출 0).

    056 FR-501 — 노출 통과 자산에 관계 주제 렌즈를 함께 싣는다(신규 LLM 0·FR-505):
    - ``topics``: 이 자산의 active 관계 주제 투영(``project_asset_topics``).
    - ``same_topic_groups``: 같은 주제를 공유하는 다른 자산을 **공유 주제(topic_ko)별로 묶은** 그룹
      (``find_topic_neighbor_groups``·직접 관계 여부 ``already_linked`` 포함) — ego-network(``relations``)
      옆의 두 번째 탐색 렌즈(US1·의료 제외). 057-후속: 평면 목록의 ``overlap_weight`` 오라벨 혼선을
      "무슨 주제로 같은지" 그룹 구조로 대체.
    상세 조회와 **같은 읽기 트랜잭션**에서 계산한다(추가 풀 획득 없음). 게이트 미통과(None)면
    주제 seam 을 호출하지 않는다(불필요한 조회 없음).
    """

    def _work(conn: Any) -> dict[str, Any] | None:
        detail = fetch_asset_detail(conn, asset_id=asset_id, clearance=principal.clearance)
        if detail is None:
            return None
        detail["topics"] = project_asset_topics(conn, asset_id=asset_id)
        detail["same_topic_groups"] = find_topic_neighbor_groups(conn, asset_id=asset_id)
        return detail

    detail = _run_in_db(_work)
    if detail is None:
        raise HTTPException(status_code=404, detail="자산을 찾을 수 없거나 노출 대상이 아님")
    return detail


@app.get("/topics")
def topics_list(
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """관계 주제 목록(topic→subtopic 2단계·주제별 자산 수)을 반환한다(056 FR-502·US2). 조회 전용·LLM 0.

    ``list_topics`` 가 active·의료 제외 엣지의 ``(topic_ko, subtopic_ko)`` 별 distinct 자산 수를
    결정적 정렬(topic_ko asc→subtopic_ko asc)로 집계한다. 주제 브라우즈 진입점.

    057 FR-105 — 각 행에 ``topic_asset_count``(주제 전체 distinct 자산 수)를 함께 싣는다(하위호환
    필드 추가). 프론트는 하위주제 ``asset_count`` 를 합산(중복카운트)하지 않고 이 값을 그대로 쓴다.
    """
    return {"topics": _run_in_db(list_topics)}


@app.get("/topics/{topic}")
def topic_assets(
    topic: str,
    subtopic: str | None = Query(None, description="세부주제(주면 topic 하위로 좁힘·정확 일치)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> dict[str, Any]:
    """특정 주제에 속한 자산을 페이징 조회한다(056 FR-502·US2). 조회 전용·의료 제외·LLM 0.

    ``assets_in_topic`` 이 그 주제(active 엣지) 양끝 자산을 distinct·``asset_id asc`` 결정적 정렬로
    페이징한다. ``subtopic`` 미지정이면 topic 하위 전체. 반환 ``{rows:[{asset_id, fs_uri, file_name}], total}``.
    """
    return _run_in_db(
        lambda conn: assets_in_topic(
            conn, topic_ko=topic, subtopic_ko=subtopic, limit=limit, offset=offset
        )
    )


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


@app.get("/assets/{asset_id}/thumbnail")
def asset_thumbnail(
    asset_id: str,
    size: str = Query("card", description="크기 프리셋: card(320·목록/hover 기본) | detail(640·상세 히어로)"),
    principal: Annotated[Principal, Depends(require_principal)] = ...,
) -> Response:
    """이미지·영상 자산의 축소 썸네일(JPEG)을 반환한다(057-후속·멀티모달 시각 미리보기). 조회 전용.

    절차
        1. ``resolve_download_target`` 노출 게이트(registered·**비의료**·FR-014) 통과 확인 → None → 404
           (의료 자산 썸네일=PHI 이므로 여기서 원천 차단).
        2. 이미지·영상이 아니면 404(오디오/텍스트/unknown 은 시각 표현 없음 → 프론트 아이콘 폴백).
        3. 원본 파일 부재/접근 불가 → 410(FR-009). 생성 실패(손상·코덱) → 404.
    ``size`` 는 용도별 크기 프리셋(card 320·detail 640) — 미지원 값은 card 로 폴백(썸네일은 장식적이라
    타이포에 404 대신 기본 서빙). ``cached_thumbnail`` 은 **디스크 캐시 경유(generate-once·크기별)** —
    첫 요청만 원본을 읽어 생성·저장하고 이후는 캐시 파일만 서빙(영상 재디코드 0). 원본 무수정·결정적·LLM 0.
    브라우저 캐시(``Cache-Control``)는 같은 사용자 반복을, 디스크 캐시는 전 사용자 재생성을 막는다. 캐시는
    파생물(``THUMBNAIL_CACHE_DIR``)이라 삭제·재생성 안전. 읽기 전용·마이그레이션 0·자체 완결.
    """
    target = _run_in_db(lambda conn: resolve_download_target(conn, asset_id=asset_id))
    if target is None:
        raise HTTPException(status_code=404, detail="썸네일 대상을 찾을 수 없거나 노출 대상이 아님")
    modality = target.get("modality")
    if modality not in THUMBNAILABLE_MODALITIES:
        raise HTTPException(status_code=404, detail="썸네일을 제공하지 않는 자산 유형")
    fs_path = target.get("fs_path")
    if not fs_path or not os.path.isfile(fs_path):
        raise HTTPException(status_code=410, detail="원본 파일이 존재하지 않거나 접근할 수 없음")
    data = cached_thumbnail(asset_id, fs_path, modality, size=size)  # 디스크 캐시 경유(크기별 generate-once)
    if data is None:
        raise HTTPException(status_code=404, detail="썸네일을 생성할 수 없음")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
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
