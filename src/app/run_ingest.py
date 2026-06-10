"""F-1.x 수집→라우팅→(분류)→추출→등록 오케스트레이터 (모델 A).

로컬 파일 리스트를 순회하며 각 파일을:
    1. route_file 로 modality 판정(없는 파일은 skip).
    2. create_asset 로 ``received`` 조기 INSERT(별도 트랜잭션, 커밋).
    3. set_status 로 routing→classifying→extracting (단계별 짧은 트랜잭션).
    4. dispatch_extract 로 추출(트랜잭션 밖에서 실행 — LLM/IO).
    5. finalize_asset 로 메타·임베딩 적재 + ``registered`` (한 트랜잭션).
    6. 어디서든 예외 → **fresh 트랜잭션**으로 mark_failed (이전 트랜잭션이 abort 돼도 안전). 다음 파일 계속.

디스패처 단일 권위: 미지원 modality 는 사전 차단하지 않고 dispatch_extract 의
UnsupportedModalityError 를 6번에서 흡수한다(asset.status='failed').

``classify_fn``/``extract_fn`` 은 **테스트·e2e 전용 override** 주입점이다(운영 호출부 없음).
운영에서는 미주입으로 두어 레지스트리 기본 경로(분류 cascade_v1 + 도메인 팩 extract/embed)를 타고,
주입 시 무거운 실경로(LLM/CLIP/STT)를 결정적 stub 으로 대체한다.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

from src.classify.types import ClassificationResult
from src.config.settings import get_current_settings
from src.database.postgres_util import PostgresUtil
from src.dispatch.types import AssetRecord, ExtractContext
from src.file.hashing import file_hash_and_size
from src.ingest.router import REASON_MISSING, route_file
from src.ingest.status import AssetStatus, InvalidTransitionError, mark_failed, set_status
from src.pipeline import builtins as _builtins  # noqa: F401 — DEFAULT_REGISTRY 등록 부수효과
from src.pipeline.packs import for_domain
from src.pipeline.policy import validate as policy_validate
from src.pipeline.registry import DEFAULT_REGISTRY
from src.registry.asset_persist import create_asset, finalize_asset, find_registered_asset_by_hash
from src.registry.classification_persist import record_classification
from src.registry.lineage_persist import record_lineage
from src.registry.schema_registry import validate_ext_meta

REASON_DUPLICATE = "duplicate"

_LOG = logging.getLogger("meta_extract.run_ingest")

ExtractFn = Callable[[ExtractContext], AssetRecord]
ClassifyFn = Callable[[str, str], ClassificationResult]  # (file_path, modality) -> 분류 결과


def _configure_logging() -> None:
    if _LOG.handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


def _make_opensearch_indexer(*, db: PostgresUtil, settings: Any) -> Callable[[Any], None]:
    """ingest 배치용 best-effort 증분 색인기를 만든다(spec 020, FR-002 · 클라이언트 1회 재사용).

    반환한 콜러블 ``index(asset_id)`` 를 자산 finalize(PG 커밋) **직후**마다 부른다.

    안전 게이트(프로덕션 적재 경로 — 회귀 0·격리):
      · **opt-in off(기본)이면 콜러블이 즉시 반환** — ``opensearch_sync_enabled`` 미설정/False 면 아무
        것도 하지 않고, ``src.search.opensearch_sync``·opensearch-py 를 **import 조차 하지 않는다**(아래
        지연 import). 미도입 환경의 run_ingest 동작이 완전 불변(SC-001). 레거시 settings 에 필드가
        없어도 ``getattr`` 폴백으로 off 취급.
      · **클라이언트 재사용**: OpenSearch 클라이언트·활성 채널은 **첫 색인 성공에서 1회** 만들어
        ``cache`` 에 담아 배치 전체에서 재사용한다 — 디렉터리/파일리스트 수집에서 자산마다 새 연결을
        열던 낭비를 없앤다. "배치당 1회"는 **성공 경로 보증**이다: OS 가 **지속 다운**이면 ``get_client``
        실패로 ``cache['client']`` 가 미설정이라 자산마다 셋업을 재시도한다(상한 = 배치 크기, 무한 아님).
        이는 OS 가 배치 도중 복구되면 그때부터 재사용을 재개하는 **일과성 복구**가 의도다. 부분 캐시
        손상은 없다(게이트는 ``'client'`` 단일 키 기준, ``channel``/``index_asset`` 은 멱등 재대입).
      · **격리**: 각 색인을 ``try/except Exception`` 으로 감싸 OS 미도달·색인 오류를 ``_LOG.warning``
        만 남기고 삼킨다 — ingest 를 중단·롤백하지 않는다(SC-003). finalize 트랜잭션 커밋 뒤라 OS
        색인은 본질적으로 PG 와 분리된 best-effort 작업이다.
    """
    enabled = getattr(settings, "opensearch_sync_enabled", False)
    cache: dict[str, Any] = {}  # 첫 성공 셋업 후 client·index_asset·channel 을 담아 배치 내 재사용

    def index(asset_id: Any) -> None:
        if not enabled:
            return  # off(기본) — OpenSearch 코드 미접촉, 기존 동작 불변
        try:
            if "client" not in cache:
                # 지연 import — 플래그 off 환경(opensearch-py 미설치 가능)의 순수성을 보존한다.
                from src.config.settings import active_embed_channel
                from src.search.opensearch_sync import get_client, index_asset

                cache["channel"] = active_embed_channel(settings)  # 적재·검색과 같은 채널(018)
                cache["client"] = get_client(settings.opensearch_url)  # 배치당 1회 생성·재사용
                cache["index_asset"] = index_asset
            # PG 는 읽기전용(SELECT만, FR-004) → OpenSearch 에만 쓰기(CQRS). finalize 와 별도 트랜잭션.
            with db.transaction() as conn:
                cache["index_asset"](
                    cache["client"], conn, str(asset_id),
                    index=settings.opensearch_index, channel=cache["channel"],
                )
        except Exception as exc:  # noqa: BLE001 — OS 색인 실패가 적재를 막지 않는다(best-effort 격리)
            _LOG.warning("opensearch 증분 색인 실패(무시): asset_id=%s (%s)", asset_id, exc)

    return index


def run_ingest(
    files: list[str],
    *,
    db: PostgresUtil,
    extract_fn: ExtractFn | None = None,   # 테스트·e2e 전용 override(미주입=팩 기본 extract/embed)
    classify_fn: ClassifyFn | None = None,  # 테스트·e2e 전용 override(미주입=cascade_v1)
    registry=DEFAULT_REGISTRY,
    settings: Any = None,
) -> dict[str, list[Any]]:
    """파일 리스트를 적재. 반환: {'registered': [asset_id], 'failed': [(asset_id,err)], 'skipped': [(path,reason)]}."""
    _configure_logging()
    cfg = settings or get_current_settings()
    result: dict[str, list[Any]] = {"registered": [], "failed": [], "skipped": [], "deferred": []}
    # OpenSearch 증분 색인기 — 배치당 1회 생성(opt-in off 면 no-op·미import). 클라이언트는 첫 색인에서
    # 만들어 배치 전체 재사용(자산마다 새 연결 X). 자산 finalize 직후 os_index(asset_id) 로 호출.
    os_index = _make_opensearch_indexer(db=db, settings=cfg)

    for path in files:
        # 파일 단위 격리: 한 파일의 어떤 실패도 배치를 멈추지 않는다.
        asset_id: uuid.UUID | None = None
        try:
            route = route_file(path)
            if not route.routable and route.reason == REASON_MISSING:
                _LOG.info("skip(missing): %s", path)
                result["skipped"].append((path, route.reason))
                continue

            # 1.5) 내용 해시 기반 중복 방지: 동일 file_hash 가 이미 적재(registered/deferred)면 skip.
            # 009(#4): dedup 범위가 deferred 까지 확장돼, DICOM 등 보류 자산을 재수집해도
            # 새 deferred 행을 만들지 않고 기존 자산을 가리키며 skip 한다(중복 0). failed 는
            # 여기서 dup=None 으로 떨어져 재처리된다(find_registered_asset_by_hash 가 IN 범위로 제어).
            file_hash, file_size = file_hash_and_size(path)
            with db.transaction() as conn:
                dup = find_registered_asset_by_hash(conn, file_hash)
            if dup is not None:
                _LOG.info("skip(duplicate of %s): %s", dup, path)
                result["skipped"].append((path, f"{REASON_DUPLICATE}:{dup}"))
                continue

            # 2) 조기 INSERT (received) — 자체 트랜잭션
            with db.transaction() as conn:
                asset_id = create_asset(
                    conn,
                    fs_path=path,
                    modality=route.modality,
                    domain=route.domain,
                    file_hash=file_hash,
                    file_size=file_size,
                )
                record_lineage(conn, asset_id, activity="ingest.received.v1", agent="run_ingest",
                               generated={"modality": route.modality}, payload={"fs_path": path})

            # 3) routing → classifying (한 트랜잭션 묶음 커밋)
            # status 는 '단계 진입 전'에 찍는 진행형 마커다(완료는 다음 전이로 암시). 그래서:
            #  - routing: 실작업(route_file, 위)이 row 생성 전 끝나 사후 마커이고, classifying 과 같은
            #    트랜잭션이라 단독 관측되지 않는다 — received→classifying 사이를 FSM 순차성상 거쳐갈 뿐.
            #  - classifying: 바로 아래 실분류(cascade) 진입 직전을 정확히 반영.
            with db.transaction() as conn:
                set_status(conn, asset_id, AssetStatus.ROUTING)
                record_lineage(conn, asset_id, activity="ingest.routing.v1", agent="run_ingest")
                set_status(conn, asset_id, AssetStatus.CLASSIFYING)
                record_lineage(conn, asset_id, activity="ingest.classifying.v1", agent="run_ingest")

            # 3.5) 도메인 분류: override 우선, 없으면 레지스트리 기본 분류기(ctx 기반)
            ctx = ExtractContext(
                file_path=path, modality=route.modality, domain=route.domain, settings=cfg, db=db
            )
            if classify_fn is not None:
                classification = classify_fn(path, route.modality)
            else:
                classification = registry.resolve("classify", "cascade_v1")(ctx)
            with db.transaction() as conn:
                record_classification(conn, asset_id, classification)
                record_lineage(conn, asset_id, activity="ingest.classified.v1",
                               agent=("classify_fn" if classify_fn is not None else "cascade_v1"),
                               generated={"final_label": classification.final_label,
                                          "decided_stage": classification.decided_stage,
                                          "confidence": classification.confidence})
            domain = classification.final_label
            ctx.domain = domain

            # 시그니처로 확정된 포맷(stage1)이지만 해당 어댑터가 없으면 보류(deferred) — 실패 아닌 계획적 대기.
            # 현재 시그니처를 등록한 프로파일이 medical(DICOM/HL7/FHIR)뿐이라 사실상 의료 포맷만 보류되지만,
            # 코드 자체는 도메인-불가지(메커니즘은 범용; 어느 도메인이든 추출기 없는 시그니처 포맷이면 보류).
            # cascade v2 이후 stage1_scores 는 {domain: {signature, ...}} 중첩 구조.
            signature = None
            if classification.decided_stage == 1:
                signature = classification.stage1_scores.get(domain, {}).get("signature")
            if signature:
                with db.transaction() as conn:
                    set_status(conn, asset_id, AssetStatus.DEFERRED, reason=f"{domain}_format:{signature}")
                    record_lineage(conn, asset_id, activity="ingest.deferred.v1", agent="run_ingest",
                                   payload={"domain": domain, "signature": signature})
                _LOG.info("deferred(%s/%s): asset_id=%s %s", domain, signature, asset_id, path)
                result["deferred"].append(asset_id)
                continue

            # 도메인 팩 선택 + 정책 검증(컴포지션 시점)
            pack = for_domain(domain)
            policy_validate(pack, registry)

            with db.transaction() as conn:
                set_status(conn, asset_id, AssetStatus.EXTRACTING)
                record_lineage(conn, asset_id, activity="ingest.extracting.v1", agent="run_ingest")

            # 4) 추출/임베딩 — override(full record) 또는 팩 경로(extract_meta + embed)
            if extract_fn is not None:
                record = extract_fn(ctx)
            else:
                extract = registry.resolve("extract", pack.per_asset["extract"])
                embed = registry.resolve("embed", pack.per_asset["embed"])
                record = extract(ctx)
                record.embeddings = embed(ctx, record)

            # 5) 검증 + 적재 + registered — 한 트랜잭션
            with db.transaction() as conn:
                validate_ext_meta(conn, domain, record.ext_meta)
                finalize_asset(conn, asset_id, record)
                record_lineage(conn, asset_id, activity="ingest.registered.v1",
                               agent=("extract_fn" if extract_fn is not None else pack.per_asset["extract"]),
                               generated={"channels": sorted({e.channel for e in record.embeddings}),
                                          "n_embeddings": len(record.embeddings),
                                          "models": sorted({e.model_name for e in record.embeddings})})

            _LOG.info("registered: asset_id=%s %s", asset_id, path)
            result["registered"].append(asset_id)

            # 증분 색인(opt-in·격리) — 위 finalize 트랜잭션이 **PG 커밋된 직후** 호출한다(FR-002).
            # off(기본)면 즉시 반환해 OpenSearch 코드를 전혀 건드리지 않으므로 기존 동작 불변(SC-001).
            # 색인기 내부 try/except 로 OS 실패를 삼켜 적재를 중단·롤백하지 않는다(SC-003).
            os_index(asset_id)
        except Exception as exc:  # noqa: BLE001 — route/create/추출/적재 모든 실패 흡수
            reason = f"{type(exc).__name__}: {exc}"
            _LOG.warning("failed: asset_id=%s %s (%s)", asset_id, path, reason)
            if asset_id is None:
                # asset 생성 전(route_file/create_asset) 실패 → 경로 기준 기록
                result["failed"].append((None, f"{path}: {reason}"))
            else:
                # 6) fresh 트랜잭션으로 실패 기록(이전 트랜잭션 abort 가능성 격리)
                with db.transaction() as conn:
                    try:
                        mark_failed(conn, asset_id, reason)
                        record_lineage(conn, asset_id, activity="ingest.failed.v1", agent="run_ingest",
                                       payload={"reason": reason})
                    except InvalidTransitionError:
                        # 이미 종료 상태면 무시. 009: ConcurrentTransitionError(동시 전이 충돌 —
                        # 다른 워커가 먼저 종료시킨 경우)도 이 계열의 하위라 같은 경로로 흡수돼
                        # 배치가 멈추지 않는다(호출부 시그니처 무변경).
                        pass
                result["failed"].append((asset_id, reason))

    _LOG.info(
        "ingest done: registered=%s failed=%s skipped=%s deferred=%s",
        len(result["registered"]),
        len(result["failed"]),
        len(result["skipped"]),
        len(result["deferred"]),
    )
    return result


# ── 초기 설정(부트스트랩) 절차 ────────────────────────────────────────────────
# [import 시점·자동·호출순서 무관] 이 모듈을 import 하는 것만으로 완료되는 등록(부수효과):
#   · 전략 레지스트리 DEFAULT_REGISTRY ← 위의 `from src.pipeline import builtins` 가
#     register_defaults 를 실행 → classify/cascade_v1·extract·embed/by_modality·persist + cross-asset 슬롯.
#   · 도메인 프로파일 DOMAIN_PROFILES ← builtins→cascade→`src.classify.domains`→medical 체인 →
#     {"medical"}(DICOM/HL7/FHIR 시그니처+어휘). general 은 미등록 — cascade 엔진 내장 폴백.
#   (run_relations/run_relations_review 는 이 import 가 없어 레지스트리·프로파일이 비어 있다.)
# [런타임·main() 안·순서 중요] 아래 3단계는 반드시 이 순서로:
#   1) load_dotenv(.env.{env}, override=False) — OS 기존 환경변수 우선.
#   2) init_settings(env) — 필수 환경변수 검증 후 frozen PipelineSettings 생성(이후 get_current_settings 활성).
#   3) PostgresUtil() + `with db:` — 연결 풀 생성 + PG17 검증.
#   로깅(_configure_logging)·LLM 클라이언트(get_llm_client)는 모듈 싱글턴 없이 첫 사용 시 지연 초기화.
def main() -> int:
    """CLI: 로컬 파일 수집 후 asset_* 적재. 예) python -m src.app.run_ingest --env dev --input-dir DIR"""
    import argparse
    import json
    from pathlib import Path

    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.ingest.collector import collect_files

    parser = argparse.ArgumentParser(description="수집→라우팅→추출→등록 (asset_* 적재)")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--input-dir", dest="input_dir", default=None)
    parser.add_argument("--file-list", dest="file_list", default=None)
    parser.add_argument("paths", nargs="*", metavar="FILE")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    files = collect_files(
        input_dir=args.input_dir,
        file_list=args.file_list,
        paths=args.paths or None,
    )

    db = PostgresUtil()
    with db:
        result = run_ingest(files, db=db)
    print(json.dumps({k: len(v) for k, v in result.items()}, ensure_ascii=False))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
