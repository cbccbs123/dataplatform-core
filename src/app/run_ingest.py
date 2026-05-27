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

분류(F-5.1)는 ``classify_fn`` 으로 주입(미구현 시 route 의 domain='general' 유지).
추출은 ``extract_fn`` 으로 주입 가능(기본 dispatch_extract; 테스트에서 대체).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

from src.classify.types import ClassificationResult
from src.config.settings import get_current_settings
from src.database.postgres_util import PostgresUtil
from src.dispatch.types import AssetRecord, ExtractContext
from src.file.hashing import file_hash_and_size
from src.ingest.router import REASON_MISSING, route_file
from src.ingest.status import AssetStatus, InvalidTransitionError, mark_failed, set_status
from src.registry.asset_persist import create_asset, finalize_asset, find_registered_asset_by_hash
from src.registry.classification_persist import record_classification
from src.registry.lineage_persist import record_lineage
from src.registry.schema_registry import validate_ext_meta
from src.pipeline import builtins as _builtins  # noqa: F401 — DEFAULT_REGISTRY 등록 부수효과
from src.pipeline.packs import for_domain
from src.pipeline.policy import validate as policy_validate
from src.pipeline.registry import DEFAULT_REGISTRY

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


def run_ingest(
    files: list[str],
    *,
    db: PostgresUtil,
    extract_fn: ExtractFn | None = None,
    classify_fn: ClassifyFn | None = None,
    registry=DEFAULT_REGISTRY,
    settings: Any = None,
) -> dict[str, list[Any]]:
    """파일 리스트를 적재. 반환: {'registered': [asset_id], 'failed': [(asset_id,err)], 'skipped': [(path,reason)]}."""
    _configure_logging()
    cfg = settings or get_current_settings()
    result: dict[str, list[Any]] = {"registered": [], "failed": [], "skipped": [], "deferred": []}

    for path in files:
        # 파일 단위 격리: 한 파일의 어떤 실패도 배치를 멈추지 않는다.
        asset_id: uuid.UUID | None = None
        try:
            route = route_file(path)
            if not route.routable and route.reason == REASON_MISSING:
                _LOG.info("skip(missing): %s", path)
                result["skipped"].append((path, route.reason))
                continue

            # 1.5) 내용 해시 기반 중복 방지: 동일 file_hash 가 이미 registered 면 skip
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

            # 3) routing → classifying (단계별 커밋)
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

            # 의료 표준 포맷(DICOM/HL7/FHIR, stage1 시그니처)은 일반 추출 대상이 아님 — 보류(deferred)
            # cascade v2 이후 stage1_scores 는 {domain: {signature, ...}} 중첩 구조
            signature = None
            if classification.decided_stage == 1:
                signature = classification.stage1_scores.get(domain, {}).get("signature")
            if signature:
                with db.transaction() as conn:
                    set_status(conn, asset_id, AssetStatus.DEFERRED, reason=f"medical_format:{signature}")
                    record_lineage(conn, asset_id, activity="ingest.deferred.v1", agent="run_ingest",
                                   payload={"signature": signature})
                _LOG.info("deferred(medical %s): asset_id=%s %s", signature, asset_id, path)
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
                        pass  # 이미 종료 상태면 무시
                result["failed"].append((asset_id, reason))

    _LOG.info(
        "ingest done: registered=%s failed=%s skipped=%s deferred=%s",
        len(result["registered"]),
        len(result["failed"]),
        len(result["skipped"]),
        len(result["deferred"]),
    )
    return result


def main() -> int:
    """CLI: 로컬 파일 수집 후 asset_* 적재. 예) python -m src.app.run_ingest --env dev --input-dir DIR"""
    import argparse
    import json
    import sys
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
