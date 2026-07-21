#!/usr/bin/env python3
"""테스트 무결성 가드 — 자율 루프가 "테스트를 약화시켜 통과 위조"하는 것을 차단.

자율 개발 루프(feature-builder 서브에이전트)가 테스트를 green 으로 만들 때,
정답을 구현하는 대신 테스트를 지우거나(assert 삭제) skip 으로 무력화하는 함정에 빠질 수 있다.
본 스크립트는 기준 시점(base) 대비 tests/ 의 "검증 강도"가 줄지 않았는지 검사한다.

검사(BASE..현재 작업트리):
  1) [차단] 테스트 함수(def test_*) 총개수 감소
  2) [차단] assert 계열 호출(self.assert*, assert ) 총개수 감소
  3) [차단] @skip/@skipIf/@expectedFailure 총개수 증가(무조건 무력화 패턴)
  → 하나라도 걸리면 exit 1. "테스트를 약하게 만들어 통과"를 거부한다.

  주의: ``@skipUnless`` 는 **조건부 실행 게이트**(실 DB ``RUN_DB_E2E`` e2e·선택 의존성 등)로
  이 코드베이스의 표준 e2e 패턴이라 약화가 아니다 → 증가 검사에서 제외한다(신규 e2e 추가 시 오탐 방지).
  무조건 무력화(@skip/@skipIf/@expectedFailure)와 assert/테스트 함수 감소만 차단한다.

사용:
    python scripts/test_integrity.py            # main 대비
    python scripts/test_integrity.py --base HEAD~1
표준 라이브러리만 사용(의존성 0).
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = "tests/"

# 레포 분리(077)로 **백엔드 레포(dataplatform-service)로 이관**된 테스트 파일들.
# 이건 "약화 위조"(몰래 삭제)가 아니라 정당한 이관(다른 레포에 그대로 존재)이므로,
# base(main)·현재 양쪽 집계에서 함께 제외한다 → 이관분으로 인한 거짓 "감소" 오탐 방지.
# ※ 여기 없는 파일의 삭제·assert 감소는 계속 차단된다(감시원 본래 기능 보존).
_EXTRACTED_TO_BACKEND = frozenset({
    "tests/test_basename_single_source.py", "tests/test_dashboard_summary.py",
    "tests/test_display_file_name.py", "tests/test_ext_expr_single_source.py",
    "tests/test_modality_migration_e2e.py", "tests/test_portal_access_log.py",
    "tests/test_portal_api_dashboard.py", "tests/test_portal_api_history.py",
    "tests/test_portal_api_relations.py", "tests/test_portal_api.py",
    "tests/test_portal_asset_detail.py", "tests/test_portal_asset_stats.py",
    "tests/test_portal_auth_config.py", "tests/test_portal_auth.py",
    "tests/test_portal_db_singleton.py", "tests/test_portal_download.py",
    "tests/test_portal_e2e.py", "tests/test_portal_history_e2e.py",
    "tests/test_portal_lineage_query.py", "tests/test_portal_range.py",
    "tests/test_portal_search_group.py", "tests/test_portal_snapshot_e2e.py",
    "tests/test_portal_topics.py", "tests/test_relations_review_api_e2e.py",
    "tests/test_search_modalities.py", "tests/test_thumbnail.py",
})

# 레포 분리(078·G4)로 **파이프라인 레포(dataplatform-pipeline)로 이관/제거**된 테스트 파일들.
# 코어 물리 분리로 파이프라인 코드가 이 레포를 떠나 그 테스트도 함께 빠진 것 — 약화 위조 아님.
# base(main)·현재 양쪽에서 함께 제외해 거짓 감소 오탐 방지(그 외 삭제는 계속 차단).
# ※ test_asset_topic_classify.py 는 파이프라인(classify write)행이나, 그 안의 코어 read 함수 테스트
#   (fetch_asset_topic·find_same_topic_groups)는 코어 tests/test_asset_topic_query.py 로 **재편입**했다
#   (코어 커버리지 유지·유실 방지). 즉 파일은 이관되되 코어 소속 검증은 코어에 남는다.
_EXTRACTED_TO_PIPELINE = frozenset({
    "tests/test_aboutness.py", "tests/test_archiver.py", "tests/test_archiver_e2e.py",
    "tests/test_asset_persist.py", "tests/test_asset_relations.py", "tests/test_asset_topic_classify.py",
    "tests/test_asset_topic_e2e.py", "tests/test_asset_topic_query_e2e.py", "tests/test_audio_meta_extractor.py",
    "tests/test_backfill_topic_canonical.py",
    "tests/test_batch_runner.py", "tests/test_builtins.py", "tests/test_classification_persist.py",
    "tests/test_classify.py", "tests/test_classify_profiles.py", "tests/test_collector.py",
    "tests/test_content_guard.py", "tests/test_count_tokens_tokenizer.py", "tests/test_cross_runner.py",
    "tests/test_dag_load.py", "tests/test_dedup_deferred.py", "tests/test_dispatcher.py",
    "tests/test_domain_medical.py", "tests/test_evidence_rescue_harness.py", "tests/test_graph_persist.py",
    "tests/test_ingest_split.py", "tests/test_keyframe_dedup.py", "tests/test_keyframe_dedup_defaults.py",
    "tests/test_opensearch_search_e2e.py", "tests/test_packs.py", "tests/test_packs_cross_asset.py",
    "tests/test_pipeline_contracts.py", "tests/test_policy.py", "tests/test_reextract_stage2.py",
    "tests/test_registry.py", "tests/test_router.py", "tests/test_run_ingest.py",
    "tests/test_run_ingest_e2e.py", "tests/test_run_ingest_opensearch_hook.py", "tests/test_run_ingest_packs.py",
    "tests/test_run_opensearch_resync.py", "tests/test_run_relations_retry.py", "tests/test_run_relations_sample_e2e.py",
    "tests/test_run_search.py", "tests/test_sample_pack_slot.py", "tests/test_search_parity_live.py",
    "tests/test_skill_split.py", "tests/test_skills_active_channel.py", "tests/test_status.py",
    "tests/test_status_transition_atomic.py", "tests/test_status_vocab.py", "tests/test_stt.py",
    "tests/test_text_probe.py", "tests/test_topic_canonicalize_e2e.py", "tests/test_topic_grounding_report.py",
    "tests/test_usf_cleanup_dead_files.py", "tests/test_video_keyframes_dedup.py", "tests/test_video_keyframes_fallback.py",
    "tests/test_video_skill_dedup.py", "tests/test_video_skill_keyframe_zero.py",
    "tests/test_backfill_bge.py", "tests/test_backfill_topic_canonical_e2e.py",
    "tests/test_reextract_summaries.py",
    # 077서 scripts 로 개명 이동된 백필 테스트의 **main(개명 전) 이름** — 개명(077 test_backfill_*)→제거(G4)
    # 연쇄로 base(main)에만 존재. 개명 후 이름은 위에 있으나 개명 전 이름도 제외해야 base 과대계상 방지.
    "tests/test_run_about_backfill.py", "tests/test_topic_backfill.py",
})


TESTDEF_RX = re.compile(r"^\s*def\s+test_\w+", re.M)
ASSERT_RX = re.compile(r"\bself\.assert\w+\(|^\s*assert\s", re.M)
# skipUnless 는 조건부 실행 게이트(e2e/선택 의존성)라 약화 아님 → 제외.
# 무조건 무력화 패턴(skip·skipIf·expectedFailure)만 증가 차단 대상.
SKIP_RX = re.compile(r"@(?:unittest\.)?(?:skip|skipIf|expectedFailure)\b")


def _counts(text: str) -> tuple[int, int, int]:
    return (
        len(TESTDEF_RX.findall(text)),
        len(ASSERT_RX.findall(text)),
        len(SKIP_RX.findall(text)),
    )


def _git_show(ref: str, path: str) -> str:
    try:
        return subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return ""  # base 에 없던 파일(신규) → 빈 문자열


def _list_test_files(ref: str | None) -> list[str]:
    if ref is None:  # 현재 작업트리
        return [
            str(p.relative_to(ROOT))
            for p in (ROOT / "tests").glob("test_*.py")
        ]
    out = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, TESTS],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln.endswith(".py") and "/test_" in f"/{ln}"]


def _aggregate(ref: str | None) -> tuple[int, int, int]:
    td = ta = ts = 0
    files = set(_list_test_files(ref)) | set(_list_test_files(None) if ref else [])
    files -= _EXTRACTED_TO_BACKEND | _EXTRACTED_TO_PIPELINE  # 이관 파일은 base·현재 모두에서 제외(거짓 감소 방지)
    for f in files:
        text = (ROOT / f).read_text(encoding="utf-8") if ref is None and (ROOT / f).exists() \
            else _git_show(ref, f) if ref else ""
        a, b, c = _counts(text)
        td += a
        ta += b
        ts += c
    return td, ta, ts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="테스트 무결성 가드")
    ap.add_argument("--base", default="main", help="기준 ref(기본 main)")
    args = ap.parse_args(argv)

    # base 존재 확인
    if subprocess.run(["git", "rev-parse", "--verify", args.base],
                      cwd=ROOT, capture_output=True).returncode != 0:
        print(f"기준 ref '{args.base}' 없음 — 검사 건너뜀(첫 커밋/CI 환경 가능).")
        return 0

    b_td, b_ta, b_ts = _aggregate(args.base)
    c_td, c_ta, c_ts = _aggregate(None)

    print(f"기준({args.base}) → 현재")
    print(f"  test 함수 : {b_td} → {c_td}")
    print(f"  assert    : {b_ta} → {c_ta}")
    print(f"  skip 데코  : {b_ts} → {c_ts}")

    fails = []
    if c_td < b_td:
        fails.append(f"테스트 함수가 {b_td}→{c_td} 로 감소(삭제 의심)")
    if c_ta < b_ta:
        fails.append(f"assert 가 {b_ta}→{c_ta} 로 감소(검증 약화 의심)")
    if c_ts > b_ts:
        fails.append(f"skip 데코가 {b_ts}→{c_ts} 로 증가(테스트 무력화 의심)")

    if fails:
        print("\n🔴 테스트 무결성 위반 — 통과를 위해 테스트를 약화시킨 정황:")
        for f in fails:
            print(f"  - {f}")
        print("\n→ 테스트를 지우지 말고 구현으로 통과시키세요. 의도적 테스트 정리면 사람이 검토 후 진행.")
        return 1
    print("\n✅ 테스트 무결성 유지(약화 없음).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
