"""점수 축 v3 전환(2026-07-31)의 소급 처리 — 재생성 전 리셋 + 재생성 후 잔재 소거.

**왜 그냥 재생성하면 안 되는가** (이 스크립트가 존재하는 이유):

1. `graph_persist` 의 upsert 는 신뢰도를 ``GREATEST(기존, 신규)`` 로 화해한다 — 옛 점수
   (기준 없는 감각치·0.95 등)가 새 점수(v3 · 최고 0.9)보다 커서 **옛 점수가 살아남는다.**
   그대로 재생성하면 새 점수 체계가 기존 행에 한 건도 적용되지 않는다.
2. upsert 는 ``status`` 를 절대 덮지 않는다(사람 결정 보호) — 자동승인 시절의 ``active``
   499건(이름표 정확도 67% 실측)이 "확인됨"으로 계속 노출된다.
3. 이번에 재제안되지 않는 엣지(새 게이트 미달·프롬프트 개선으로 소멸)는 지워지는 경로가
   없다 — upsert 는 삭제를 모른다(specs/081 발견 7).

**3단계 절차** (각 단계는 별도 실행 · 2단계는 파이프라인 레포의 전량 재생성):

  ① ``--phase reset`` : 전량 백업(JSON) 후 ``status='proposed'``, ``confidence=NULL`` 로 리셋.
     confidence=NULL 이면 GREATEST(COALESCE(NULL,0), 신규) = 신규 — 새 점수가 항상 이긴다.
     ⚠️ ``reviewed_by IS NOT NULL`` (사람이 만진) 행은 건드리지 않는다 — 현재 0건 실측이나
     원칙을 코드로 박는다.
  ② (파이프 레포) run_relations 전량 재생성 — 재제안된 행은 새 점수·이유를 받고
     P2 게이트(RELATION_PERSIST_MIN_CONF_SIMILARITY=0.70)가 신규 적재를 거른다.
  ③ ``--phase purge`` : ``confidence IS NULL`` 잔재(재제안되지 않은 엣지)를 DELETE.
     이것이 "이번에 다시 제안되지 않은 관계의 소멸"을 구현한다 — 만료 status 신설 없이.

기본은 ``--dry-run``(건수·내역만 출력·DB 무변경). 실제 실행은 ``--apply`` 를 명시해야 하며,
①은 백업 파일 생성이 성공해야만 UPDATE 를 실행한다(백업 없는 리셋 금지).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.config.bootstrap import bootstrap_env
from src.database.postgres_util import PostgresUtil

# 백업·리셋 대상 열 — graph_edge 전 열(복원 시 INSERT 로 그대로 되돌릴 수 있어야 한다).
_BACKUP_SQL = """
SELECT ge.edge_id::text, ge.src_node::text, ge.dst_node::text,
       ge.relation_kind_id::text, rk.kind_code,
       ge.confidence, ge.decision_id::text, ge.reason, ge.status,
       ge.topic::text, ge.reviewed_by, ge.reviewed_at::text,
       ge.created_at::text, ge.updated_at::text
FROM graph_edge ge
JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
ORDER BY ge.edge_id
"""

_RESET_SQL = """
UPDATE graph_edge
SET status = 'proposed', confidence = NULL, updated_at = now()
WHERE reviewed_by IS NULL
"""

_PURGE_SQL = "DELETE FROM graph_edge WHERE confidence IS NULL AND reviewed_by IS NULL"


def _counts(cur) -> dict[str, int]:
    """현재 graph_edge 의 status·NULL confidence 분포(진행 보고용)."""
    cur.execute(
        "SELECT status, COUNT(*), COUNT(*) FILTER (WHERE confidence IS NULL)"
        " FROM graph_edge GROUP BY status ORDER BY status"
    )
    out: dict[str, int] = {}
    for st, n, n_null in cur.fetchall():
        out[st] = n
        out[f"{st}(conf NULL)"] = n_null
    return out


def phase_reset(db: PostgresUtil, *, apply: bool, backup_dir: Path) -> int:
    """①리셋 — 전량 백업 후 status/confidence 초기화. 반환값은 대상 행 수.

    Args:
        db: DB 유틸(트랜잭션은 apply 시에만 연다).
        apply: False(기본)면 dry-run — 대상 건수만 출력하고 아무것도 바꾸지 않는다.
        backup_dir: 백업 JSON 을 둘 디렉터리. apply 시 백업 성공이 UPDATE 의 선행 조건.
    """
    with db.connection() as conn, conn.cursor() as cur:
        print("현재 분포:", _counts(cur))
        cur.execute("SELECT COUNT(*) FROM graph_edge WHERE reviewed_by IS NULL")
        target = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM graph_edge WHERE reviewed_by IS NOT NULL")
        protected = int(cur.fetchone()[0])
    print(f"리셋 대상 {target}건 · 사람 결정 보호 {protected}건")
    if not apply:
        print("[dry-run] 변경 없음. 실행하려면 --apply")
        return 0

    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_dir / f"graph_edge_backup_{stamp}.json"
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(_BACKUP_SQL)
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
    backup.write_text(json.dumps(rows, ensure_ascii=False, default=str), encoding="utf-8")
    if not backup.is_file() or backup.stat().st_size < 100:
        print("🔴 백업 파일 생성 실패 — 리셋을 중단한다(백업 없는 리셋 금지)")
        return 1
    print(f"백업: {backup} ({len(rows)}행 · {backup.stat().st_size // 1024}KB)")

    with db.transaction() as conn, conn.cursor() as cur:
        cur.execute(_RESET_SQL)
        print(f"리셋 완료: {cur.rowcount}행 → status='proposed' · confidence=NULL")
    return 0


def phase_purge(db: PostgresUtil, *, apply: bool) -> int:
    """③잔재 소거 — 재생성 후 confidence 가 여전히 NULL 인(재제안 안 된) 엣지 DELETE.

    ⚠️ 반드시 ②(전량 재생성)가 **완주한 뒤** 실행한다 — 중간에 돌리면 아직 순번이 안 온
    자산의 멀쩡한 관계까지 지운다. 완주 판정은 relation_resolution 미해소 0 으로 확인.

    Args:
        db: DB 유틸.
        apply: False(기본)면 dry-run.
    """
    with db.connection() as conn, conn.cursor() as cur:
        print("현재 분포:", _counts(cur))
        # 완주 가드 — pending 이 남아 있으면 재생성 미완주로 보고 중단한다.
        cur.execute("SELECT COUNT(*) FROM relation_resolution WHERE status = 'pending'")
        pending = int(cur.fetchone()[0])
        cur.execute(
            "SELECT COUNT(*) FROM graph_edge WHERE confidence IS NULL AND reviewed_by IS NULL"
        )
        target = int(cur.fetchone()[0])
    if pending > 0:
        print(f"🔴 relation_resolution pending {pending}건 — 재생성 미완주. purge 를 중단한다")
        return 1
    print(f"소거 대상(재제안 안 된 엣지) {target}건")
    if not apply:
        print("[dry-run] 변경 없음. 실행하려면 --apply")
        return 0
    with db.transaction() as conn, conn.cursor() as cur:
        cur.execute(_PURGE_SQL)
        print(f"소거 완료: {cur.rowcount}행 DELETE")
    return 0


def main() -> int:
    """CLI 진입점 — phase 와 apply 여부를 받아 해당 단계를 실행한다."""
    ap = argparse.ArgumentParser(description="점수 축 v3 전환 소급 처리 (기본 dry-run)")
    ap.add_argument("--env", choices=["dev", "prod"], default="dev",
                    help="설정 환경(.env.{env} 로드)")
    ap.add_argument("--phase", choices=["reset", "purge"], required=True,
                    help="reset=①백업+리셋(재생성 전) · purge=③잔재 소거(재생성 완주 후)")
    ap.add_argument("--apply", action="store_true",
                    help="실제 변경 실행(생략 시 dry-run — 건수만 보고)")
    ap.add_argument("--backup-dir", default="backups",
                    help="reset 백업 JSON 디렉터리(기본 ./backups · gitignore 대상)")
    args = ap.parse_args()

    bootstrap_env(args.env)
    db = PostgresUtil()
    with db:
        if args.phase == "reset":
            return phase_reset(db, apply=args.apply, backup_dir=Path(args.backup_dir))
        return phase_purge(db, apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
