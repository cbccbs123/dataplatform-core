"""점수 축 v3 전환(2026-07-31)의 소급 처리 — 재생성 전 리셋 + 재생성 후 잔재 소거.

**왜 그냥 재생성하면 안 되는가** (이 스크립트가 존재하는 이유):

1. `graph_persist` 의 upsert 는 신뢰도를 ``GREATEST(기존, 신규)`` 로 화해한다 — 옛 점수
   (기준 없는 감각치·0.95 등)가 새 점수(v3 · 최고 0.9)보다 커서 **옛 점수가 살아남는다.**
   그대로 재생성하면 새 점수 체계가 기존 행에 한 건도 적용되지 않는다.
2. upsert 는 ``status`` 를 절대 덮지 않는다(사람 결정 보호) — 자동승인 시절의 ``active``
   499건(이름표 정확도 67% 실측)이 "확인됨"으로 계속 노출된다.
3. 이번에 재제안되지 않는 엣지(새 게이트 미달·프롬프트 개선으로 소멸)는 지워지는 경로가
   없다 — upsert 는 삭제를 모른다(specs/081 발견 7).

**절차 — 두 갈래** (각 단계는 별도 실행 · 재생성은 파이프라인 레포):

**A. 백지 재생성(권장 · ``--phase clear``)** — "진짜 결과"를 보려면 이 쪽이다.
  ① ``--phase clear`` : 전량 백업(JSON) 후 ``graph_edge`` **전량 DELETE** +
     ``relation_resolution`` 전량 DELETE(재처리 대상으로 되돌림). ``node`` 는 **지우지 않는다**
     (자산 노드는 재생성이 그대로 재사용).
  ② run_relations 전량 재생성 — DB 에 남는 모든 엣지가 새 프롬프트 산물이다.
  ③ 없음. 잔재가 애초에 없다.
  왜 이 쪽이 정직한가: upsert 는 종류가 바뀌면 **행을 대체하지 않고 하나 더 만들고**
  (키가 `(src,dst,kind_id)`), 재제안되지 않은 옛 엣지는 지우는 경로가 없다(발견 7).
  옛 행을 남긴 채 재생성하면 새 프롬프트의 결과와 옛 잔재가 섞여 측정이 오염된다.

**B. 점수만 갱신(``--phase reset`` → 재생성 → ``--phase purge``)** — 옛 ``edge_id``·생성시각을
  보존해야 할 때만.
  ① ``--phase reset`` : 백업 후 ``status='proposed'``, ``confidence=NULL``.
     confidence=NULL 이면 GREATEST(COALESCE(NULL,0), 신규) = 신규 — 새 점수가 항상 이긴다.
  ② 재생성. ③ ``--phase purge`` : ``confidence IS NULL`` 잔재 DELETE.
  ⚠️ 종류가 바뀐 쌍은 옛 종류 행이 함께 남는다(그래서 A 가 권장).

세 phase 모두 ``reviewed_by IS NOT NULL``(사람이 만진) 행은 건드리지 않는다 — 현재 0건
실측이나 원칙을 코드로 박는다. 기본은 dry-run(건수만 출력·DB 무변경) · 실제 실행은 ``--apply``
명시 필요 · 파괴적 phase 는 **백업 파일 생성 성공이 선행 조건**이다.
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

    if _write_backup(db, backup_dir, "reset") is None:
        print("🔴 백업 파일 생성 실패 — 리셋을 중단한다(백업 없는 리셋 금지)")
        return 1

    with db.transaction() as conn, conn.cursor() as cur:
        cur.execute(_RESET_SQL)
        print(f"리셋 완료: {cur.rowcount}행 → status='proposed' · confidence=NULL")
    return 0


def _write_backup(db: PostgresUtil, backup_dir: Path, tag: str) -> Path | None:
    """graph_edge 전 열을 JSON 으로 떠서 경로를 돌려준다. 실패하면 ``None``.

    파괴적 phase 의 선행 조건 — 이 함수가 성공해야만 DELETE/UPDATE 를 실행한다.

    Args:
        db: DB 유틸.
        backup_dir: 백업 파일을 둘 디렉터리(없으면 만든다).
        tag: 파일명에 넣을 단계 표식(``clear``/``reset``) — 어느 단계의 백업인지 사후 식별용.

    Returns:
        생성된 백업 파일 경로. 파일이 없거나 비정상적으로 작으면 ``None``.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_dir / f"graph_edge_{tag}_{stamp}.json"
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(_BACKUP_SQL)
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
    backup.write_text(json.dumps(rows, ensure_ascii=False, default=str), encoding="utf-8")
    if not backup.is_file() or backup.stat().st_size < 100:
        return None
    print(f"백업: {backup} ({len(rows)}행 · {backup.stat().st_size // 1024}KB)")
    return backup


def phase_clear(db: PostgresUtil, *, apply: bool, backup_dir: Path) -> int:
    """①백지화 — ``graph_edge``·``relation_resolution`` 전량 삭제(백업 선행).

    ``node`` 는 지우지 않는다 — 자산 노드는 재생성이 그대로 재사용하며, 지우면 노드 재생성
    비용만 늘고 얻는 것이 없다. ``graph_edge`` 를 참조하는 FK 는 없음을 실측 확인했다.

    Args:
        db: DB 유틸.
        apply: False(기본)면 dry-run — 대상 건수만 출력하고 아무것도 지우지 않는다.
        backup_dir: 백업 JSON 디렉터리. 백업 성공이 DELETE 의 선행 조건.
    """
    with db.connection() as conn, conn.cursor() as cur:
        print("현재 분포:", _counts(cur))
        cur.execute("SELECT COUNT(*) FROM graph_edge WHERE reviewed_by IS NOT NULL")
        protected = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM graph_edge")
        edges = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM relation_resolution")
        res = int(cur.fetchone()[0])
    print(f"삭제 대상: graph_edge {edges}건 · relation_resolution {res}건 "
          f"(node 는 유지) · 사람 결정 {protected}건")
    if protected:
        print("🔴 사람 검토 흔적이 있다 — 백지화를 중단한다(보존 정책 재검토 필요)")
        return 1
    if not apply:
        print("[dry-run] 변경 없음. 실행하려면 --apply")
        return 0
    if _write_backup(db, backup_dir, "clear") is None:
        print("🔴 백업 생성 실패 — 삭제를 중단한다(백업 없는 삭제 금지)")
        return 1
    with db.transaction() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM graph_edge")
        n_edges = cur.rowcount
        cur.execute("DELETE FROM relation_resolution")
        n_res = cur.rowcount
        print(f"백지화 완료: graph_edge {n_edges}행 · relation_resolution {n_res}행 DELETE")
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
    ap.add_argument("--phase", choices=["clear", "reset", "purge"], required=True,
                    help="clear=백지 재생성용 전량 삭제(권장) · reset=점수만 갱신용 리셋 · "
                         "purge=reset 경로의 잔재 소거(재생성 완주 후)")
    ap.add_argument("--apply", action="store_true",
                    help="실제 변경 실행(생략 시 dry-run — 건수만 보고)")
    ap.add_argument("--backup-dir", default="backups",
                    help="clear·reset 백업 JSON 디렉터리(기본 ./backups · gitignore 대상)")
    args = ap.parse_args()

    bootstrap_env(args.env)
    db = PostgresUtil()
    with db:
        if args.phase == "clear":
            return phase_clear(db, apply=args.apply, backup_dir=Path(args.backup_dir))
        if args.phase == "reset":
            return phase_reset(db, apply=args.apply, backup_dir=Path(args.backup_dir))
        return phase_purge(db, apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
