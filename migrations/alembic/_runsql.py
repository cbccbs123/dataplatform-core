"""revision 의 upgrade() 가 ``migrations/sql/*.sql`` 을 그대로 실행하기 위한 헬퍼."""

from __future__ import annotations

from pathlib import Path

from alembic import op

# migrations/sql/ 디렉터리 (이 파일: migrations/alembic/_runsql.py)
_SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def run_sql_file(name: str) -> None:
    """``migrations/sql/<name>`` 의 DDL 전체를 현재 마이그레이션 트랜잭션에서 실행한다."""
    path = _SQL_DIR / name
    sql = path.read_text(encoding="utf-8")
    op.execute(sql)
