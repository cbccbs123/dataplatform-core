"""Alembic 마이그레이션 환경.

DB 접속 정보는 ``src.database.postgres_util`` 의 환경변수 규칙을 재사용한다:
``DATABASE_URL`` / ``POSTGRES_DSN``(URL 형식) 우선, 없으면 ``POSTGRES_*`` 개별 변수.
``.env.dev`` / ``.env.prod`` 가 있으면 먼저 로드한다(``ALEMBIC_ENV`` 로 프로파일 선택, 기본 dev).

각 revision 의 ``upgrade()`` 는 ``migrations/sql/*.sql`` 의 손으로 작성한 DDL 을 그대로 실행한다
(pgvector ``vector``/``tsvector``/GIN ``jsonb_path_ops`` 는 autogenerate 가 부정확하므로 raw SQL 유지).
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import URL, make_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    profile = os.getenv("ALEMBIC_ENV", "dev")
    dotenv_path = PROJECT_ROOT / f".env.{profile}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _sqlalchemy_url() -> URL:
    """psycopg(v3) 드라이버를 쓰는 SQLAlchemy URL 을 만든다.

    ``DATABASE_URL``/``POSTGRES_DSN`` 은 URL 형식을 가정하고 드라이버를
    ``postgresql+psycopg`` 로 보정한다. 없으면 ``POSTGRES_*`` 로 조립한다.
    """
    dsn = (os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_DSN") or "").strip()
    if dsn:
        url = make_url(dsn)
        if url.get_backend_name() in ("postgresql", "postgres"):
            url = url.set(drivername="postgresql+psycopg")
        return url

    from src.database.postgres_util import PostgresConfig

    cfg = PostgresConfig.from_env()
    return URL.create(
        "postgresql+psycopg",
        username=cfg.user,
        password=cfg.password,
        host=cfg.host,
        port=cfg.port,
        database=cfg.dbname,
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_sqlalchemy_url().render_as_string(hide_password=False),
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    from sqlalchemy import create_engine

    engine = create_engine(_sqlalchemy_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


_load_dotenv_if_present()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
