"""
PostgreSQL utility module (default: require server_version_num >= 170000 / PG17).

Connection when ``PostgresUtil()`` is called with no arguments:

- ``DATABASE_URL`` or ``POSTGRES_DSN`` if set, else
- ``PostgresConfig.from_env()`` (``POSTGRES_HOST``, ``POSTGRES_PORT``, ``POSTGRES_DB`` or
  ``POSTGRES_DBNAME``, ``POSTGRES_USER``, ``POSTGRES_PASSWORD``).

Override minimum version with ``POSTGRES_MIN_SERVER_VERSION_NUM`` (e.g. ``160000`` for PG16).

Features:
- Connection pool support (psycopg_pool)
- Configurable logging
- Explicit transaction context manager
- Server version detection and minimum-version guard
- 편의 쿼리 헬퍼 (execute, execute_in_transaction)
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import psycopg
from psycopg import Connection, errors
from psycopg.rows import dict_row

PG17_NUMERIC_VERSION = 170000

_POSTGRES_MIN_VER_ENV = "POSTGRES_MIN_SERVER_VERSION_NUM"


class PostgreSQLVersionError(RuntimeError):
    """Raised when connected PostgreSQL server is below required version."""


class PostgresConfigError(ValueError):
    """Raised when PostgresConfig contains invalid values."""


@dataclass
class PostgresConfig:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "postgres"
    user: str = "postgres"
    password: str = ""
    connect_timeout: int = 10
    sslmode: str = "prefer"
    min_pool_size: int = 1
    max_pool_size: int = 10
    application_name: str = "postgres-util"
    statement_timeout_ms: int | None = None
    retry_attempts: int = 3
    retry_base_delay_ms: int = 100
    retry_max_delay_ms: int = 2000
    retry_jitter_ms: int = 50
    retry_on_sqlstates: tuple[str, ...] = (
        "40001",  # serialization_failure
        "40P01",  # deadlock_detected
        "08000",  # connection_exception
        "08003",  # connection_does_not_exist
        "08006",  # connection_failure
        "08001",  # sqlclient_unable_to_establish_sqlconnection
        "08004",  # sqlserver_rejected_establishment_of_sqlconnection
        "57P01",  # admin_shutdown
    )
    retry_exclude_sqlstates: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> PostgresConfig:
        """Build config from env (used when neither ``PostgresUtil(dsn=...)`` nor explicit config is given)."""
        port_raw = os.getenv("POSTGRES_PORT", "5432")
        try:
            port = int(str(port_raw).strip())
        except ValueError as e:
            raise PostgresConfigError(f"Invalid POSTGRES_PORT: {port_raw!r}") from e
        db = os.getenv("POSTGRES_DB") or os.getenv("POSTGRES_DBNAME") or "postgres"
        return cls(
            host=str(os.getenv("POSTGRES_HOST", "localhost")).strip() or "localhost",
            port=port,
            dbname=str(db).strip() or "postgres",
            user=str(os.getenv("POSTGRES_USER", "postgres")).strip() or "postgres",
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )


def _resolve_min_server_version(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    raw = os.getenv(_POSTGRES_MIN_VER_ENV)
    if raw is None or not str(raw).strip():
        return PG17_NUMERIC_VERSION
    try:
        return int(str(raw).strip())
    except ValueError as e:
        raise PostgresConfigError(
            f"Invalid {_POSTGRES_MIN_VER_ENV}: {raw!r} (expected integer server_version_num)"
        ) from e


class PostgresUtil:
    """psycopg3 풀·트랜잭션·재시도를 감싼 단일 DB 접근 seam.

    persist/search/relations 등 모든 DB 접근이 이 래퍼를 공유한다(CLAUDE.md: PG17+ 풀·트랜잭션).
    ``connection()``/``transaction()`` 으로 풀에서 커넥션을 빌리고, ``run_with_retry`` 로 일시 오류를
    흡수하며, 첫 커넥션 사용 시 server_version 가드로 PG17 미만 서버를 거부한다(헌법: PG17 고정).
    """

    def __init__(
        self,
        *,
        config: PostgresConfig | None = None,
        dsn: str | None = None,
        min_server_version: int | None = None,
        logger: logging.Logger | None = None,
        on_retry: Callable[[Mapping[str, Any]], None] | None = None,
        on_failure: Callable[[Mapping[str, Any]], None] | None = None,
        on_success: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not config and not dsn:
            env_dsn = (os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_DSN") or "").strip()
            if env_dsn:
                dsn = env_dsn
            else:
                config = PostgresConfig.from_env()

        self.config = config
        self.dsn = dsn
        self.min_server_version = _resolve_min_server_version(min_server_version)
        self.logger = logger or logging.getLogger(__name__)
        self._pool: Any = None
        self._version_checked = False
        self._on_retry = on_retry
        self._on_failure = on_failure
        self._on_success = on_success
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config is None:
            return

        if self.config.min_pool_size <= 0:
            raise PostgresConfigError("min_pool_size must be >= 1.")
        if self.config.max_pool_size < self.config.min_pool_size:
            raise PostgresConfigError("max_pool_size must be >= min_pool_size.")
        if self.config.connect_timeout <= 0:
            raise PostgresConfigError("connect_timeout must be >= 1.")
        if (
            self.config.statement_timeout_ms is not None
            and self.config.statement_timeout_ms <= 0
        ):
            raise PostgresConfigError("statement_timeout_ms must be >= 1 when set.")
        if self.config.retry_attempts <= 0:
            raise PostgresConfigError("retry_attempts must be >= 1.")
        if self.config.retry_base_delay_ms < 0:
            raise PostgresConfigError("retry_base_delay_ms must be >= 0.")
        if self.config.retry_max_delay_ms < self.config.retry_base_delay_ms:
            raise PostgresConfigError("retry_max_delay_ms must be >= retry_base_delay_ms.")
        if self.config.retry_jitter_ms < 0:
            raise PostgresConfigError("retry_jitter_ms must be >= 0.")

    def _build_conninfo(self) -> str:
        if self.dsn:
            return self.dsn
        assert self.config is not None
        conninfo = (
            f"host={self.config.host} "
            f"port={self.config.port} "
            f"dbname={self.config.dbname} "
            f"user={self.config.user} "
            f"password={self.config.password} "
            f"connect_timeout={self.config.connect_timeout} "
            f"sslmode={self.config.sslmode} "
            f"application_name={self.config.application_name}"
        )
        if self.config.statement_timeout_ms is not None:
            conninfo += f" options='-c statement_timeout={self.config.statement_timeout_ms}'"
        return conninfo

    @staticmethod
    def _extract_sqlstate(exc: BaseException) -> str | None:
        sqlstate = getattr(exc, "sqlstate", None)
        if isinstance(sqlstate, str) and sqlstate:
            return sqlstate
        pgcode = getattr(exc, "pgcode", None)
        if isinstance(pgcode, str) and pgcode:
            return pgcode
        return None

    def _is_retryable_error(self, exc: BaseException) -> bool:
        sqlstate = self._extract_sqlstate(exc)
        if self.config is not None and sqlstate is not None:
            if sqlstate in self.config.retry_exclude_sqlstates:
                return False
            if sqlstate in self.config.retry_on_sqlstates:
                return True
        if isinstance(exc, psycopg.OperationalError | psycopg.InterfaceError):
            return True
        if isinstance(
            exc,
            errors.SerializationFailure | errors.DeadlockDetected | errors.ConnectionException,
        ):
            return True
        return False

    def _retry_config(self) -> tuple[int, int, int, int]:
        if self.config is None:
            return (3, 100, 2000, 50)
        return (
            self.config.retry_attempts,
            self.config.retry_base_delay_ms,
            self.config.retry_max_delay_ms,
            self.config.retry_jitter_ms,
        )

    def run_with_retry(
        self,
        operation: Callable[[], Any],
        *,
        operation_name: str = "db-operation",
        idempotent: bool = True,
    ) -> Any:
        """일시적(transient) DB 오류에 대해 ``operation`` 을 지수 백오프+지터로 재시도한다.

        ``execute``/``execute_in_transaction`` 등 모든 DB 호출이 경유하는 재시도 chokepoint다.
        재시도 대상은 ``_is_retryable_error`` 가 True 인 일시 오류(직렬화 실패·교착·커넥션 단절 등
        SQLSTATE 화이트리스트)뿐이며, 그 외 오류나 마지막 시도 실패는 즉시 raise 한다.

        **불변식(중요)**: ``idempotent=False`` 면 단 1회만 시도한다(재시도 안 함). 부분 적용된 비멱등
        쓰기를 재시도하면 중복 적용될 수 있어서다 — 안전한 재시도는 호출자의 멱등성 보장이 전제다.

        ``on_retry``/``on_success``/``on_failure`` 는 메트릭·로깅용 관측 훅(주입 콜백)이다.
        """
        attempts, base_delay_ms, max_delay_ms, jitter_ms = self._retry_config()
        if not idempotent:
            attempts = 1  # 비멱등 연산은 재시도 금지(부분 적용 시 중복 위험) → 단일 시도로 강제.
        last_error: BaseException | None = None

        for attempt in range(1, attempts + 1):
            try:
                result = operation()
                if self._on_success is not None:
                    self._on_success(
                        {
                            "operation_name": operation_name,
                            "attempt": attempt,
                            "idempotent": idempotent,
                        }
                    )
                return result
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_error(exc) or attempt >= attempts:
                    failure_payload = {
                        "operation_name": operation_name,
                        "attempt": attempt,
                        "attempts": attempts,
                        "idempotent": idempotent,
                        "error_type": type(exc).__name__,
                        "sqlstate": self._extract_sqlstate(exc),
                    }
                    if self._on_failure is not None:
                        self._on_failure(failure_payload)
                    self.logger.exception(
                        "Operation %s failed (attempt %s/%s, idempotent=%s).",
                        operation_name,
                        attempt,
                        attempts,
                        idempotent,
                    )
                    raise

                backoff_ms = min(max_delay_ms, base_delay_ms * (2 ** (attempt - 1)))
                jitter = random.randint(0, jitter_ms) if jitter_ms > 0 else 0
                sleep_ms = backoff_ms + jitter
                retry_payload = {
                    "operation_name": operation_name,
                    "attempt": attempt,
                    "attempts": attempts,
                    "idempotent": idempotent,
                    "error_type": type(exc).__name__,
                    "sqlstate": self._extract_sqlstate(exc),
                    "sleep_ms": sleep_ms,
                }
                if self._on_retry is not None:
                    self._on_retry(retry_payload)
                self.logger.warning(
                    "Retryable error in %s (attempt %s/%s): %s. Sleeping %sms.",
                    operation_name,
                    attempt,
                    attempts,
                    exc,
                    sleep_ms,
                )
                time.sleep(sleep_ms / 1000.0)

        if last_error is not None:
            raise last_error
        raise RuntimeError("run_with_retry reached unexpected state.")

    def open_pool(self) -> Any:
        """커넥션 풀을 지연 생성한다(최초 호출 때만; 이후 같은 풀 재사용). psycopg_pool 은 지연 import."""
        if self._pool is None:
            from psycopg_pool import ConnectionPool  # pyright: ignore[reportMissingImports]

            assert self.config is not None or self.dsn is not None
            min_size = self.config.min_pool_size if self.config else 1
            max_size = self.config.max_pool_size if self.config else 10
            self.logger.debug(
                "Opening PostgreSQL connection pool (min=%s, max=%s).",
                min_size,
                max_size,
            )
            self._pool = ConnectionPool(
                conninfo=self._build_conninfo(),
                min_size=min_size,
                max_size=max_size,
                open=True,
            )
        return self._pool

    @contextmanager
    def connection(self) -> Iterator[Connection[Any]]:
        """풀에서 커넥션을 빌려 제공하고, 블록 종료 시 풀에 반환한다."""
        pool = self.open_pool()
        with pool.connection() as conn:
            self._ensure_version_checked(conn)
            yield conn

    @contextmanager
    def transaction(self) -> Iterator[Connection[Any]]:
        """
        Transaction context manager.

        Commits on success and rolls back on error.
        """
        with self.connection() as conn:
            self.logger.debug("Starting transaction.")
            try:
                with conn.transaction():
                    yield conn
            except Exception:
                self.logger.exception("Transaction failed. Rolled back.")
                raise
            else:
                self.logger.debug("Transaction committed.")

    def close(self) -> None:
        if self._pool is not None:
            self.logger.debug("Closing PostgreSQL connection pool.")
            self._pool.close()
            self._pool = None

    def __enter__(self) -> PostgresUtil:
        # Default path in services is pooling, so warm up pool first.
        self.open_pool()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _validate_server_version(self, conn: psycopg.Connection[Any]) -> None:
        server_version = conn.info.server_version
        if server_version < self.min_server_version:
            raise PostgreSQLVersionError(
                f"PostgreSQL {self.min_server_version // 10000}+ required, "
                f"but connected server version is {server_version}."
            )
        self.logger.info("Connected to PostgreSQL server_version_num=%s", server_version)
        self._version_checked = True

    def _ensure_version_checked(self, conn: Connection[Any]) -> None:
        # PG 버전 가드는 풀 수명 동안 1회만(첫 커넥션 사용 시점)에 수행한다 — 매 borrow 마다 재검증 회피.
        if not self._version_checked:
            self._validate_server_version(conn)

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
        *,
        idempotent: bool = False,
    ) -> int:
        """단일 쿼리를 실행하고 영향 행 수(rowcount)를 반환한다(자체 connection→commit).

        ``idempotent`` 기본 False(재시도 안 함) — 임의 쿼리는 멱등 보장이 없어서다. 안전히 재시도하려면
        멱등 쿼리에 한해 ``idempotent=True`` 로 호출한다(``run_with_retry`` 불변식 참고).
        """
        def _op() -> int:
            self.logger.debug("Execute query: %s", query)
            with self.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(cast(Any, query), params)
                    affected = cur.rowcount
                conn.commit()
            return affected

        return cast(
            int,
            self.run_with_retry(
                _op,
                operation_name="execute",
                idempotent=idempotent,
            ),
        )

    def execute_in_transaction(
        self,
        callback: Callable[[Connection[Any]], Any],
        *,
        idempotent: bool = False,
    ) -> Any:
        """
        Run arbitrary DB logic inside a managed transaction.

        Useful in services where multiple statements must succeed/fail together.
        """
        def _op() -> Any:
            with self.transaction() as conn:
                return callback(conn)

        return self.run_with_retry(
            _op,
            operation_name="execute_in_transaction",
            idempotent=idempotent,
        )
