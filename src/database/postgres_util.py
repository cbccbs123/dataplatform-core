"""
PostgreSQL 17+ compatible utility module.

Features:
- Connection pool support (psycopg_pool)
- Configurable logging
- Explicit transaction context manager
- Server version detection and minimum-version guard
- Convenience query helpers (execute, fetch_one, fetch_all)
- Health check utility
"""

from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence, cast

import psycopg
from psycopg import errors
from psycopg import Connection
from psycopg.rows import dict_row


PG17_NUMERIC_VERSION = 170000


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
    password: str = "cbccbs"
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
    def from_dsn(cls, dsn: str) -> "PostgresConfig":
        """Optional helper when caller prefers DSN parsing externally."""
        # Keep config simple; DSN can be passed directly to PostgresUtil.
        raise NotImplementedError("Use PostgresUtil(dsn=...) directly.")


class PostgresUtil:
    """Small utility wrapper for psycopg3 with pooling."""

    def __init__(
        self,
        *,
        config: PostgresConfig | None = None,
        dsn: str | None = None,
        min_server_version: int = PG17_NUMERIC_VERSION,
        logger: logging.Logger | None = None,
        on_retry: Callable[[Mapping[str, Any]], None] | None = None,
        on_failure: Callable[[Mapping[str, Any]], None] | None = None,
        on_success: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not config and not dsn:
            config = PostgresConfig()

        self.config = config
        self.dsn = dsn
        self.min_server_version = min_server_version
        self.logger = logger or logging.getLogger(__name__)
        self._conn: Connection[Any] | None = None
        self._pool: Any = None
        self._version_checked = False
        self._on_retry = on_retry
        self._on_failure = on_failure
        self._on_success = on_success
        self._metrics: dict[str, int] = {
            "operations_total": 0,
            "operations_succeeded": 0,
            "operations_failed": 0,
            "retries_total": 0,
        }
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
        if isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)):
            return True
        if isinstance(
            exc,
            (
                errors.SerializationFailure,
                errors.DeadlockDetected,
                errors.ConnectionException,
            ),
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
        attempts, base_delay_ms, max_delay_ms, jitter_ms = self._retry_config()
        if not idempotent:
            attempts = 1
        last_error: BaseException | None = None
        self._metrics["operations_total"] += 1

        for attempt in range(1, attempts + 1):
            try:
                result = operation()
                self._metrics["operations_succeeded"] += 1
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
                    self._metrics["operations_failed"] += 1
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
                self._metrics["retries_total"] += 1
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

    def get_metrics(self) -> Mapping[str, int]:
        """Returns retry/operation counters for monitoring."""
        return dict(self._metrics)

    def reset_metrics(self) -> None:
        """Resets in-memory counters."""
        for key in self._metrics:
            self._metrics[key] = 0

    def connect(self) -> Connection[Any]:
        if self._conn is None or self._conn.closed:
            self.logger.debug("Opening single PostgreSQL connection.")
            conn = psycopg.connect(self._build_conninfo())
            self._validate_server_version(conn)
            self._conn = conn
        return self._conn

    def open_pool(self) -> Any:
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
    def connection(self, *, use_pool: bool = True) -> Iterator[Connection[Any]]:
        """
        Provide a DB connection.

        - use_pool=True: borrow and return connection from pool.
        - use_pool=False: use single persistent connection.
        """
        if use_pool:
            pool = self.open_pool()
            with pool.connection() as conn:
                self._ensure_version_checked(conn)
                yield conn
        else:
            conn = self.connect()
            self._ensure_version_checked(conn)
            yield conn

    @contextmanager
    def transaction(self, *, use_pool: bool = True) -> Iterator[Connection[Any]]:
        """
        Transaction context manager.

        Commits on success and rolls back on error.
        """
        with self.connection(use_pool=use_pool) as conn:
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
        if self._conn is not None and not self._conn.closed:
            self.logger.debug("Closing single PostgreSQL connection.")
            self._conn.close()
        self._conn = None
        if self._pool is not None:
            self.logger.debug("Closing PostgreSQL connection pool.")
            self._pool.close()
            self._pool = None

    def __enter__(self) -> "PostgresUtil":
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
        if not self._version_checked:
            self._validate_server_version(conn)

    def server_version(self) -> Mapping[str, Any]:
        row = self.fetch_one("SHOW server_version;")
        return {"server_version": row["server_version"]}

    def health_check(self) -> Mapping[str, Any]:
        with self.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    cast(Any, "SELECT 1 AS ok, NOW() AS server_time, current_setting('application_name') AS app_name;")
                )
                row = cur.fetchone()
            if row is None:
                return {"ok": False}
            return {
                "ok": bool(row["ok"] == 1),
                "server_time": row["server_time"],
                "server_version_num": conn.info.server_version,
                "application_name": row["app_name"],
            }

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
        *,
        use_pool: bool = True,
        idempotent: bool = False,
    ) -> int:
        def _op() -> int:
            self.logger.debug("Execute query: %s", query)
            with self.connection(use_pool=use_pool) as conn:
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
        use_pool: bool = True,
        idempotent: bool = False,
    ) -> Any:
        """
        Run arbitrary DB logic inside a managed transaction.

        Useful in services where multiple statements must succeed/fail together.
        """
        def _op() -> Any:
            with self.transaction(use_pool=use_pool) as conn:
                return callback(conn)

        return self.run_with_retry(
            _op,
            operation_name="execute_in_transaction",
            idempotent=idempotent,
        )

    def fetch_one(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
        *,
        use_pool: bool = True,
    ) -> Mapping[str, Any]:
        def _op() -> Mapping[str, Any]:
            self.logger.debug("Fetch one query: %s", query)
            with self.connection(use_pool=use_pool) as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(cast(Any, query), params)
                    row = cur.fetchone()
            if row is None:
                return {}
            return cast(Mapping[str, Any], row)

        return cast(
            Mapping[str, Any],
            self.run_with_retry(_op, operation_name="fetch_one", idempotent=True),
        )

    def fetch_all(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
        *,
        use_pool: bool = True,
    ) -> list[Mapping[str, Any]]:
        def _op() -> list[Mapping[str, Any]]:
            self.logger.debug("Fetch all query: %s", query)
            with self.connection(use_pool=use_pool) as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(cast(Any, query), params)
                    rows = cur.fetchall()
            return cast(list[Mapping[str, Any]], rows)

        return cast(
            list[Mapping[str, Any]],
            self.run_with_retry(_op, operation_name="fetch_all", idempotent=True),
        )


# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s %(levelname)s %(name)s - %(message)s",
#     )
#     util = PostgresUtil()
#     with util:
#         print("Connected:", util.server_version())
#         print("Health:", util.health_check())
