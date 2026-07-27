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
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import psycopg
from psycopg import Connection, errors
from psycopg.conninfo import make_conninfo
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
        """환경변수에서 접속 설정을 만든다 — DSN 도 설정 객체도 주어지지 않았을 때의 마지막 수단.

        Returns:
            채워진 ``PostgresConfig``. 호스트·포트·계정은 값이 없으면 로컬 기본값을 쓴다.

        Raises:
            PostgresConfigError: 포트가 정수가 아닐 때.
        """
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
    """요구할 최소 PostgreSQL 버전을 정한다(인자 → 환경변수 → 기본값 순).

    버전 가드를 두는 이유: 이 프로젝트는 특정 버전 이상의 기능(벡터 집계 등)에 의존해서,
    낮은 서버에 붙으면 한참 뒤 모호한 SQL 오류로 터진다. 접속 시점에 막는 편이 낫다.

    Args:
        explicit: 코드에서 직접 지정한 값. 있으면 그대로 쓴다(환경변수보다 우선).

    Returns:
        숫자 버전(예: 170000).

    Raises:
        PostgresConfigError: 환경변수 값이 정수가 아닐 때.
    """
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
        """접속 설정을 확정한다(연결은 하지 않는다 — 첫 사용 시점에 풀이 열린다).

        설정 출처는 **인자 → 환경변수 DSN → 환경변수 개별 항목** 순으로 찾는다.

        Args:
            config: 접속 설정 객체. ``dsn`` 과 함께 주면 ``dsn`` 이 우선한다.
            dsn: 접속 문자열. 주면 개별 항목을 조립하지 않는다.
            min_server_version: 요구 최소 서버 버전. ``None`` 이면 환경변수·기본값.
            logger: 로거. ``None`` 이면 이 모듈 로거.
            on_retry: 재시도할 때마다 부를 콜백(관측·지표용).
            on_failure: 최종 실패 시 콜백.
            on_success: 성공 시 콜백.

        Raises:
            PostgresConfigError: 설정 값이 서로 모순일 때(``_validate_config``).
        """
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
        # 069 P1-1: open_pool 의 check-then-set 경합 차단용 락(멀티스레드 서버에서 풀 2중 생성·누수 방지).
        self._pool_lock = threading.Lock()
        self._version_checked = False
        self._on_retry = on_retry
        self._on_failure = on_failure
        self._on_success = on_success
        self._validate_config()

    def _validate_config(self) -> None:
        """설정 값의 앞뒤가 맞는지 확인한다 — 어긋나면 **기동 시점에** 실패시킨다.

        풀 크기가 뒤집혀 있거나 재시도 간격이 음수인 설정은 실행 중에야 이상하게 드러난다.
        DSN 만 준 경우(``config is None``)는 검사할 항목이 없어 그냥 통과한다.

        Raises:
            PostgresConfigError: 값이 범위를 벗어나거나 서로 모순일 때.
        """
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
        """접속 문자열을 만든다 — DSN 이 있으면 그대로, 없으면 항목을 조립한다.

        Returns:
            libpq 접속 문자열.
        """
        if self.dsn:
            return self.dsn
        assert self.config is not None
        # libpq conninfo 는 값에 공백·작은따옴표·백슬래시가 있으면 quoting/escaping 이 필요하다.
        # f-string 손조립은 특수문자 비밀번호 등에서 문자열이 깨지므로(값이 다음 키로 새어들어감),
        # psycopg 표준 make_conninfo 에 위임해 값별 이스케이프를 맡긴다(port·timeout 은 str 로 정규화).
        params: dict[str, Any] = {
            "host": self.config.host,
            "port": str(self.config.port),
            "dbname": self.config.dbname,
            "user": self.config.user,
            "password": self.config.password,
            "connect_timeout": str(self.config.connect_timeout),
            "sslmode": self.config.sslmode,
            "application_name": self.config.application_name,
        }
        if self.config.statement_timeout_ms is not None:
            params["options"] = f"-c statement_timeout={self.config.statement_timeout_ms}"
        return make_conninfo(**params)

    @staticmethod
    def _extract_sqlstate(exc: BaseException) -> str | None:
        """예외에서 SQL 오류 코드를 꺼낸다(드라이버마다 속성 이름이 달라 둘 다 본다).

        Args:
            exc: 잡힌 예외.

        Returns:
            5자리 오류 코드. DB 유래가 아닌 예외면 ``None``.
        """
        sqlstate = getattr(exc, "sqlstate", None)
        if isinstance(sqlstate, str) and sqlstate:
            return sqlstate
        pgcode = getattr(exc, "pgcode", None)
        if isinstance(pgcode, str) and pgcode:
            return pgcode
        return None

    def _is_retryable_error(self, exc: BaseException) -> bool:
        """일시(transient) 오류만 True — 재시도해 성공 가능한 것만 가른다(run_with_retry chokepoint).

        우선순위: config 의 SQLSTATE 제외/포함 목록이 최종 결정권(운영 튜닝 hook). 그 외엔 커넥션
        계열(Operational/Interface) + 직렬화 실패·교착·커넥션 예외만 재시도. 무결성 위반·문법 오류
        등 영구 오류는 재시도해도 동일하므로 False(즉시 raise — 헛 재시도·중복 부작용 방지).
        """
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
        """재시도 파라미터를 한 번에 꺼낸다(설정이 없으면 기본값).

        Returns:
            ``(시도 횟수, 기본 대기 ms, 최대 대기 ms, 지터 ms)``.
        """
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
        """커넥션 풀을 지연 생성한다(최초 호출 때만; 이후 같은 풀 재사용). psycopg_pool 은 지연 import.

        069 P1-1: 더블체크 락킹 — 멀티스레드 서버(포탈 스레드풀)에서 동시 최초 호출 시 check-then-set
        경합으로 풀이 2개 만들어져 한쪽이 close 없이 누수되던 것을 차단한다(락은 최초 생성 때만 경합).
        """
        if self._pool is None:
            with self._pool_lock:
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
        """풀에서 커넥션을 빌려 주고, 블록이 끝나면 풀에 돌려준다.

        빌릴 때마다 서버 버전을 확인하지만 **실제 검사는 첫 번째 한 번뿐**이다.

        Yields:
            빌린 커넥션. 커밋·롤백은 호출자 몫이다(트랜잭션이 필요하면 ``transaction()``).
        """
        pool = self.open_pool()
        with pool.connection() as conn:
            self._ensure_version_checked(conn)
            yield conn

    @contextmanager
    def transaction(self) -> Iterator[Connection[Any]]:
        """트랜잭션 안에서 커넥션을 빌려 준다 — 정상 종료면 커밋, 예외면 롤백.

        Yields:
            트랜잭션이 열린 커넥션. 블록 안에서 예외가 나면 그때까지의 쓰기가 **전부**
            되돌아간다(부분 반영 없음).
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
        """커넥션 풀을 닫는다(이미 닫혔으면 아무 것도 하지 않는다).

        다시 쓰면 풀이 새로 열리므로 종료 후 재사용도 가능하다.
        """
        if self._pool is not None:
            self.logger.debug("Closing PostgreSQL connection pool.")
            self._pool.close()
            self._pool = None

    def __enter__(self) -> PostgresUtil:
        """``with`` 진입 — 풀을 **미리 열어 둔다**.

        첫 질의에서 연결 비용을 물지 않게 하려는 예열이다.
        """
        self.open_pool()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """``with`` 종료 — 예외 여부와 무관하게 풀을 닫는다."""
        self.close()

    def _validate_server_version(self, conn: psycopg.Connection[Any]) -> None:
        """서버 버전이 요구 사양에 못 미치면 **접속 직후 실패시킨다**.

        낮은 버전에서도 대부분의 질의는 통과하다가, 뒤늦게 특정 기능(벡터 집계 등)에서
        모호한 오류로 터진다. 원인이 분명한 시점에 막는 편이 낫다.

        Raises:
            PostgreSQLVersionError: 서버 버전이 최소 요구보다 낮을 때.
        """
        server_version = conn.info.server_version
        if server_version < self.min_server_version:
            raise PostgreSQLVersionError(
                f"PostgreSQL {self.min_server_version // 10000}+ required, "
                f"but connected server version is {server_version}."
            )
        self.logger.info("Connected to PostgreSQL server_version_num=%s", server_version)
        self._version_checked = True

    def _ensure_version_checked(self, conn: Connection[Any]) -> None:
        """버전 검사를 **풀 수명 동안 한 번만** 수행한다.

        같은 서버에 붙는 커넥션마다 다시 물으면 왕복만 늘고 얻는 게 없다.
        """
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
            """재시도 단위 — 커넥션을 빌려 질의하고 커밋까지 마친다(실패 시 통째로 재실행)."""
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
        """여러 문장을 **한 트랜잭션으로 묶어** 실행한다 — 전부 성공하거나 전부 되돌아간다.

        Args:
            callback: 커넥션을 받아 실제 작업을 하는 함수. 이 안에서 난 예외는 롤백을 부른다.
            idempotent: **여러 번 실행해도 결과가 같은 작업일 때만 True**. True 면 일시
                오류에서 재시도하는데, 멱등하지 않은 쓰기를 재시도하면 중복 반영된다.

        Returns:
            ``callback`` 의 반환값.
        """
        def _op() -> Any:
            """재시도 단위 — 트랜잭션을 열고 콜백을 실행한다."""
            with self.transaction() as conn:
                return callback(conn)

        return self.run_with_retry(
            _op,
            operation_name="execute_in_transaction",
            idempotent=idempotent,
        )
