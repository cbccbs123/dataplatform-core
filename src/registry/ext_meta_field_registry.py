"""F-4.13 ext_meta 거버넌스 레지스트리 — ``ext_meta_field_registry`` 단일 정본 (spec 039·040·041).

스펙별 책임
    **039 (v280)** — ``json_schema``(JSON Schema)로 ext_meta **값 형태** 검증.
        ``check_ext_meta_values`` · ``fetch_ext_key_schemas`` · ``validate_ext_meta`` 값 루프.
    **040-W1 (v290)** — ``access_tier`` 컬럼·시드. ``fetch_access_tiers`` (write 집행은 042).
    **041 (v291)** — 테이블 정본 ``ext_meta_field_registry``(레거시 ``schema_registry`` 는 DDL만 유지).

경로 분리(헌법 6조)
    - **write path** — ``run_ingest`` → ``validate_ext_meta`` (키 허용·값 스키마; tier 무관 전량 적재).
    - **read path** — 포탈 042 → ``fetch_access_tiers`` + ``project_ext_meta`` (clearance 미달 **키 omit**).

레지스트리에 ``status='active'`` 인 행만 ingest·read API 대상. 미등록 도메인(allowed 빈 집합)은
``validate_ext_meta`` 가 **검증 생략**(키 미등록 시 게이트 무력화 — 운영 시 시드 필수).
"""

from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row


class ExtMetaValidationError(ValueError):
    """``ext_meta`` 키 위반(미등록 키) 또는 값 위반(JSON Schema 불일치) — ingest 중단용."""


def fetch_allowed_ext_keys(conn: Connection[Any], domain: str) -> set[str]:
    """``domain`` 의 활성 ext_meta 허용 키 집합 (039 키 게이트)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT meta_key FROM ext_meta_field_registry "
            "WHERE domain = %s AND status = 'active'",
            (domain,),
        )
        return {r["meta_key"] for r in cur.fetchall()}


def _schema_is_validatable(schema: dict[str, Any] | None) -> bool:
    """값 검증을 돌릴 만한 스키마인지 본다.

    Args:
        schema: 등록된 스키마(없을 수도 있다).

    Returns:
        타입이 정의돼 있으면 참. **빈 스키마는 건너뛴다** — 검증할 규칙이 없는데 돌리면
        무엇이든 통과해, "검증했다"는 착각만 남는다.
    """
    return bool(schema and isinstance(schema, dict) and schema.get("type"))


def check_ext_meta_values(
    schemas: dict[str, dict[str, Any]],
    ext_meta: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    """ext_meta **값** 검증 (039).

    Args:
        schemas: 키별 등록 스키마.
        ext_meta: 검사할 확장 메타. ``None``·빈 dict 면 검사할 것이 없다.

    Returns:
        ``[(키, 사유)]`` 위반 목록. **정렬해 돌려준다** — 순서가 흔들리면 같은 입력에
        다른 오류 메시지가 나와 테스트와 로그가 불안정해진다. 스키마가 없거나 검증할 규칙이
        없는 키는 조용히 넘어간다.
    """
    if not ext_meta:
        return []
    from jsonschema import Draft202012Validator, ValidationError

    violations: list[tuple[str, str]] = []
    for key in sorted(ext_meta.keys()):
        schema = schemas.get(key)
        if not _schema_is_validatable(schema):
            continue
        try:
            Draft202012Validator(schema).validate(ext_meta[key])
        except ValidationError as exc:
            violations.append((key, exc.message))
    return sorted(violations, key=lambda x: (x[0], x[1]))


def fetch_ext_key_schemas(conn: Connection[Any], domain: str) -> dict[str, dict[str, Any]]:
    """``domain`` 의 활성 ext_meta 키→JSON Schema 맵 (039, validatable 만)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT meta_key, json_schema FROM ext_meta_field_registry "
            "WHERE domain = %s AND status = 'active'",
            (domain,),
        )
        out: dict[str, dict[str, Any]] = {}
        for row in cur.fetchall():
            schema = row["json_schema"]
            if _schema_is_validatable(schema):
                out[row["meta_key"]] = schema
        return out


def fetch_access_tiers(conn: Connection[Any], domain: str) -> dict[str, str]:
    """``domain`` 의 활성 ext_meta 키→``access_tier`` 맵 (040-W1).

    read projection(042) 전용 — ingest 는 tier 와 무관하게 전량 DB 적재.
    값은 ``AccessTier`` StrEnum(``status_vocab``)과 CHECK 동기.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT meta_key, access_tier FROM ext_meta_field_registry "
            "WHERE domain = %s AND status = 'active'",
            (domain,),
        )
        return {row["meta_key"]: row["access_tier"] for row in cur.fetchall()}


def validate_ext_meta(conn: Connection[Any], domain: str, ext_meta: dict[str, Any] | None) -> None:
    """write path 일괄 검증 (039 키·값). 위반 시 ``ExtMetaValidationError``.

    순서: 허용 키 조회 → 등록되지 않은 키 거부 → 값 검증.

    ⚠️ **열람 등급은 여기서 검사하지 않는다** — 등급은 읽기 경로에서 집행한다. 쓰기에서
    막으면 등급 정책이 바뀔 때 이미 저장된 데이터가 규칙에 안 맞는 상태로 남는다.

    Args:
        conn: DB 연결.
        domain: 도메인. 허용 키 집합이 도메인마다 다르다.
        ext_meta: 검사할 확장 메타.

    Raises:
        ExtMetaValidationError: 등록되지 않은 키가 있거나 값이 스키마를 어겼을 때.

    ⚠️ 그 도메인에 등록된 키가 **하나도 없으면 검증을 건너뛴다** — 시드가 안 된 도메인에서
    게이트가 통째로 무력해지므로, 새 도메인을 열 때 키 등록을 빠뜨리지 말 것.
    """
    allowed = fetch_allowed_ext_keys(conn, domain)
    if not allowed:
        # 시드 미등록 도메인은 검증 생략 — 키 미등록 시 게이트 무력화 주의(040 US3).
        return
    violations = sorted(k for k in (ext_meta or {}) if k not in allowed)
    if violations:
        raise ExtMetaValidationError(f"미등록 ext_meta 키(domain={domain}): {violations}")
    schemas = fetch_ext_key_schemas(conn, domain)
    # 키 게이트 통과 후 값 검증(039) — tier(040)는 write path 에서 검사하지 않음.
    value_violations = check_ext_meta_values(schemas, ext_meta)
    if value_violations:
        raise ExtMetaValidationError(
            f"ext_meta 값 위반(domain={domain}): {value_violations}"
        )
