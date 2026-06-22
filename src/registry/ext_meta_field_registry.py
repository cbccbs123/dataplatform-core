"""F-4.13 ext_meta 키·값 검증 — ``ext_meta_field_registry`` 레지스트리 기반.

도메인별 허용 ext_meta 키(``status='active'``)를 ``ext_meta_field_registry`` 에서 읽어,
``asset_metadata.ext_meta`` 의 키가 허용 집합 안인지 검증한다(키-허용목록).

값 검증은 동일 레지스트리의 ``json_schema``(JSON Schema)로 수행한다.
``type`` 키가 없거나 빈 스키마는 값 검증에서 skip 한다.

레거시 ``schema_registry`` 테이블은 main 호환을 위해 유지하나 OM 런타임은 본 테이블만 사용한다.
"""

from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row


class ExtMetaValidationError(ValueError):
    """``ext_meta`` 키 위반(미등록 키) 또는 값 위반(JSON Schema 불일치)."""


def fetch_allowed_ext_keys(conn: Connection[Any], domain: str) -> set[str]:
    """``domain`` 의 활성(status='active') ext_meta 허용 키 집합."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT meta_key FROM ext_meta_field_registry "
            "WHERE domain = %s AND status = 'active'",
            (domain,),
        )
        return {r["meta_key"] for r in cur.fetchall()}


def _schema_is_validatable(schema: dict[str, Any] | None) -> bool:
    return bool(schema and isinstance(schema, dict) and schema.get("type"))


def check_ext_meta_values(
    schemas: dict[str, dict[str, Any]],
    ext_meta: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    """``ext_meta`` 에 존재하고 ``schemas`` 에 validatable 스키마가 있는 키만 검증."""
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
    """``domain`` 의 활성 ext_meta 키→JSON Schema 맵(validatable 스키마만)."""
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
    """``domain`` 의 활성 ext_meta 키→access_tier 맵."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT meta_key, access_tier FROM ext_meta_field_registry "
            "WHERE domain = %s AND status = 'active'",
            (domain,),
        )
        return {row["meta_key"]: row["access_tier"] for row in cur.fetchall()}


def validate_ext_meta(conn: Connection[Any], domain: str, ext_meta: dict[str, Any] | None) -> None:
    """``ext_meta`` 키·값을 도메인 레지스트리 기준으로 검증. 위반 시 ``ExtMetaValidationError``."""
    allowed = fetch_allowed_ext_keys(conn, domain)
    if not allowed:
        # 시드 미등록 도메인은 검증 생략(무조건 실패 방지) — 키 미등록 시 게이트 무력화 주의
        return
    violations = sorted(k for k in (ext_meta or {}) if k not in allowed)
    if violations:
        raise ExtMetaValidationError(f"미등록 ext_meta 키(domain={domain}): {violations}")
    schemas = fetch_ext_key_schemas(conn, domain)
    value_violations = check_ext_meta_values(schemas, ext_meta)
    if value_violations:
        raise ExtMetaValidationError(
            f"ext_meta 값 위반(domain={domain}): {value_violations}"
        )
