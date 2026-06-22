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
    """``type`` 이 있는 JSON Schema 만 값 검증 대상(039 — 빈 스키마는 skip)."""
    return bool(schema and isinstance(schema, dict) and schema.get("type"))


def check_ext_meta_values(
    schemas: dict[str, dict[str, Any]],
    ext_meta: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    """ext_meta **값** 검증 (039).

    ``ext_meta`` 에 존재하고 ``schemas`` 에 validatable 스키마가 있는 키만 Draft202012 검증.
    결정성: 위반 목록은 (key, message) 로 정렬해 반환.
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

    순서: 허용 키 집합 → 미등록 키 거부 → JSON Schema 값 검증.
    tier(access_tier)는 **검사하지 않음** — 노출 등급은 read path(042)에서만 집행.
    """
    allowed = fetch_allowed_ext_keys(conn, domain)
    if not allowed:
        # 시드 미등록 도메인은 검증 생략 — 키 미등록 시 게이트 무력화 주의(040 US3).
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
