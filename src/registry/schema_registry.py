"""F-4.13 ext_meta 키 검증 — ``schema_registry`` 레지스트리 기반.

도메인별 허용 ext_meta 키(``status='active'``)를 ``schema_registry`` 에서 읽어,
``asset_metadata.ext_meta`` 의 키가 허용 집합 안인지 검증한다(키-허용목록).

- 값(JSON Schema) 검증은 후속(``jsonschema`` 미선언). ``json_schema`` 컬럼은 시드만, 본 검증은 키만 본다.
- 해당 도메인 정의가 0개면 검증을 **skip**(미시드 도메인이 무조건 실패하는 부작용 차단).
  general 은 마이그레이션(v220)에서 시드되므로 정상 게이트로 동작한다.
"""

from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row


class ExtMetaValidationError(ValueError):
    """``ext_meta`` 에 도메인 레지스트리 미등록 키가 있음."""


def fetch_allowed_ext_keys(conn: Connection[Any], domain: str) -> set[str]:
    """``domain`` 의 활성(status='active') ext_meta 허용 키 집합.

    deprecated 키는 DB 에 남아도 결과에서 제외된다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT meta_key FROM schema_registry WHERE domain = %s AND status = 'active'",
            (domain,),
        )
        return {r["meta_key"] for r in cur.fetchall()}


def _schema_is_validatable(schema: dict[str, Any] | None) -> bool:
    return bool(schema and isinstance(schema, dict) and schema.get("type"))


def check_ext_meta_values(
    schemas: dict[str, dict[str, Any]],
    ext_meta: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    """``ext_meta`` 에 존재하고 ``schemas`` 에 validatable 스키마가 있는 키만 검증.

    위반은 ``(key, message)`` 튜플로 수집해 키·메시지 기준 정렬해 반환한다.
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


def validate_ext_meta(conn: Connection[Any], domain: str, ext_meta: dict[str, Any] | None) -> None:
    """``ext_meta`` 키가 도메인 허용 집합 안인지 검증. 위반 시 ``ExtMetaValidationError``.

    **함정**: 허용 키가 0개면 검증을 skip 한다(통과). 이는 새 도메인 시드 누락 시
    무조건 실패하는 부작용을 막기 위한 의도적 설계다. general 도메인은 v220
    마이그레이션에서 시드되므로 정상 게이트로 동작한다. 새 도메인 키를 추가하지 않으면
    게이트가 무력화됨에 주의.
    """
    allowed = fetch_allowed_ext_keys(conn, domain)
    if not allowed:
        # 시드 미등록 도메인은 검증 생략 — 의도적 방어 설계(위 docstring 참조)
        return
    violations = sorted(k for k in (ext_meta or {}) if k not in allowed)
    if violations:
        raise ExtMetaValidationError(f"미등록 ext_meta 키(domain={domain}): {violations}")
