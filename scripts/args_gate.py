"""파라미터 설명 게이트 — high/mid 등급 인자에 ``Args:`` 항목이 있는지 검사한다.

**흐름에서의 위치**: 각 레포 CI 가 lint 와 나란히 돌린다. 표준 라이브러리만 쓰므로 의존성
설치 없이 체크아웃 직후 실행된다(코어 ``policy_gate.py`` 와 같은 방식).

**이 게이트가 보는 것은 빈칸뿐이다.** 설명이 사실인지, 쓸모가 있는지는 판단하지 못한다 —
틀린 설명도 형식만 맞으면 통과한다. 사실 확인은 리뷰 몫이고, 이 게이트는 "새 함수가 파라미터
설명 없이 들어오는 것"만 막는다. 규약이 시간이 지나 다시 비는 것을 막는 장치다.

**3레포 공통 파일이다.** 정본은 코어 ``scripts/args_gate.py`` 이며, 파이프·백엔드에는 **그대로
복사**해 둔다(각 레포 CI 가 자기 코드를 검사해야 하는데, 코어 CI 는 다른 레포를 볼 수 없다).
판정 규칙을 바꿀 때는 코어에서 고치고 세 벌을 다시 맞춘다.

판정 규칙은 ``docs/코드_주석_규약.md`` §4.1(등급)·§2.2①(면제)를 그대로 옮긴 것이다.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

# 규약 §1 — 검사 대상 밖. 테스트는 함수명이 설명 역할을 하고, 마이그레이션은 자동 생성 관례다.
EXCLUDE_DIRS = frozenset({
    "tests", "migrations", ".claude", ".git", "__pycache__", ".venv", "venv", "build", "dist",
})

# 규약 §4.1 low — 이름과 타입만으로 자명해 설명하면 동어반복이 되는 관용 인자.
OBVIOUS_NAMES = frozenset({
    "self", "cls", "conn", "cur", "cursor", "db", "settings", "cfg", "config", "logger",
    "asset_id", "chunk_id", "edge_id", "node_id", "job_id", "run_id", "topic_id", "user_id",
})

# 규약 §4.1 high — enum 성 문자열(값 집합을 알아야 쓸 수 있는 인자).
ENUM_NAMES = frozenset({
    "mode", "kind", "strategy", "status", "state", "how", "method", "order",
    "direction", "scope", "level", "fmt", "format", "policy", "backend", "interval",
})

# 규약 §4.1 high — 숫자 임계값·한계를 뜻하는 이름 조각.
# ⚠️ "cap" 은 접미 형태(`retry_cap`)만 본다 — 맨이름 ``cap`` 은 영상 핸들(VideoCapture)이라
#    부분문자열로 잡으면 오탐이 난다(실제로 파이프에서 걸렸다).
THRESHOLD_PARTS = ("threshold", "min_", "max_", "limit", "top_k", "_k", "tau", "cutoff", "_cap")

# 규약 §4.1 high — bool 플래그 관용 접두.
FLAG_PREFIXES = ("is_", "has_", "use_", "with_", "no_", "skip_", "force_", "dry", "enable")

# 규약 §2.2① 면제 — FastAPI 가 파라미터를 선언하는 호출. 타입·기본값·범위가 OpenAPI 문서로
# 그대로 나가므로 docstring 에 같은 말을 또 쓰지 않는다(둘 중 하나는 반드시 낡는다).
FASTAPI_PARAM_CALLS = frozenset({"Query", "Path", "Body", "Header", "Cookie", "Form", "File"})

ARGS_SECTION = re.compile(r"^\s*Args:\s*$", re.M)


def _grade(arg: ast.arg, default: ast.expr | None) -> str:
    """인자 하나의 등급을 규약 §4.1 로 판정한다.

    Args:
        arg: 대상 인자 노드.
        default: 그 인자의 기본값 노드(없으면 ``None``).

    Returns:
        ``"high"``·``"mid"``·``"low"``·``"skip"``. ``skip`` 은 ``self``/``cls`` 처럼 애초에
        설명 대상이 아닌 것이다.
    """
    name = arg.arg
    if name in ("self", "cls"):
        return "skip"
    ann = ast.unparse(arg.annotation) if arg.annotation else ""

    # high — bool 플래그(타입 또는 이름). "켜면 무엇이 벌어지나"를 반드시 적어야 한다.
    if "bool" in ann or name.startswith(FLAG_PREFIXES):
        return "high"
    # high — `= None`. 없음/전체/기본값 사용 중 어느 뜻인지 코드만 봐선 모른다.
    if isinstance(default, ast.Constant) and default.value is None:
        return "high"
    # high — 주입 seam. 미주입 시 무엇이 쓰이는지가 계약이다.
    if "Callable" in ann or name.endswith(("_fn", "_client", "_factory")) or name == "client":
        return "high"
    # high — enum 성 문자열·숫자 임계값. 허용 값 집합과 단위를 알아야 쓸 수 있다.
    if name in ENUM_NAMES or any(p in name for p in THRESHOLD_PARTS):
        return "high"

    if name in OBVIOUS_NAMES:
        return "low"
    # mid — 타입힌트가 없으면 이름만으로는 무엇이 들어오는지 알 수 없다.
    if not ann:
        return "mid"
    # mid — 구조가 있는 컬렉션. 원소의 모양이 계약이다.
    if any(k in ann for k in ("list", "dict", "Sequence", "Mapping", "Iterable", "set", "tuple")):
        return "mid"
    return "low"


def _is_fastapi_param(arg: ast.arg, default: ast.expr | None) -> bool:
    """FastAPI 가 주입하는 인자인지(규약 §2.2① 면제 대상).

    두 형태를 모두 본다. 어느 쪽이든 **핸들러 docstring 이 그 인자의 설명 자리가 아니다** —
    타입·기본값·범위는 OpenAPI 로 나가고, 의존성 주입 인자는 의존성 함수 쪽이 설명을 갖는다.

    Args:
        arg: 대상 인자 노드(주석에서 ``Depends`` 를 찾는다).
        default: 인자의 기본값 노드(``Query(...)`` 등을 찾는다).

    Returns:
        ``Query(...)``·``Path(...)`` 기본값이거나 주석에 ``Depends(...)`` 가 있으면 참.
    """
    ann = ast.unparse(arg.annotation) if arg.annotation else ""
    if "Depends(" in ann:
        return True
    if not isinstance(default, ast.Call):
        return False
    fn = default.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
    return name in FASTAPI_PARAM_CALLS


def _graded_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """설명이 필요한(high·mid) 인자 이름을 모은다 — 면제 대상은 뺀다.

    Args:
        fn: 대상 함수 노드.

    Returns:
        인자 이름 목록. 비어 있으면 그 함수는 ``Args:`` 가 없어도 된다.
    """
    a = fn.args
    positional = a.posonlyargs + a.args
    # 기본값은 뒤쪽 인자부터 붙으므로 앞을 None 으로 채워 자리를 맞춘다.
    pos_defaults: list[ast.expr | None] = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    pairs = list(zip(positional, pos_defaults, strict=True))
    pairs += list(zip(a.kwonlyargs, a.kw_defaults, strict=True))

    need: list[str] = []
    for arg, default in pairs:
        if _is_fastapi_param(arg, default):
            continue
        if _grade(arg, default) in ("high", "mid"):
            need.append(arg.arg)
    return need


def _call_owner_docs(tree: ast.AST) -> dict[int, str]:
    """``__call__`` 메서드의 줄 번호 → 그 클래스의 docstring.

    규약 §2.2① 면제 — ``__call__`` 하나만 있는 ``Protocol`` 은 **클래스 docstring 이 곧 호출
    계약**이라, 메서드에 같은 설명을 다시 쓰지 않는다.

    Args:
        tree: 파싱된 모듈.

    Returns:
        ``{메서드 시작 줄: 클래스 docstring}``.
    """
    out: dict[int, str] = {}
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        doc = ast.get_docstring(cls) or ""
        for member in cls.body:
            if isinstance(member, ast.FunctionDef) and member.name == "__call__":
                out[member.lineno] = doc
    return out


def scan(root: Path, targets: list[str]) -> list[str]:
    """대상 디렉터리를 훑어 위반 목록을 만든다.

    Args:
        root: 레포 루트.
        targets: 검사할 하위 경로들(예: ``["src", "scripts"]``).

    Returns:
        ``"경로:줄 함수명 [인자…]"`` 형태의 위반 문자열 목록. 비어 있으면 통과다.
    """
    violations: list[str] = []
    for target in targets:
        base = root / target
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if any(part in EXCLUDE_DIRS for part in path.relative_to(root).parts):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                violations.append(f"{path.relative_to(root)}: 파싱 실패 — {exc}")
                continue
            owner_docs = _call_owner_docs(tree)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                need = _graded_params(node)
                if not need:
                    continue
                # __call__ 은 클래스 docstring 을 자기 계약으로 인정한다(§2.2① 면제).
                doc = ast.get_docstring(node) or owner_docs.get(node.lineno, "")
                if not ARGS_SECTION.search(doc):
                    rel = path.relative_to(root)
                    violations.append(f"{rel}:{node.lineno} {node.name}  [{', '.join(need)}]")
    return violations


def main(argv: list[str] | None = None) -> int:
    """high/mid 인자에 ``Args:`` 가 없는 함수를 찾아 차단한다.

    Args:
        argv: 명령행 인자. ``None`` 이면 실제 인자를 읽는다(테스트 주입용).

    Returns:
        0=위반 없음, 1=위반 있음(CI 가 이 값으로 차단한다).
    """
    ap = argparse.ArgumentParser(description="파라미터 설명 게이트(규약 §4)")
    ap.add_argument("targets", nargs="*", default=None,
                    help="검사할 하위 경로(미지정이면 이 레포의 기본 대상)")
    ap.add_argument("--root", default=".", help="레포 루트(기본: 현재 디렉터리)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    # 미지정이면 이 레포에 실제로 있는 패키지를 대상으로 삼는다 — 세 레포가 같은 파일을
    # 공유하므로 레포마다 기본값을 따로 두지 않는다.
    targets = args.targets or [d for d in ("src", "scripts", "processing", "service",
                                           "deploy/airflow/dags") if (root / d).exists()]
    violations = scan(root, targets)
    if violations:
        print(f"❌ 파라미터 설명 누락 {len(violations)}건 — 규약 §4 (docs/코드_주석_규약.md)")
        for v in violations:
            print(f"  {v}")
        print("\n  high/mid 등급 인자에는 ``Args:`` 항목이 필요하다.")
        print("  자명해 보이면 §4.3(쓰지 말아야 할 것)을 먼저 읽을 것 — 등급 판정이 과하면 규약을 고친다.")
        return 1
    print(f"✅ 파라미터 설명 게이트 통과(대상: {', '.join(targets)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
