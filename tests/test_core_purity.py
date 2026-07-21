"""코어 순수성 봉인(078 G1) — 코어가 파이프라인 전용 패키지에 **직접·간접으로도 닿지 않음**을 정적 검사.

레포 분리 Phase 2에서 코어(`src.*`)는 파이프라인(Airflow)과 별도 레포로 물리 분리된다. 그러려면 코어
그룹의 어떤 모듈도 파이프라인 전용 패키지를 import 하면 안 된다(import 하면 코어만 설치했을 때 깨진다).

접근: 코어 그룹의 모든 모듈에서 시작해 ``from src.* import`` / ``import src.*`` 간선을 따라 **도달 가능한
모든 src 모듈**을 AST 로 모은 뒤(전이 폐쇄), 그 집합에 파이프라인 전용 패키지가 하나도 없음을 단언한다.
새 cross-boundary 결합이 생기면 이 테스트가 즉시 실패한다(077 ``test_core_boundary`` 계승).

표준 라이브러리만 사용(ast·pathlib) — 실제 import 실행 없이 정적 분석이라 무거운 의존·DB/OS 불필요.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

# 코어 그룹(공유·설치형) — 이 패키지들의 전이 폐쇄가 파이프라인에 닿으면 안 된다.
_CORE_PKGS = {
    "config", "database", "llm", "embedders", "search",
    "relations", "topic", "registry", "domain", "file",
}
# 파이프라인 전용(코어 레포에 없어야 함) — 코어 폐쇄가 이 중 하나라도 포함하면 위반.
_PIPELINE_ONLY = {
    "app", "skills", "extractors", "classify", "pipeline",
    "ingest", "dispatch", "preprocess",
}


def _module_name(path: Path) -> str:
    """src/a/b.py → 'src.a.b', src/a/__init__.py → 'src.a' (내부 import 해석용)."""
    rel = path.relative_to(_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _src_imports(path: Path) -> set[str]:
    """파일이 import 하는 내부(src.*) 모듈명 집합(AST). ``from src.a.b import c``는 src.a.b 와
    src.a.b.c(서브모듈일 수 있음) 둘 다 후보로 넣어 파일 존재로 해석한다."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("src"):
                out.add(node.module)
                for alias in node.names:
                    out.add(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src"):
                    out.add(alias.name)
    return out


def _resolve(mod: str) -> Path | None:
    """'src.a.b' → 파일 경로(모듈 .py 또는 패키지 __init__.py). 없으면 None."""
    rel = Path(*mod.split("."))
    cand = _ROOT / rel.with_suffix(".py")
    if cand.is_file():
        return cand
    pkg = _ROOT / rel / "__init__.py"
    if pkg.is_file():
        return pkg
    return None


def _top_pkg(mod: str) -> str | None:
    """'src.search.x' → 'search'(src 바로 아래 최상위 패키지)."""
    parts = mod.split(".")
    return parts[1] if len(parts) >= 2 and parts[0] == "src" else None


def _core_reachable() -> set[str]:
    """코어 그룹 전 모듈에서 시작해 src.* import 를 따라 도달 가능한 모든 모듈(전이 폐쇄)."""
    seen: set[str] = set()
    frontier: list[Path] = []
    for pkg in _CORE_PKGS:
        for p in (_SRC / pkg).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            frontier.append(p)
    while frontier:
        path = frontier.pop()
        mod = _module_name(path)
        if mod in seen:
            continue
        seen.add(mod)
        for imp in _src_imports(path):
            resolved = _resolve(imp)
            if resolved is not None and _module_name(resolved) not in seen:
                frontier.append(resolved)
    return seen


class CorePurityTest(unittest.TestCase):
    def test_core_does_not_reach_pipeline(self):
        reachable = _core_reachable()
        self.assertTrue(reachable, "코어 모듈을 하나도 수집하지 못함(경로/구조 확인)")
        violations = sorted(
            m for m in reachable if _top_pkg(m) in _PIPELINE_ONLY
        )
        self.assertEqual(
            violations, [],
            "코어가 파이프라인 전용 패키지에 도달(직접·간접) — 물리 분리 불가:\n  "
            + "\n  ".join(violations),
        )

    def test_pipeline_only_and_core_are_disjoint(self):
        """구성 자체 sanity — 코어/파이프라인 패키지 집합이 겹치지 않는다(오분류 가드)."""
        self.assertEqual(_CORE_PKGS & _PIPELINE_ONLY, set())


if __name__ == "__main__":
    unittest.main()
