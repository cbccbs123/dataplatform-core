"""077 레포 분리 G2 — 코어 경계 봉인(백엔드가 파이프라인 전용 패키지를 끌어오지 않음).

레포 분리(spec 077)의 핵심 불변식(SC-2)을 **정적 import 검사**로 봉인한다: 백엔드(``src/app/portal``·
``src/portal``)가 **직접·간접으로 결국 import 하는 모든 src 모듈** 중에 파이프라인 전용 패키지가 하나도
없어야 한다. (직접·간접 = 백엔드가 import 하는 모듈 → 그 모듈이 import 하는 모듈 → … 끝까지 따라가 모은
전체. 마치 친구의 친구의 친구까지 = 내 인맥 전체.) 있으면 백엔드 레포가 코어만 설치·참조할 수 없고(Phase 2
코어 패키징이 파이프라인 코드를 딸려온다) 두 레포 독립 개발이 깨진다.

정적(AST) 분석이라 DB·모델·실행 없이 결정적으로 판정한다 — 파일 맨 위 import 뿐 아니라 **함수 안에서 하는
import(지연 import)도** ``ast.walk`` 로 전부 수집한다. 백엔드가 파이프라인 모듈을 새로 import 하면(새 경계
위반) 이 테스트가 즉시 실패해, US-E 정지작업(E3·E6·E4)과 G1 디커플링이 지켜온 경계를 회귀로부터 보호한다.
"""
from __future__ import annotations

import ast
import pathlib
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

# 파이프라인(Airflow 처리) 전용 top-level 패키지 — 백엔드 폐포에 있으면 안 된다.
# (추출·스킬·조합층·수집/적재·분류 로직·전처리·디스패치. read seam 은 core 로 승격됨: topic·relations.graph_query 등.)
_PIPELINE_ONLY = {"extractors", "skills", "pipeline", "ingest", "classify", "preprocess", "dispatch"}


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _src_imports(path: pathlib.Path) -> set[str]:
    """모듈이 import 하는 ``src.*`` 후보(module + 함수 내부 전부·AST).

    ``from src.a.b import c`` 는 ``src.a.b``(모듈이면 그것) **와** ``src.a.b.c``(c 가 서브모듈인 경우)
    둘 다 후보로 담아, ``from src.pkg import submodule`` 형태의 서브모듈 import 를 놓치지 않는다(과대추정
    아님 — BFS 는 실제 모듈(by_name)만 따라가므로 함수/클래스명 후보는 자동 무시)."""
    out: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            out.add(node.module)
            for alias in node.names:
                out.add(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    out.add(alias.name)
    return out


def _backend_reachable_modules() -> tuple[set[str], bool]:
    """백엔드(app.portal·portal)가 직접·간접으로 결국 닿는 모든 src 모듈을 모은다(끝까지 따라감).

    반환: (도달 모듈 집합, 백엔드 루트를 찾았는지). 실제 모듈(파일)만 따라가므로 함수·클래스명 후보는
    자동 무시된다.
    """
    by_name: dict[str, pathlib.Path] = {}
    graph: dict[str, set[str]] = {}
    for p in _SRC.rglob("*.py"):
        name = _module_name(p)
        by_name[name] = p
        graph[name] = _src_imports(p)

    roots = [n for n in by_name if n.startswith("src.app.portal") or n.startswith("src.portal")]
    reached: set[str] = set()
    stack = list(roots)
    while stack:
        cur = stack.pop()
        if cur in reached:
            continue
        reached.add(cur)
        for imp in graph.get(cur, ()):
            # 실제 모듈(파일)로 존재하는 import 후보만 계속 따라간다.
            if imp in by_name and imp not in reached:
                stack.append(imp)
    return reached, bool(roots)


class TestCoreBoundary(unittest.TestCase):
    def test_backend_does_not_reach_pipeline_only_package(self) -> None:
        reached, had_roots = _backend_reachable_modules()
        self.assertTrue(had_roots, "백엔드 루트(app.portal·portal)를 찾지 못함 — 경로 확인")
        self.assertGreater(len(reached), 10, "도달 모듈이 비정상적으로 적음 — 분석 오류 의심")
        violations = sorted(
            m for m in reached
            if len(m.split(".")) >= 2 and m.split(".")[1] in _PIPELINE_ONLY
        )
        self.assertEqual(
            violations, [],
            "백엔드가 파이프라인 전용 패키지를 전이적으로 import 함(코어 경계 위반·SC-2):\n"
            + "\n".join(f"  ✗ {v}" for v in violations),
        )


if __name__ == "__main__":
    unittest.main()
