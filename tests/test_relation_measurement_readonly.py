"""측정 경로가 DB 에 쓰지 않음을 봉인한다(spec SC-004).

⚠️ "실험 전후 graph_edge 행 수가 같다" 로 검증하면 안 된다 — `dag_relations` 가 상시 도는
환경이라 실험과 무관하게 행이 늘어난다. 벽시계 비교가 아니라 **코드 경로**로 증명한다.
"""
import ast
import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TARGETS = [
    "scripts/judge_relations.py",
    "scripts/measure_relation_quality.py",
    "src/relations/quality/verdicts.py",
    "src/relations/quality/metrics.py",
    "src/relations/quality/rubric.py",
]
_WRITE = re.compile(r"\b(INSERT\s+INTO|UPDATE\s+\w|DELETE\s+FROM|TRUNCATE|DROP\s+|ALTER\s+)",
                    re.IGNORECASE)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """docstring 인 문자열 노드의 id — 설명문의 'INSERT 하지 않는다' 같은 문장은 봐준다."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                out.add(id(body[0].value))
    return out


class TestMeasurementIsReadOnly(unittest.TestCase):
    def test_측정_경로에_쓰기_SQL_이_없다(self):
        for rel in _TARGETS:
            path = _ROOT / rel
            with self.subTest(file=rel):
                self.assertTrue(path.is_file(), f"{rel} 이(가) 없다")
                tree = ast.parse(path.read_text(encoding="utf-8"))
                skip = _docstring_nodes(tree)
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                            and id(node) not in skip):
                        hit = _WRITE.search(node.value)
                        self.assertIsNone(
                            hit, f"{rel}:{node.lineno} 에 쓰기 SQL 로 보이는 문자열: "
                                 f"{hit.group(0) if hit else ''!r}")


if __name__ == "__main__":
    unittest.main()
