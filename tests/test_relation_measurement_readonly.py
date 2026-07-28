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

# 측정 경로가 **불러서는 안 되는** 쓰기 함수들.
# 왜 위 _TARGETS 확장으로 해결되지 않는가: 측정 스크립트가 import 하는
# `src/relations/relation_type_catalog.py` 에는 정당한 쓰기 함수
# (`ensure_relation_kind_for_llm_proposal` — `INSERT INTO relation_kind`, 운영 propose 경로용)가
# **있어야 한다**. 그 파일을 스캔 대상에 넣으면 테스트가 정당한 코드를 잡아 red 가 된다.
# 그래서 "파일에 쓰기 SQL 이 있나"가 아니라 **"측정 경로가 그 함수를 부르나"** 를 본다.
_FORBIDDEN_CALLS = frozenset({
    "ensure_relation_kind_for_llm_proposal",   # INSERT INTO relation_kind
    "sync_graph_edges",                        # graph_edge upsert(운영 영속화)
    "persist_lineage",                         # 계보 기록
    "record_resolution",                       # relation_resolution 기록
    "bulk_review",                             # 검토 결과 반영(status 변경)
    "revise_edge",                             # 엣지 정정
})
_MEASURE_SCRIPTS = ["scripts/judge_relations.py", "scripts/measure_relation_quality.py"]


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

    def test_측정_경로가_쓰기_함수를_부르지_않는다(self):
        """리터럴 스캔이 못 잡는 구멍을 막는다 — 남의 쓰기 함수를 호출하는 경우.

        측정 스크립트 자체에 쓰기 SQL 문자열이 없어도, ``ensure_relation_kind_for_llm_proposal``
        처럼 **다른 모듈의 쓰기 함수**를 부르면 DB 가 바뀐다. import 와 호출 양쪽을 본다.
        """
        for rel in _MEASURE_SCRIPTS:
            path = _ROOT / rel
            with self.subTest(file=rel):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    # ① `from … import ensure_relation_kind_for_llm_proposal` 형태
                    if isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            self.assertNotIn(
                                alias.name, _FORBIDDEN_CALLS,
                                f"{rel}:{node.lineno} 가 쓰기 함수 {alias.name!r} 를 import 한다")
                    # ② 이름으로 직접 호출 / 속성 접근 호출
                    if isinstance(node, ast.Call):
                        fn = node.func
                        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                        self.assertNotIn(
                            name, _FORBIDDEN_CALLS,
                            f"{rel}:{node.lineno} 가 쓰기 함수 {name!r} 를 호출한다")


if __name__ == "__main__":
    unittest.main()
