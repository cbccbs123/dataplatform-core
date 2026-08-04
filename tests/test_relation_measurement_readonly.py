"""측정·판정·보고 도구가 DB 에 쓰지 않음을 봉인한다(spec SC-004).

관계뿐 아니라 검색·주제 측정 도구까지 함께 덮는다 — 편입 시점에 셋 다 쓰기 0건이었고(2026-07-30 확인),
범위를 좁게 두면 "이 파일은 검사 밖"이라는 구멍이 계속 생긴다.

⚠️ "실험 전후 graph_edge 행 수가 같다" 로 검증하면 안 된다 — `dag_relations` 가 상시 도는
환경이라 실험과 무관하게 행이 늘어난다. 벽시계 비교가 아니라 **코드 경로**로 증명한다.
"""
import ast
import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# ⚠️ 측정·판정·보고 도구를 새로 만들면 **여기에 추가**한다. 빠뜨리면 그 파일만 안전망에 구멍이
# 뚫려, 나중에 누가 쓰기 함수를 실수로 넣어도 이 테스트가 통과한다(2026-07-30 리뷰 지적으로
# `judge_snapshot.py`·`report_hollow_assets.py` 누락이 발견됐다). 아래 `test_스캔_대상이_측정_
# 도구_전체를_덮는다` 가 목록 누락 자체를 잡는다.
_TARGETS = [
    "scripts/judge_relations.py",
    "scripts/judge_snapshot.py",
    "scripts/measure_relation_quality.py",
    "scripts/measure_rerank_ab.py",
    "scripts/measure_search_golden.py",
    "scripts/report_hollow_assets.py",
    "scripts/report_topic_unclassified.py",
    "scripts/review_golden_draft.py",
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
_MEASURE_SCRIPTS = [
    "scripts/judge_relations.py",
    "scripts/judge_snapshot.py",
    "scripts/measure_relation_quality.py",
    "scripts/measure_rerank_ab.py",
    "scripts/measure_search_golden.py",
    "scripts/report_hollow_assets.py",
    "scripts/report_topic_unclassified.py",
    "scripts/review_golden_draft.py",
]

# 측정·판정·보고 도구의 파일명 규칙 — 이 접두어를 가진 `scripts/*.py` 는 전부 위 목록에 있어야
# 한다. 목록을 손으로 관리하면 새 도구를 추가할 때 잊는데, 그 누락이 곧 안전망 구멍이다.
_TOOL_PREFIXES = ("measure_", "judge_", "report_", "review_")


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


class TestTargetListCoversAllTools(unittest.TestCase):
    """목록 누락 자체를 잡는다 — 안전망의 안전망.

    `_TARGETS`/`_MEASURE_SCRIPTS` 를 손으로 관리하면 새 도구를 만들 때 추가를 잊는다. 그러면
    그 파일만 검사에서 빠지는데 **테스트는 초록이라 아무도 모른다**(2026-07-30 리뷰가 실제로
    `judge_snapshot.py`·`report_hollow_assets.py` 누락을 찾아냈다).
    """

    def _tool_files(self) -> set[str]:
        return {f"scripts/{p.name}" for p in sorted((_ROOT / "scripts").glob("*.py"))
                if p.name.startswith(_TOOL_PREFIXES)}

    def test_스캔_대상이_측정_도구_전체를_덮는다(self):
        missing = sorted(self._tool_files() - set(_TARGETS))
        self.assertEqual(missing, [],
                         "측정·판정 도구인데 _TARGETS 에 없다 — 쓰기 SQL 검사에서 빠진다:\n  "
                         + "\n  ".join(missing))

    def test_쓰기함수_호출검사가_측정_도구_전체를_덮는다(self):
        missing = sorted(self._tool_files() - set(_MEASURE_SCRIPTS))
        self.assertEqual(missing, [],
                         "측정·판정 도구인데 _MEASURE_SCRIPTS 에 없다 — 금지 함수 호출 검사에서 "
                         "빠진다:\n  " + "\n  ".join(missing))

    def test_목록에_적힌_파일이_실제로_존재한다(self):
        # 파일을 지우거나 이름을 바꾸면 검사가 조용히 0건을 훑게 된다.
        for rel in (*_TARGETS, *_MEASURE_SCRIPTS):
            with self.subTest(rel):
                self.assertTrue((_ROOT / rel).is_file(), f"{rel} 없음 — 목록이 낡았다")


if __name__ == "__main__":
    unittest.main()
