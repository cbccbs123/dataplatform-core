"""판정 러너 — 층화 표본의 결정성과 SQL 형상."""
import unittest

from scripts.judge_relations import (
    CELL_EXPR,
    ERROR_RATE_MAX,
    SAMPLE_SQL,
    build_judge_prompt,
    group_and_sample,
    human_kappa,
    judge_one,
)

from src.relations.quality.rubric import RUBRIC_KO_V1
from src.relations.quality.verdicts import Verdict, VerdictSet


def _rows(*specs):
    """(cell, edge_id) 쌍들을 표본 행 모양으로 부풀린다."""
    return [{"cell": c, "edge_id": e, "a_name": "A", "b_name": "B"} for c, e in specs]


class TestGroupAndSample(unittest.TestCase):
    ROWS = _rows(("k1", "e1"), ("k1", "e2"), ("k1", "e3"), ("k2", "e4"))

    def test_셀마다_상한만큼_뽑는다(self):
        got = group_and_sample(self.ROWS, per_cell=2, seed=1)
        by_cell = {}
        for r in got:
            by_cell.setdefault(r["cell"], []).append(r)
        self.assertEqual(len(by_cell["k1"]), 2)
        self.assertEqual(len(by_cell["k2"]), 1)   # 풀이 작으면 전수

    def test_같은_시드는_같은_표본을_준다(self):
        self.assertEqual(group_and_sample(self.ROWS, per_cell=2, seed=7),
                         group_and_sample(self.ROWS, per_cell=2, seed=7))

    def test_다른_시드는_대체로_다른_표본을_준다(self):
        a = [r["edge_id"] for r in group_and_sample(self.ROWS, per_cell=1, seed=1)]
        b = [r["edge_id"] for r in group_and_sample(self.ROWS, per_cell=1, seed=99)]
        self.assertNotEqual(a, b)

    def test_반환은_edge_id_로_전순서_정렬된다(self):
        got = [r["edge_id"] for r in group_and_sample(self.ROWS, per_cell=3, seed=1)]
        self.assertEqual(got, sorted(got))

    def test_빈_입력(self):
        self.assertEqual(group_and_sample([], per_cell=5, seed=1), [])


class TestSampleSql(unittest.TestCase):
    def test_읽기_전용이다(self):
        for verb in ("INSERT", "UPDATE", "DELETE"):
            self.assertNotIn(verb, SAMPLE_SQL.upper())

    def test_전순서_정렬로_결정성을_확보한다(self):
        self.assertIn("ORDER BY ge.edge_id", SAMPLE_SQL)

    def test_네_층화_축을_모두_지원한다(self):
        self.assertEqual(set(CELL_EXPR), {"kind", "conf", "cohort", "kind-conf"})


_ROW = {"edge_id": "e1", "cell": "k1", "kind_code": "duplicate_near", "confidence": 0.95,
        "a_name": "경복궁.txt", "a_mod": "text", "a_topic": "역사", "a_sum": "경복궁 안내",
        "a_kw": "[]", "b_name": "창덕궁.mp4", "b_mod": "video", "b_topic": "역사",
        "b_sum": "창덕궁 후원", "b_kw": "[]"}


class TestJudgePrompt(unittest.TestCase):
    def test_시스템_판단을_판정자에게_노출하지_않는다(self):
        # 맹검의 핵심 — kind·confidence 가 새면 순환 평가가 된다.
        p = build_judge_prompt(_ROW, RUBRIC_KO_V1)
        self.assertNotIn("duplicate_near", p)
        self.assertNotIn("0.95", p)

    def test_양끝_자산_내용은_들어간다(self):
        p = build_judge_prompt(_ROW, RUBRIC_KO_V1)
        for s in ("경복궁.txt", "창덕궁.mp4", "경복궁 안내", "창덕궁 후원"):
            self.assertIn(s, p)

    def test_루브릭이_그대로_들어간다(self):
        self.assertIn(RUBRIC_KO_V1, build_judge_prompt(_ROW, RUBRIC_KO_V1))


class TestJudgeOne(unittest.TestCase):
    def test_정상_판정(self):
        v = judge_one(_ROW, rubric=RUBRIC_KO_V1,
                      llm=lambda p: {"verdict": "weak", "why": "둘 다 궁궐이나 대상이 다름"})
        self.assertEqual(v.verdict, "weak")
        self.assertEqual(v.edge_id, "e1")
        self.assertEqual(len(v.prompt_sha256), 64)

    def test_알_수_없는_판정값은_error로_떨어진다(self):
        v = judge_one(_ROW, rubric=RUBRIC_KO_V1, llm=lambda p: {"verdict": "아마도"})
        self.assertEqual(v.verdict, "error")

    def test_LLM_예외는_건별_error로_기록하고_전체를_멈추지_않는다(self):
        def boom(p):
            raise RuntimeError("settings 미초기화")
        v = judge_one(_ROW, rubric=RUBRIC_KO_V1, llm=boom)
        self.assertEqual(v.verdict, "error")
        self.assertEqual(v.why, "RuntimeError")

    def test_같은_행은_같은_프롬프트_해시를_낸다(self):
        a = judge_one(_ROW, rubric=RUBRIC_KO_V1, llm=lambda p: {"verdict": "strong"})
        b = judge_one(_ROW, rubric=RUBRIC_KO_V1, llm=lambda p: {"verdict": "strong"})
        self.assertEqual(a.prompt_sha256, b.prompt_sha256)


class TestErrorGate(unittest.TestCase):
    def test_임계는_5퍼센트다(self):
        self.assertAlmostEqual(ERROR_RATE_MAX, 0.05)


_M = {
    "measure_id": "m", "method": "", "rubric_version": "v1", "rubric_text": "",
    "judge_model": "m", "seed": 1, "strata": "kind",
    "created_at": "2026-07-28T00:00:00+09:00",
}


class TestHumanKappa(unittest.TestCase):
    def test_사람이_판정한_건만_센다(self):
        vs = VerdictSet(sample_edge_ids=("e1", "e2", "e3"), verdicts=(
            Verdict("e1", "strong", "", "h", judged_by_human="strong"),
            Verdict("e2", "weak", "", "h", judged_by_human="weak"),
            Verdict("e3", "strong", "", "h")), **_M)   # e3 는 사람 미판정
        kappa, n = human_kappa(vs)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(kappa, 1.0)

    def test_error는_비교에서_뺀다(self):
        vs = VerdictSet(sample_edge_ids=("e1",), verdicts=(
            Verdict("e1", "error", "", "h", judged_by_human="strong"),), **_M)
        self.assertEqual(human_kappa(vs), (0.0, 0))

    def test_사람_판정이_없으면_0건이다(self):
        vs = VerdictSet(sample_edge_ids=("e1",),
                        verdicts=(Verdict("e1", "strong", "", "h"),), **_M)
        self.assertEqual(human_kappa(vs), (0.0, 0))
