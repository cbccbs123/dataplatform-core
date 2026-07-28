"""스냅샷 판정 도구의 순수 함수 — 쌍 키 대칭·합집합 접기·팔별 지표 (spec 079 T505).

DB·LLM 을 쓰지 않는 부분만 덮는다. 실제 판정은 G5 사람 실행에서 확인된다.
"""
from __future__ import annotations

import unittest

from scripts.judge_snapshot import collect_pairs, compare_arms, pair_key, to_judge_row

from src.relations.quality.snapshot import ProposedEdge, Snapshot, SourceSnapshot
from src.relations.quality.verdicts import Verdict, VerdictSet

_META = {
    "measure_id": "m", "method": "", "rubric_version": "v1", "rubric_text": "",
    "judge_model": "m", "seed": 0, "strata": "shadow-ab-pair",
    "created_at": "2026-07-28T00:00:00+09:00",
}


def _snap(*specs):
    """(소스, [(타깃, kind)…]) 들로 스냅샷을 만든다."""
    sources = {
        sid: SourceSnapshot(
            candidates=(),
            proposed=tuple(ProposedEdge(t, k, 0.9) for t, k in edges))
        for sid, edges in specs}
    return Snapshot(config={}, sources=sources)


class TestPairKey(unittest.TestCase):
    def test_방향이_달라도_같은_키다(self):
        # A 팔에서 x→y, B 팔에서 y→x 로 제안돼도 같은 자산 쌍이라 판정이 같아야 한다.
        self.assertEqual(pair_key("x", "y"), pair_key("y", "x"))

    def test_다른_쌍은_다른_키다(self):
        self.assertNotEqual(pair_key("x", "y"), pair_key("x", "z"))

    def test_DB_edge_id_와_형태가_다르다(self):
        # 실제 edge_id 는 UUID 단일 문자열이라 '__' 구분자가 없다.
        self.assertIn("__", pair_key("x", "y"))


class TestCollectPairs(unittest.TestCase):
    def test_제안_엣지를_쌍으로_모은다(self):
        got = collect_pairs(_snap(("s1", [("t1", "duplicate_near"), ("t2", "same_domain")])))
        self.assertEqual(set(got), {pair_key("s1", "t1"), pair_key("s1", "t2")})

    def test_같은_쌍이_여러_소스에서_나오면_하나로_접힌다(self):
        # s1→s2 와 s2→s1 이 양쪽에서 제안돼도 판정은 한 번이면 된다.
        got = collect_pairs(_snap(("s1", [("s2", "duplicate_near")]),
                                  ("s2", [("s1", "duplicate_near")])))
        self.assertEqual(len(got), 1)

    def test_제안이_없으면_빈_결과다(self):
        self.assertEqual(collect_pairs(_snap(("s1", []))), {})


class TestToJudgeRow(unittest.TestCase):
    ASSET_A = {"name": "경복궁.txt", "modality": "text", "topic": "역사",
               "summary": "경복궁 안내", "keywords": "[]"}
    ASSET_B = {"name": "창덕궁.mp4", "modality": "video", "topic": "역사",
               "summary": "창덕궁 후원", "keywords": "[]"}

    def test_build_judge_prompt_가_요구하는_키를_모두_갖춘다(self):
        from scripts.judge_relations import build_judge_prompt

        from src.relations.quality.rubric import RUBRIC_KO_V1
        row = to_judge_row("k1", self.ASSET_A, self.ASSET_B)
        p = build_judge_prompt(row, RUBRIC_KO_V1)      # KeyError 없이 조립되면 통과
        self.assertIn("경복궁.txt", p)
        self.assertIn("창덕궁 후원", p)

    def test_쌍_키가_edge_id_자리에_들어간다(self):
        self.assertEqual(to_judge_row("k1", self.ASSET_A, self.ASSET_B)["edge_id"], "k1")


class TestCompareArms(unittest.TestCase):
    def _setup(self):
        a = _snap(("s1", [("t1", "duplicate_near"), ("t2", "duplicate_near")]))
        b = _snap(("s1", [("t1", "duplicate_near"), ("t2", "same_domain")]))
        arm_pairs = {"A": set(collect_pairs(a)), "B": set(collect_pairs(b))}
        vs = VerdictSet(
            sample_edge_ids=(pair_key("s1", "t1"), pair_key("s1", "t2")),
            verdicts=(Verdict(pair_key("s1", "t1"), "strong", "", "h"),
                      Verdict(pair_key("s1", "t2"), "weak", "", "h")), **_META)
        return {"A": a, "B": b}, vs, arm_pairs

    def test_팔별_kind_분포를_낸다(self):
        snaps, vs, arm_pairs = self._setup()
        got = compare_arms(snaps, vs, arm_pairs)
        self.assertEqual(got["A"]["kind_dist"], {"duplicate_near": 2})
        self.assertEqual(got["B"]["kind_dist"], {"duplicate_near": 1, "same_domain": 1})

    def test_같은_쌍은_같은_판정을_공유한다(self):
        # 두 팔이 같은 쌍을 제안했으므로 strong 수가 같아야 한다 — 판정을 두 번 하지 않는다.
        snaps, vs, arm_pairs = self._setup()
        got = compare_arms(snaps, vs, arm_pairs)
        self.assertEqual(got["A"]["strong_count"], got["B"]["strong_count"])

    def test_자산_커버리지를_센다(self):
        snaps, vs, arm_pairs = self._setup()
        got = compare_arms(snaps, vs, arm_pairs)
        self.assertEqual(got["A"]["assets_with_edge"], 3)   # s1·t1·t2

    def test_미판정은_rated_에서_빠진다(self):
        snaps, _, arm_pairs = self._setup()
        empty = VerdictSet(sample_edge_ids=(), verdicts=(), **_META)
        got = compare_arms(snaps, empty, arm_pairs)
        self.assertEqual(got["A"]["rated"], 0)
        self.assertEqual(got["A"]["strong_count"], 0)


if __name__ == "__main__":
    unittest.main()
