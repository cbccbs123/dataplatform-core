"""판정 라벨 모델 — 왕복·검증·집계·개인정보 화이트리스트."""
import json
import unittest

from src.relations.quality.verdicts import (
    Verdict,
    VerdictSet,
    dump_verdicts,
    error_rate,
    load_verdicts,
    verdict_counts,
)

_META = {
    "measure_id": "20260728-cohort", "method": "코호트 비교", "rubric_version": "v1",
    "rubric_text": "판정 기준…", "judge_model": "google/gemma-4-31B-it", "seed": 20260728,
    "strata": "cohort", "created_at": "2026-07-28T10:00:00+09:00",
}


def _vs(*verdicts: Verdict) -> VerdictSet:
    return VerdictSet(sample_edge_ids=tuple(v.edge_id for v in verdicts),
                      verdicts=verdicts, **_META)


class TestVerdictsRoundTrip(unittest.TestCase):
    def test_dump_load_왕복이_동일하다(self):
        vs = _vs(Verdict("e1", "strong", "같은 궁궐", "a" * 64),
                 Verdict("e2", "weak", "분야만 같음", "b" * 64, judged_by_human="weak"))
        back = load_verdicts(json.loads(json.dumps(dump_verdicts(vs))))
        self.assertEqual(back, vs)

    def test_알_수_없는_판정값은_거부한다(self):
        bad = dump_verdicts(_vs(Verdict("e1", "strong", "x", "a" * 64)))
        bad["verdicts"][0]["verdict"] = "maybe"
        with self.assertRaises(ValueError):
            load_verdicts(bad)

    def test_version이_1이_아니면_거부한다(self):
        bad = dump_verdicts(_vs(Verdict("e1", "strong", "x", "a" * 64)))
        bad["version"] = 2
        with self.assertRaises(ValueError):
            load_verdicts(bad)


class TestVerdictsPrivacy(unittest.TestCase):
    def test_판정_레코드는_화이트리스트_필드만_갖는다(self):
        # 요약 본문·파일 경로가 새어 나가면 안 된다(spec 엣지케이스 7).
        dumped = dump_verdicts(_vs(Verdict("e1", "strong", "x", "a" * 64)))
        self.assertEqual(
            set(dumped["verdicts"][0]),
            {"edge_id", "verdict", "why", "prompt_sha256", "judged_by_human"})

    def test_사유는_20자로_잘린다(self):
        v = Verdict("e1", "strong", "가" * 50, "a" * 64)
        self.assertEqual(len(dump_verdicts(_vs(v))["verdicts"][0]["why"]), 20)


class TestVerdictsAggregate(unittest.TestCase):
    def test_집계는_판정값별_건수를_센다(self):
        vs = _vs(Verdict("e1", "strong", "", "h"), Verdict("e2", "strong", "", "h"),
                 Verdict("e3", "weak", "", "h"), Verdict("e4", "error", "", "h"))
        self.assertEqual(verdict_counts(vs), {"strong": 2, "weak": 1, "error": 1})

    def test_사람_집계는_미판정을_제외한다(self):
        vs = _vs(Verdict("e1", "strong", "", "h", judged_by_human="weak"),
                 Verdict("e2", "strong", "", "h"))
        self.assertEqual(verdict_counts(vs, human=True), {"weak": 1})

    def test_error율(self):
        vs = _vs(Verdict("e1", "error", "", "h"), Verdict("e2", "strong", "", "h"))
        self.assertAlmostEqual(error_rate(vs), 0.5)

    def test_빈_묶음의_error율은_0이다(self):
        self.assertEqual(error_rate(_vs()), 0.0)


class TestVerdictsPackageExport(unittest.TestCase):
    def test_패키지에서_바로_import_된다(self):
        from src.relations.quality import (  # noqa: F401
            VERDICTS_DIR,
            Verdict,
            VerdictSet,
            dump_verdicts,
            load_verdicts,
        )

    def test_저장_경로는_골든이_아니라_fixtures다(self):
        # 판정(측정 기록)과 골든(사람이 검증한 정답)을 경로로 갈라 둔다 — 자동 승격 방지.
        # 둘 다 버전관리 제외지만(2026-08-05) 경로 분리 규율은 유지한다.
        from src.relations.quality import VERDICTS_DIR
        self.assertEqual(VERDICTS_DIR, "tests/fixtures/relations/verdicts")
        self.assertNotIn("golden", VERDICTS_DIR)


if __name__ == "__main__":
    unittest.main()
