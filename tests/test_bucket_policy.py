"""045 Phase A — apply_bucket_policy 단위·동작 동일 검증."""

from __future__ import annotations

import copy
import unittest
from typing import Any
from unittest import mock

from src.search.bucket_policy import apply_bucket_policy
from src.search.opensearch_search import cut_rows, passes_cutoff, rerank_reorder
from src.search.query_plan import SearchPolicy
from src.search.search_tuning import SearchTuning


def _row(
    asset_id: str,
    *,
    bm25: bool = False,
    cos: float = 0.5,
    similarity: float = 0.5,
    matched_queries: list[str] | None = None,
) -> dict[str, Any]:
    r: dict[str, Any] = {
        "id": asset_id,
        "file_uri": "",
        "modality": "text",
        "domain_label": "general",
        "summary": f"요약 {asset_id}",
        "similarity": similarity,
        "_cos": cos,
    }
    if bm25:
        r["_bm25"] = True
    if matched_queries is not None:
        r["matched_queries"] = matched_queries
    return r


def _policy(*, rescue: str = "restricted") -> SearchPolicy:
    return SearchPolicy(
        content_query="테스트",
        lexical_rescue=rescue,  # type: ignore[arg-type]
        generic_single_term=True,
        mode="auto",
    )


class ApplyBucketPolicyTest(unittest.TestCase):
    # 069 US-E(FR-E5②): apply_bucket_policy 튜닝 12종이 SearchTuning 한 묶음으로 축소됐다. 개별 테스트는
    # 종전대로 cutoff_enabled=… 같은 kw 를 넘기고, 이 헬퍼가 튜닝 필드/나머지로 분리해 SearchTuning 을
    # 조립한다(테스트 메서드 무변경). _TUNING_FIELDS 에 없는 kw(top·baseline·k·rerank_fn·policy·주입 fn)는
    # 그대로 개별 인자로 전달.
    _TUNING_FIELDS = {
        "weights", "cutoff_enabled", "cutoff_eps", "cutoff_floor", "result_floor",
        "bm25_operator", "rerank_enabled", "rerank_top_r", "rerank_tau", "rerank_model",
        "about_filter_enabled", "evidence_rescue_enabled", "evidence_debug",
    }

    def _call(self, fused: list[dict], **kw: Any):
        tuning_kw: dict[str, Any] = {
            "cutoff_enabled": True,
            "cutoff_eps": 0.17,
            "cutoff_floor": 0.50,
            "result_floor": 0.40,
            "bm25_operator": "and",
            "rerank_enabled": False,
            "rerank_top_r": 10,
            "rerank_tau": 0.0,
            "rerank_model": "cross-encoder",
            "evidence_rescue_enabled": False,
            "evidence_debug": False,
        }
        rest: dict[str, Any] = {
            "query": "테스트",
            "top": 0.85,
            "baseline": 0.30,
            "k": 20,
            "rerank_fn": None,
            "policy": _policy(),
            "passes_cutoff_fn": passes_cutoff,
            "cut_rows_fn": cut_rows,
            "rerank_reorder_fn": rerank_reorder,
        }
        for key, val in kw.items():
            (tuning_kw if key in self._TUNING_FIELDS else rest)[key] = val
        return apply_bucket_policy(fused, tuning=SearchTuning(**tuning_kw), **rest)

    def test_cutoff_disabled_passthrough(self) -> None:
        fused = [_row("a1"), _row("a2", cos=0.1, similarity=0.1)]
        out = self._call(fused, cutoff_enabled=False)
        self.assertTrue(out.gate_passed)
        self.assertEqual(out.cut_count, 0)
        self.assertEqual([r["id"] for r in out.rows], ["a1", "a2"])
        for r in out.rows:
            self.assertNotIn("_cos", r)
            self.assertNotIn("_bm25", r)

    def test_gate_passed_cut_rows(self) -> None:
        fused = [
            _row("hi", cos=0.80, similarity=0.8, bm25=True),
            _row("lo", cos=0.30, similarity=0.3),
        ]
        out = self._call(fused, top=0.85, baseline=0.30, result_floor=0.40)
        self.assertTrue(out.gate_passed)
        self.assertEqual([r["id"] for r in out.rows], ["hi"])
        self.assertEqual(out.cut_count, 1)

    def test_gate_passed_rerank_augment(self) -> None:
        fused = [
            _row("r1", cos=0.90, similarity=0.9, bm25=True),
            _row("r2", cos=0.85, similarity=0.85, bm25=True),
        ]

        def fake_rerank(_q: str, _texts: list[str], *, model_name: str) -> list[float]:
            return [0.2, 0.9]

        def fake_reorder(rows, query, rerank_fn, *, top_r, tau, model_name):
            scores = fake_rerank(query, [], model_name=model_name)
            pairs = list(zip(list(rows), scores, strict=True))
            pairs.sort(key=lambda p: -p[1])
            return [p[0] for p in pairs], scores

        out = self._call(
            fused,
            rerank_enabled=True,
            rerank_reorder_fn=fake_reorder,
            rerank_fn=fake_rerank,
        )
        self.assertTrue(out.gate_passed)
        self.assertEqual([r["id"] for r in out.rows], ["r2", "r1"])
        self.assertIsNotNone(out.rerank)
        self.assertEqual(out.rerank["kept"], 2)

    def test_gate_fail_lexical_rescue_legacy_and_restricted(self) -> None:
        weak = _row(
            "w1",
            bm25=True,
            matched_queries=["hit_summary", "hit_cross_meta"],
        )
        # RESCUE off → legacy keep
        out_off = self._call(
            [weak],
            top=0.40,
            baseline=0.39,
            cutoff_eps=0.9,
            cutoff_floor=0.9,
            evidence_rescue_enabled=False,
        )
        self.assertFalse(out_off.gate_passed)
        self.assertEqual([r["id"] for r in out_off.rows], ["w1"])

        # RESCUE on + restricted → weak drop
        out_on = self._call(
            [weak],
            top=0.40,
            baseline=0.39,
            cutoff_eps=0.9,
            cutoff_floor=0.9,
            evidence_rescue_enabled=True,
        )
        self.assertEqual(out_on.rows, [])

        strong = _row("k1", bm25=True, matched_queries=["hit_keywords"])
        out_strong = self._call(
            [strong],
            top=0.40,
            baseline=0.39,
            cutoff_eps=0.9,
            cutoff_floor=0.9,
            evidence_rescue_enabled=True,
        )
        self.assertEqual([r["id"] for r in out_strong.rows], ["k1"])

    def test_gate_fail_no_lexical_empty(self) -> None:
        fused = [_row("n1", cos=0.35, similarity=0.35)]
        out = self._call(
            fused,
            top=0.40,
            baseline=0.39,
            cutoff_eps=0.9,
            cutoff_floor=0.9,
        )
        self.assertFalse(out.gate_passed)
        self.assertFalse(out.lexical_evidence)
        self.assertEqual(out.rows, [])
        self.assertGreater(out.cut_count, 0)

    def test_evidence_debug_meta(self) -> None:
        fused = [_row("d1", bm25=True, matched_queries=["hit_summary"])]
        out = self._call(
            fused,
            top=0.40,
            baseline=0.39,
            cutoff_eps=0.9,
            cutoff_floor=0.9,
            evidence_debug=True,
        )
        row = out.rows[0]
        self.assertIn("evidence_score", row)
        self.assertIn("strong_evidence_score", row)
        self.assertIn("keep_reason", row)
        self.assertFalse(row["gate_passed"])

    def test_k_limit_applied(self) -> None:
        fused = [_row(f"x{i}", cos=0.9 - i * 0.01, similarity=0.9 - i * 0.01) for i in range(5)]
        out = self._call(fused, k=2, cutoff_enabled=False)
        self.assertEqual(len(out.rows), 2)


class BucketPolicyLegacyInlineParityTest(unittest.TestCase):
    """인라인 레거시 분기와 ``apply_bucket_policy`` 결과가 동일한지 고정."""

    def _legacy_inline(
        self,
        fused: list[dict[str, Any]],
        *,
        query: str,
        k: int,
        cutoff_enabled: bool,
        cutoff_eps: float,
        cutoff_floor: float,
        result_floor: float,
        bm25_operator: str,
        rerank_enabled: bool,
        rerank_fn,
        rerank_top_r: int,
        rerank_tau: float,
        rerank_model: str,
        policy: SearchPolicy,
        evidence_rescue_enabled: bool,
        evidence_debug: bool,
        top: float,
        baseline: float,
    ) -> dict[str, Any]:
        """045 이전 ``search_assets_os`` L627–697 동형 복제(회귀 고정용)."""
        from src.search.query_evidence import (
            evidence_score,
            lexical_rescue_keep,
            strong_evidence_score,
        )

        has_lexical = any(r.get("_bm25") for r in fused)
        rerank_info = None
        if not cutoff_enabled:
            kept = fused
            gate_passed = True
            cut_count = 0
        else:
            gate_passed = passes_cutoff(top, baseline, eps=cutoff_eps, floor=cutoff_floor)
            if gate_passed:
                kept = cut_rows(fused, result_floor=result_floor)
                cut_count = len(fused) - len(kept)
                if rerank_enabled and kept:
                    kept, scores = rerank_reorder(
                        kept,
                        query,
                        rerank_fn,
                        top_r=rerank_top_r,
                        tau=rerank_tau,
                        model_name=rerank_model,
                    )
                    cut_count = len(fused) - len(kept)
                    rerank_info = {
                        "enabled": True,
                        "scored": len(scores),
                        "kept": len(kept),
                        "top": (max(scores) if scores else 0.0),
                    }
            elif has_lexical and bm25_operator == "and":
                kept = []
                for row in fused:
                    if not row.get("_bm25"):
                        continue
                    keep, reason = lexical_rescue_keep(
                        row.get("matched_queries"),
                        policy=policy,
                        rescue_enabled=evidence_rescue_enabled,
                    )
                    if keep:
                        tagged = dict(row)
                        tagged["_keep_reason"] = reason
                        kept.append(tagged)
                cut_count = len(fused) - len(kept)
            else:
                kept = []
                cut_count = len(fused) - len(kept)

        clean: list[dict[str, Any]] = []
        for row in kept[: int(k)]:
            out_row = {
                key: val
                for key, val in row.items()
                if key not in ("_cos", "_bm25", "_rrtext", "_keep_reason")
            }
            if evidence_debug:
                mq = out_row.get("matched_queries")
                out_row["evidence_score"] = evidence_score(mq)
                out_row["strong_evidence_score"] = strong_evidence_score(mq)
                out_row["gate_passed"] = gate_passed
                out_row["keep_reason"] = row.get("_keep_reason") or (
                    "gate_passed" if gate_passed else "unknown"
                )
            clean.append(out_row)

        return {
            "rows": clean,
            "gate_passed": gate_passed,
            "lexical_evidence": has_lexical,
            "cut_count": cut_count,
            "rerank": rerank_info,
        }

    def _assert_parity(self, fused: list[dict], **kw: Any) -> None:
        fused_copy = copy.deepcopy(fused)
        policy = kw.get("policy") or _policy()
        top = kw.get("top", 0.85)
        baseline = kw.get("baseline", 0.30)
        legacy = self._legacy_inline(
            copy.deepcopy(fused_copy),
            query=kw.get("query", "테스트"),
            k=kw.get("k", 20),
            cutoff_enabled=kw.get("cutoff_enabled", True),
            cutoff_eps=kw.get("cutoff_eps", 0.17),
            cutoff_floor=kw.get("cutoff_floor", 0.50),
            result_floor=kw.get("result_floor", 0.40),
            bm25_operator=kw.get("bm25_operator", "and"),
            rerank_enabled=kw.get("rerank_enabled", False),
            rerank_fn=kw.get("rerank_fn"),
            rerank_top_r=kw.get("rerank_top_r", 10),
            rerank_tau=kw.get("rerank_tau", 0.0),
            rerank_model=kw.get("rerank_model", "cross-encoder"),
            policy=policy,
            evidence_rescue_enabled=kw.get("evidence_rescue_enabled", False),
            evidence_debug=kw.get("evidence_debug", False),
            top=top,
            baseline=baseline,
        )
        tuning = SearchTuning(
            cutoff_enabled=kw.get("cutoff_enabled", True),
            cutoff_eps=kw.get("cutoff_eps", 0.17),
            cutoff_floor=kw.get("cutoff_floor", 0.50),
            result_floor=kw.get("result_floor", 0.40),
            bm25_operator=kw.get("bm25_operator", "and"),
            rerank_enabled=kw.get("rerank_enabled", False),
            rerank_top_r=kw.get("rerank_top_r", 10),
            rerank_tau=kw.get("rerank_tau", 0.0),
            rerank_model=kw.get("rerank_model", "cross-encoder"),
            evidence_rescue_enabled=kw.get("evidence_rescue_enabled", False),
            evidence_debug=kw.get("evidence_debug", False),
        )
        out = apply_bucket_policy(
            fused_copy,
            query=kw.get("query", "테스트"),
            top=top,
            baseline=baseline,
            k=kw.get("k", 20),
            tuning=tuning,
            rerank_fn=kw.get("rerank_fn"),
            policy=policy,
            passes_cutoff_fn=passes_cutoff,
            cut_rows_fn=cut_rows,
            rerank_reorder_fn=rerank_reorder,
        )
        new = {
            "rows": out.rows,
            "gate_passed": out.gate_passed,
            "lexical_evidence": out.lexical_evidence,
            "cut_count": out.cut_count,
            "rerank": out.rerank,
        }
        self.assertEqual(new, legacy)

    def test_parity_matrix(self) -> None:
        scenarios = [
            {"fused": [_row("a1"), _row("a2")], "cutoff_enabled": False},
            {
                "fused": [
                    _row("hi", cos=0.80, similarity=0.8, bm25=True),
                    _row("lo", cos=0.30, similarity=0.3),
                ],
            },
            {
                "fused": [_row("w1", bm25=True, matched_queries=["hit_summary", "hit_cross_meta"])],
                "top": 0.40,
                "baseline": 0.39,
                "cutoff_eps": 0.9,
                "cutoff_floor": 0.9,
                "evidence_rescue_enabled": True,
            },
            {
                "fused": [_row("w1", bm25=True, matched_queries=["hit_summary"])],
                "top": 0.40,
                "baseline": 0.39,
                "cutoff_eps": 0.9,
                "cutoff_floor": 0.9,
                "evidence_rescue_enabled": False,
                "evidence_debug": True,
            },
            {"fused": [_row("n1", cos=0.35)], "top": 0.40, "baseline": 0.39, "cutoff_eps": 0.9, "cutoff_floor": 0.9},
            {"fused": [_row("b1", bm25=True)], "bm25_operator": "or", "top": 0.40, "baseline": 0.39, "cutoff_eps": 0.9, "cutoff_floor": 0.9},
        ]
        for i, sc in enumerate(scenarios):
            fused = sc.pop("fused")
            with self.subTest(i=i):
                self._assert_parity(fused, **sc)

    @mock.patch("src.search.reranker.score_pairs")
    def test_parity_with_rerank(self, mock_score: mock.MagicMock) -> None:
        mock_score.return_value = [0.1, 0.9]
        fused = [
            _row("r1", cos=0.90, similarity=0.9, bm25=True),
            _row("r2", cos=0.85, similarity=0.85, bm25=True),
        ]
        self._assert_parity(fused, rerank_enabled=True)


class AboutFilterPolicyTest(unittest.TestCase):
    """073 — apply_bucket_policy 의 aboutness OR-증거 필터 배선(토글·드롭·내부키 제거)."""

    # ApplyBucketPolicyTest._call 과 동일 기본값(상속하면 부모 테스트가 중복 실행되므로 복제).
    # _call 이 self._TUNING_FIELDS 를 참조하므로 그 클래스 속성도 함께 복제한다(FR-E5② tuning 분리).
    _call = ApplyBucketPolicyTest._call
    _TUNING_FIELDS = ApplyBucketPolicyTest._TUNING_FIELDS

    def _about_rows(self) -> list[dict[str, Any]]:
        a = _row("guitar", cos=0.80, similarity=0.8, bm25=True)
        a["_about"] = ["기타"]
        a["_kwtext"] = "기타 연주법 guitar.txt"
        b = _row("violin", cos=0.70, similarity=0.7, bm25=True)
        b["_about"] = ["바이올린"]
        b["_kwtext"] = "바이올린 연주 현악기 violin.txt"
        return [a, b]

    def test_enabled_drops_no_evidence_row_and_strips_keys(self) -> None:
        # '기타 연주' 질의 — 바이올린(무증거)은 필터로 드롭, cut_count 반영, 내부키는 응답 전 제거.
        out = self._call(self._about_rows(), query="기타 연주", about_filter_enabled=True)
        self.assertEqual([r["id"] for r in out.rows], ["guitar"])
        self.assertEqual(out.cut_count, 1)
        for r in out.rows:
            self.assertNotIn("_about", r)
            self.assertNotIn("_kwtext", r)

    def test_disabled_passthrough(self) -> None:
        # 토글 off(기본) — 필터 무접촉·두 행 유지(회귀 0·SC-002). 내부키 제거는 동일.
        out = self._call(self._about_rows(), query="기타 연주")
        self.assertEqual([r["id"] for r in out.rows], ["guitar", "violin"])
        for r in out.rows:
            self.assertNotIn("_about", r)
            self.assertNotIn("_kwtext", r)

    def test_enabled_failsafe_when_no_match(self) -> None:
        # 어휘가 전혀 안 겹치는 질의 — fail-safe 로 원 행 유지(필터가 검색을 비우지 않는다).
        out = self._call(self._about_rows(), query="우주 신비", about_filter_enabled=True)
        self.assertEqual([r["id"] for r in out.rows], ["guitar", "violin"])


if __name__ == "__main__":
    unittest.main()
