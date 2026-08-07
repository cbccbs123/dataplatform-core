"""관계 승인·노출 정책의 순수 판정 — 세 소비처(영속화·검토큐·노출)가 공유하는 단일 정본.

**가장 중요한 봉인**은 `test_노출_하한이_폐기_임계와_같다` 다 — 영속화 판정과 노출 판정이 어긋나면
"행은 만들어지는데 화면에 안 보이는" 유령 구간이 생기고, 그건 조용히 생겨서 아무도 모른다.
"""
import unittest

from src.relations.approval_policy import (
    EXPLICIT_KINDS,
    FOLD_PREFERRED_KIND,
    SIMILARITY_KINDS,
    choose_folded_edge,
    exposure_tier,
    is_auto_approvable,
    is_review_exempt,
    parse_kind_set,
    should_persist,
)


class TestKindFamilies(unittest.TestCase):
    def test_두_계열이_겹치지_않는다(self):
        self.assertEqual(SIMILARITY_KINDS & EXPLICIT_KINDS, frozenset())

    def test_활성_5종을_모두_덮는다(self):
        self.assertEqual(
            SIMILARITY_KINDS | EXPLICIT_KINDS,
            {"duplicate_near", "same_domain", "references", "derived_from", "same_series"})


class TestShouldPersist(unittest.TestCase):
    def test_유사도_계열_저신뢰는_행을_만들지_않는다(self):
        self.assertFalse(should_persist("same_domain", 0.6, min_conf_similarity=0.75))
        self.assertFalse(should_persist("duplicate_near", 0.74, min_conf_similarity=0.75))

    def test_경계값은_통과다(self):
        self.assertTrue(should_persist("same_domain", 0.75, min_conf_similarity=0.75))

    def test_명시적_계열은_저신뢰도_면제다(self):
        # path_signal 부활 시 저신뢰 명시적 제안이 정당하게 나온다 — 버리면 그때 손실이다.
        self.assertTrue(should_persist("references", 0.3, min_conf_similarity=0.75))
        self.assertTrue(should_persist("derived_from", 0.1, min_conf_similarity=0.75))
        self.assertTrue(should_persist("same_series", None, min_conf_similarity=0.75))

    def test_신뢰도_없음은_유사도_계열에서_폐기다(self):
        self.assertFalse(should_persist("same_domain", None, min_conf_similarity=0.75))

    def test_임계_0이면_게이트를_끈다(self):
        # 롤백이 코드 revert 가 아니라 설정 변경이어야 한다.
        self.assertTrue(should_persist("same_domain", 0.1, min_conf_similarity=0.0))
        self.assertTrue(should_persist("same_domain", None, min_conf_similarity=0.0))

    def test_모르는_kind는_보수적으로_유지한다(self):
        # 통제어휘가 늘어났을 때(promote_relation_kind) 조용히 버리면 데이터 손실이다.
        self.assertTrue(should_persist("brand_new_kind", 0.1, min_conf_similarity=0.75))

    def test_대소문자를_가리지_않는다(self):
        self.assertFalse(should_persist("SAME_DOMAIN", 0.6, min_conf_similarity=0.75))


class TestAutoApprovable(unittest.TestCase):
    def test_제외_목록의_kind는_자동승인_불가다(self):
        self.assertFalse(
            is_auto_approvable("same_domain", exclude_kinds=frozenset({"same_domain"})))

    def test_그_외는_자동승인_자격이_있다(self):
        self.assertTrue(
            is_auto_approvable("duplicate_near", exclude_kinds=frozenset({"same_domain"})))

    def test_제외_목록이_비면_전부_자격이_있다(self):
        self.assertTrue(is_auto_approvable("same_domain", exclude_kinds=frozenset()))

    def test_대소문자를_가리지_않는다(self):
        self.assertFalse(
            is_auto_approvable("Same_Domain", exclude_kinds=frozenset({"same_domain"})))


class TestExposureTier(unittest.TestCase):
    def test_active는_강칸이다(self):
        self.assertEqual(
            exposure_tier("active", "same_domain", 0.95, min_conf_similarity=0.75), "strong")

    def test_active는_저신뢰여도_강칸이다(self):
        # 이미 승인된 관계를 노출 하한으로 되돌려 숨기면 사람의 결정을 무시하는 것이다.
        self.assertEqual(
            exposure_tier("active", "same_domain", 0.1, min_conf_similarity=0.75), "strong")

    def test_proposed_고신뢰는_약칸이다(self):
        self.assertEqual(
            exposure_tier("proposed", "same_domain", 0.9, min_conf_similarity=0.75), "weak")

    def test_proposed_저신뢰는_노출하지_않는다(self):
        self.assertIsNone(
            exposure_tier("proposed", "same_domain", 0.6, min_conf_similarity=0.75))

    def test_rejected는_노출하지_않는다(self):
        # 사람이 아니라고 판단한 것을 약칸으로 되살리면 검토가 무의미해진다.
        self.assertIsNone(
            exposure_tier("rejected", "duplicate_near", 0.99, min_conf_similarity=0.75))

    def test_expired도_노출하지_않는다(self):
        self.assertIsNone(
            exposure_tier("expired", "duplicate_near", 0.99, min_conf_similarity=0.75))

    def test_노출_하한이_폐기_임계와_같다(self):
        # 🔴 어긋나면 "만들지만 안 보이는" 유령 구간이 생긴다(spec 엣지케이스 5).
        for kind in sorted(SIMILARITY_KINDS | EXPLICIT_KINDS):
            for conf in (None, 0.0, 0.70, 0.749, 0.75, 0.9, 1.0):
                persisted = should_persist(kind, conf, min_conf_similarity=0.75)
                shown = exposure_tier("proposed", kind, conf,
                                      min_conf_similarity=0.75) is not None
                self.assertEqual(
                    persisted, shown,
                    f"kind={kind} conf={conf} 에서 영속화·노출 판정이 어긋난다")


class TestParseKindSet(unittest.TestCase):
    def test_쉼표_구분_문자열을_읽는다(self):
        self.assertEqual(
            parse_kind_set("same_domain, duplicate_near", default=frozenset()),
            frozenset({"same_domain", "duplicate_near"}))

    def test_None_은_기본값이다(self):
        self.assertEqual(parse_kind_set(None, default=frozenset({"same_domain"})),
                         frozenset({"same_domain"}))

    def test_빈_문자열은_빈_집합이다(self):
        # 명시적으로 "" 를 주는 것은 "게이트를 끈다"는 뜻이다 — 기본값으로 되돌리면 끌 수 없다.
        self.assertEqual(parse_kind_set("", default=frozenset({"same_domain"})), frozenset())

    def test_공백만_있어도_빈_집합이다(self):
        self.assertEqual(parse_kind_set("   ", default=frozenset({"same_domain"})),
                         frozenset())

    def test_대소문자_공백_빈항목을_정규화한다(self):
        self.assertEqual(parse_kind_set(" SAME_DOMAIN ,, ", default=frozenset()),
                         frozenset({"same_domain"}))


class TestReviewExempt(unittest.TestCase):
    def test_면제_kind는_검토_대상이_아니다(self):
        self.assertTrue(
            is_review_exempt("same_domain", exempt_kinds=frozenset({"same_domain"})))

    def test_면제_목록이_비면_전부_검토_대상이다(self):
        self.assertFalse(is_review_exempt("same_domain", exempt_kinds=frozenset()))

    def test_대소문자를_가리지_않는다(self):
        self.assertTrue(
            is_review_exempt("SAME_DOMAIN", exempt_kinds=frozenset({"same_domain"})))


def _edge(kind: str, conf: float | None, tier: str = "weak") -> dict:
    """접기 테스트용 최소 엣지 — 판정에 쓰는 세 키만 담는다."""
    return {"kind_code": kind, "confidence": conf, "tier": tier}


class TestChooseFoldedEdge(unittest.TestCase):
    """동시보유 접기(081 조각⑤) — 규칙과 **그 근거**를 함께 봉인한다.

    이 테스트가 지키려는 것은 동작만이 아니다. `same_domain` 우선은 직관에 반해 보여
    (점수가 낮은데 이긴다) 되돌려지기 쉽다 — 실측 근거는 `approval_policy` 상단 주석에
    있고, 여기서는 **그 결론이 코드에 살아 있는지**를 확인한다.
    """

    def test_엣지가_하나면_그대로_둔다(self):
        self.assertEqual(choose_folded_edge([_edge("same_domain", 0.9)]), (0, []))

    def test_빈_입력은_오류다(self):
        # 조용히 넘기면 호출부의 그룹 구성 버그가 숨는다.
        with self.assertRaises(ValueError):
            choose_folded_edge([])

    # ── 규칙 2: 조합 규칙이 점수를 이긴다 (실측 근거 — 위 클래스 docstring)

    def test_동점이면_same_domain을_남긴다(self):
        """v4 실측 77쌍의 경우. **spec 081 원안(동점이면 duplicate_near)을 뒤집은 지점**이다.

        원안대로면 정확도 48.3% 인 `duplicate_near` 를 남기고 92.4% 인 `same_domain` 을 버린다.
        """
        keep, folded = choose_folded_edge([_edge("duplicate_near", 0.7), _edge("same_domain", 0.7)])
        self.assertEqual(keep, 1)
        self.assertEqual(folded, ["duplicate_near"])

    def test_dup이_점수가_높아도_same_domain을_남긴다(self):
        """v4 실측 15쌍(dup@0.9+sd@0.9)·2쌍(dup@0.9+sd@0.7)의 경우.

        `duplicate_near` 0.9 는 **단독일 때만** 88.9% 다 — `same_domain` 과 같이 붙은
        15쌍에서는 3/13 = 23% 였다. 이름표가 둘 달렸다는 사실 자체가 판정이 흔들렸다는
        신호이므로 그 상황에서는 점수를 신호로 쓰지 않는다.
        """
        keep, folded = choose_folded_edge([_edge("duplicate_near", 0.9), _edge("same_domain", 0.7)])
        self.assertEqual(keep, 1)
        self.assertEqual(folded, ["duplicate_near"])

    def test_입력_순서가_바뀌어도_같은_것을_남긴다(self):
        keep, folded = choose_folded_edge([_edge("same_domain", 0.7), _edge("duplicate_near", 0.9)])
        self.assertEqual(keep, 0)
        self.assertEqual(folded, ["duplicate_near"])

    # ── 규칙 1: 사람의 결정이 조합 규칙보다 앞선다

    def test_강칸이_조합규칙을_이긴다(self):
        # 사람이 승인한 것(active→strong)을 기계 규칙으로 버리면 그 결정을 무시하는 것이다.
        keep, folded = choose_folded_edge([
            _edge("duplicate_near", 0.7, tier="strong"),
            _edge("same_domain", 0.9, tier="weak"),
        ])
        self.assertEqual(keep, 0)
        self.assertEqual(folded, ["same_domain"])

    # ── 규칙 3~5: 조합 규칙이 없을 때

    def test_규칙없는_조합은_점수가_높은_쪽을_남긴다(self):
        # 표본 1뿐이라 근거가 없는 조합 — spec 081 원안(점수 우선)을 그대로 쓴다.
        keep, folded = choose_folded_edge([_edge("duplicate_near", 0.7), _edge("same_series", 0.9)])
        self.assertEqual(keep, 1)
        self.assertEqual(folded, ["duplicate_near"])

    def test_규칙도_점수도_같으면_사전순으로_결정한다(self):
        # 근거가 있어서가 아니라 **같은 입력이면 같은 출력**이어야 하기 때문이다(헌법 3조).
        keep, folded = choose_folded_edge([_edge("references", 0.9), _edge("duplicate_near", 0.9)])
        self.assertEqual(keep, 1)   # duplicate_near < references (사전순)
        self.assertEqual(folded, ["references"])

    def test_신뢰도가_없어도_깨지지_않는다(self):
        keep, _ = choose_folded_edge([_edge("references", None), _edge("derived_from", 0.9)])
        self.assertEqual(keep, 1)

    # ── 규칙표 자체의 불변식

    def test_규칙표의_선호_종류는_그_조합에_속한다(self):
        # 조합에 없는 종류를 선호로 적으면 그 규칙은 **영원히 발동하지 않는다**(조용한 죽은 규칙).
        for kinds, preferred in FOLD_PREFERRED_KIND.items():
            self.assertIn(preferred, kinds)

    def test_규칙표는_조합_전체가_일치할_때만_적용된다(self):
        # 부분 일치로 적용하면 세 종류가 붙은 미지의 경우까지 근거 없이 판정하게 된다.
        keep, folded = choose_folded_edge([
            _edge("duplicate_near", 0.7), _edge("same_domain", 0.7), _edge("references", 0.9),
        ])
        self.assertEqual(keep, 2)   # 조합 규칙 미적용 → 점수 우선
        self.assertEqual(folded, ["duplicate_near", "same_domain"])


if __name__ == "__main__":
    unittest.main()
