"""025 — 골든셋 스키마(순수) + 코퍼스-골든 정합 가드(실DB, 게이트 ``RUN_OS_E2E``).

골든 fixture(`golden_os.json`)의 구조 불변식은 순수 단위로 항상 검증하고,
"코퍼스에 새 토픽 자산이 추가되면 골든 질의도 추가"(FR-004) 강제는 실DB 게이트로 검사한다 —
미커버 토픽이 생기면 이 테스트가 실패해 골든 질의 추가를 요구한다.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

_FX = Path(__file__).resolve().parent / "fixtures" / "search"
_RUN = os.getenv("RUN_OS_E2E") == "1"


# golden_os.json 은 **실 코퍼스 asset_id 230개**를 담고 있어 이 레포(공개)에 두지 않는다 —
# 비공개 문서 레포가 소유하고 측정 시에만 가져온다. 그래서 부재 시 skip 한다.
_GOLDEN_OS = _FX / "golden_os.json"


@unittest.skipUnless(_GOLDEN_OS.is_file(), f"golden_os.json 없음(비공개 문서 레포 소유): {_GOLDEN_OS}")
class GoldenFixtureSchemaTest(unittest.TestCase):
    """golden_os.json 의 구조 불변식 — 측정 하니스가 기대하는 계약(fixture 있을 때만)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = json.loads(_GOLDEN_OS.read_text(encoding="utf-8"))

    def test_query_categories_and_fields(self) -> None:
        qs = self.golden["queries"]
        self.assertGreaterEqual(len(qs), 59)
        for q in qs:
            self.assertIn(q["category"], {"present", "absent", "edge"})
            if q["category"] == "absent":
                self.assertTrue(q.get("expect_empty"), f"{q['id']}: absent 는 expect_empty")
                self.assertNotIn("relevant", q)
            else:
                self.assertIsInstance(q.get("relevant"), list, f"{q['id']}: relevant 필요")
                self.assertTrue(q.get("topics"), f"{q['id']}: 토픽 태그 필요(정합 가드용)")

    def test_no_match_queries_included(self) -> None:
        # 골든 최초로 no-match 질의 포함(차단율 계기판) — 24종 유지.
        absent = [q for q in self.golden["queries"] if q["category"] == "absent"]
        self.assertGreaterEqual(len(absent), 24)

    def test_provenance_fixture_exists(self) -> None:
        prov = json.loads((_FX / "golden_os.judgments.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(prov["verdicts"]), 1392)


# 게이트가 둘인 이유: 실 DB 뿐 아니라 **골든 파일도 있어야** 돈다. 파일 가드를 빼면
# `RUN_OS_E2E=1` 만으로 실행돼 `FileNotFoundError` 로 죽는다 — 부재는 정상 상태이므로
# 실패가 아니라 skip 이어야 한다. 위 클래스와 **같은 가드를 붙여 둔다**(한쪽만 붙이면 갈린다).
@unittest.skipUnless(_RUN and _GOLDEN_OS.is_file(),
                     f"RUN_OS_E2E=1 + 실 DB + 골든 파일 필요: {_GOLDEN_OS}")
class GoldenCoverageGuardTest(unittest.TestCase):
    """FR-004(실DB): 코퍼스 토픽 ↔ 골든 커버리지 정합 — 미커버 토픽이 생기면 실패한다."""

    def test_all_corpus_topics_covered(self) -> None:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env.dev", override=False)
        from src.config.settings import init_settings

        init_settings("dev")
        from src.database.postgres_util import PostgresUtil
        from src.search.golden_guard import topic_of_filename, uncovered_topics

        golden = json.loads((_FX / "golden_os.json").read_text(encoding="utf-8"))
        golden_topics: set[str] = set()
        for q in golden["queries"]:
            golden_topics.update(q.get("topics", []))

        db = PostgresUtil()
        with db, db.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT fs_path FROM asset WHERE status='registered'")
            corpus_topics = {topic_of_filename(Path(r[0]).name) for r in cur.fetchall()}

        missing = uncovered_topics(corpus_topics, golden_topics)
        self.assertEqual(
            missing, [],
            f"골든 미커버 토픽 {missing} — 신규 데이터가 추가됐다면 골든 질의를 추가하라(FR-004)",
        )


if __name__ == "__main__":
    unittest.main()
