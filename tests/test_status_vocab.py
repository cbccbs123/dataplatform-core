"""닫힌 어휘 StrEnum ↔ DB CHECK 동기 검증.

**왜 이 파일이 있나**: 어휘(StrEnum)와 DB CHECK 는 서로 모르는 두 곳에 적혀 있다. 한쪽만 고치면
런타임에야 터지고(그 값으로 UPDATE 하는 순간 CHECK 위반), 그 전까지는 조용하다. 실제로 그 사고가
났다 — `approval_policy` 의 종결 상태 집합이 DB 가 모르는 값을 담고 있었다.

⚠️ **이 가드는 한동안 없었다.** 코어 전용화 때 파이프로 옮겨간 모듈(`AssetStatus`)을 import 하고
있어 파일이 통째로 삭제됐고, 그 뒤로 어휘가 어긋나도 아무도 잡지 못했다. **코어에 남은 어휘만으로**
되살렸다 — `AssetStatus` 는 파이프 소관이라 여기서 검사하지 않는다.

여기 적힌 문자열 집합이 **DDL 의 사본**이라는 점이 요점이다. 어휘를 늘릴 때 이 테스트가 같이
빨개져야 "DB CHECK 도 고쳤나"를 되묻게 된다.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from src.domain.status_vocab import (
    AccessTier,
    GraphEdgeStatus,
    RegistryFieldStatus,
    RelationResolutionStatus,
)

_SQL_DIR = Path(__file__).resolve().parents[1] / "migrations" / "sql"


class StatusVocabSyncTest(unittest.TestCase):
    """마이그레이션 CHECK IN 목록과 StrEnum 값 집합이 일치해야 한다."""

    def test_access_tier_matches_check(self):
        self.assertEqual(
            frozenset(AccessTier),
            frozenset(("public", "authenticated", "authorized", "regulated")),
        )

    def test_graph_edge_status_matches_check(self):
        self.assertEqual(
            frozenset(GraphEdgeStatus),
            frozenset(("proposed", "active", "rejected")),
        )

    def test_relation_resolution_status_matches_check(self):
        self.assertEqual(
            frozenset(RelationResolutionStatus),
            frozenset(("pending", "resolved", "isolated", "failed")),
        )

    def test_registry_field_status_matches_check(self):
        self.assertEqual(
            frozenset(RegistryFieldStatus),
            frozenset(("active", "inactive")),
        )


class GraphEdgeStatusDdlTest(unittest.TestCase):
    """어휘를 **실제 DDL 파일과** 대조한다 — 위 테스트는 사람이 적은 사본끼리 비교할 뿐이다.

    DB 없이 돌아야 하므로 마이그레이션 SQL 텍스트를 읽어 CHECK 목록을 뽑는다. 어휘만 고치고
    마이그레이션을 잊으면 여기서 잡힌다(그 반대도 마찬가지).
    """

    def test_ddl_check_list_matches_enum(self):
        sql = (_SQL_DIR / "230_graph_edge_kind_topic.sql").read_text(encoding="utf-8")
        m = re.search(r"CHECK \(status IN \(([^)]*)\)\)", sql)
        self.assertIsNotNone(m, "230 DDL 에서 status CHECK 목록을 찾지 못했다")
        ddl_values = frozenset(v.strip().strip("'") for v in m.group(1).split(","))
        self.assertEqual(ddl_values, frozenset(GraphEdgeStatus))

    def test_terminal_statuses_are_all_known_vocabulary(self):
        """종결 상태 집합이 어휘 밖을 가리키면 그 값으로 UPDATE 할 때 CHECK 위반이 난다.

        이 프로젝트에서 실제로 났던 사고다 — 만료 상태가 코드에만 있고 DB 에는 없었다.
        """
        from src.relations.approval_policy import _TERMINAL_STATUSES

        self.assertTrue(
            _TERMINAL_STATUSES <= frozenset(GraphEdgeStatus),
            f"종결 상태에 어휘 밖 값이 있다: {_TERMINAL_STATUSES - frozenset(GraphEdgeStatus)}",
        )


if __name__ == "__main__":
    unittest.main()
