"""v299 asset_topic 마이그레이션 스키마 테스트 (코어 — DDL 파일 존재·필수 컬럼·리비전 체인).

078 레포 분리: asset_topic **분류(classify)** 로직은 파이프라인 레포로 이관됐으나, v299 **마이그레이션(스키마)**
의 정본은 코어(`migrations/`)이므로 그 존재·형상 검사는 코어에 남긴다(구 `tests/test_asset_topic_classify.py`
의 `TestMigrationV299` 에서 분리·classify 의존 없음·DB 불요).
"""
from __future__ import annotations

import os
import re
import unittest

# 마이그레이션 파일 경로(레포 루트 기준·CI 무관). 이 테스트 파일: tests/test_migration_v299_asset_topic.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SQL_PATH = os.path.join(_REPO_ROOT, "migrations", "sql", "299_asset_topic.sql")
_ALEMBIC_PATH = os.path.join(
    _REPO_ROOT, "migrations", "alembic", "versions", "v299_asset_topic.py"
)


class TestMigrationV299(unittest.TestCase):
    """T101 — v299 asset_topic DDL 파일 존재 + 필수 컬럼/제약 문자열(파일 파싱·DB 불요)."""

    def test_sql_file_exists(self) -> None:
        self.assertTrue(
            os.path.isfile(_SQL_PATH), f"299_asset_topic.sql 이 없다: {_SQL_PATH}"
        )

    def test_sql_defines_asset_topic_table_and_columns(self) -> None:
        with open(_SQL_PATH, encoding="utf-8") as fh:
            sql = fh.read().lower()
        # 테이블·PK·필수 컬럼·정책버전·인덱스가 DDL 에 문자열로 존재해야 한다.
        self.assertIn("create table", sql)
        self.assertIn("asset_topic", sql)
        self.assertIn("asset_id", sql)
        self.assertIn("topic_ko", sql)
        self.assertIn("policy_version", sql)
        # 자산 삭제 시 자기주제 행 동반 삭제(FR-101) — ON DELETE CASCADE.
        self.assertIn("on delete cascade", sql)
        # 파생 조인·패싯용 (topic_ko, subtopic_ko) 인덱스.
        self.assertIn("idx_asset_topic_pair", sql)

    def test_alembic_revision_chains_and_reversible(self) -> None:
        with open(_ALEMBIC_PATH, encoding="utf-8") as fh:
            src = fh.read()
        # down_revision 이 실제 v298 revision id 로 체인 연결.
        self.assertIn("v298_labels_schema_object", src)
        # run_sql_file 관례로 SQL 실행 + downgrade 는 DROP TABLE.
        self.assertIn("run_sql_file", src)
        self.assertIn("299_asset_topic.sql", src)
        self.assertRegex(src, r"(?i)drop\s+table\s+if\s+exists\s+asset_topic")
        # revision id 는 alembic_version.version_num(VARCHAR(32)) 제약 — 32자 이하.
        m = re.search(r'^revision\s*=\s*["\']([^"\']+)["\']', src, re.MULTILINE)
        self.assertIsNotNone(m, "revision id 를 찾지 못했다")
        self.assertLessEqual(len(m.group(1)), 32)


if __name__ == "__main__":
    unittest.main()
