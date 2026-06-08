"""임베딩 채널 프로파일 토글 실DB e2e (018 G4 — T011).

``EMBED_ACTIVE_CHANNEL`` 설정 한 줄로 운영 검색의 텍스트 임베딩 채널(=모델)이 갈리는지를
**017 백필 데이터(``channel='st'`` KoSimCSE · ``channel='st_bge'`` BGE-M3 공존)** 위에서 검증한다.

검증 전략(docs/테스트_가이드.md §2~3 — 실DB e2e 는 RED 가 어려운 검증형)
  - **활성 토글(SC-001)**: 활성 채널을 ``'st'``↔``'st_bge'`` 로 바꿔 ``search_hybrid(text_channel=None)``
    을 실DB 로 호출했을 때, ``_grouped_fn`` 스파이로 **실제 ST 검색에 전달된 채널·질의모델**을
    포착해 활성 프로파일(``active_embed_channel()``·``active_embed_model()``)을 따르는지 확인한다.
    스파이는 실 ``search_media_all_grouped`` 로 그대로 위임하므로, "활성=st_bge 일 때 BGE 채널
    후보로 실제 검색돼 결과가 나온다"까지 함께 증명한다(컷오프 없는 ST 하이브리드라 그 채널에
    데이터가 있으면 후보가 반드시 반환된다).
  - **롤백(SC-004)**: ``st_bge``→``st`` 복귀 시 다시 KoSimCSE(``'st'``) 채널로 검색되고, 토글
    과정 전후로 ``asset_embedding`` 건수가 불변(재임베딩·삭제 없는 즉시 원복)임을 확인한다.
  - **결정성(SC-003)**: 동일 활성·질의 2회가 동일 랭킹(자산 id 순서)을 낸다.

격리(헌법 3조·다른 테스트 오염 금지): 활성 토글은 ``EMBED_ACTIVE_CHANNEL`` 환경변수를
``mock.patch.dict`` 로 격리한 뒤 ``init_settings("dev")`` 로 settings 싱글톤을 재빌드하고,
컨텍스트 종료 시 ``.env.dev`` 기본('st')으로 항상 복원한다. 읽기 전용(검색·조회만, 쓰기·스키마
0). 학습 0(두 모델 inference only). LLM 질의 구조화는 ``structured`` 주입으로 건너뛴다(결정성).

소스 코드 변경 0(테스트만) · 스키마 변경 0. ``RUN_DB_E2E=1`` + 실 PostgreSQL + 017 ``st_bge``
백필 데이터가 있을 때만 실행(기본 discover 에서는 skip).
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

from dotenv import load_dotenv

from src.file.file_type_defs import MEDIA_TYPES_ST_CHUNK_SEARCH
from src.search.media_search import search_media_all_grouped
from src.search.search_service import search_hybrid

_RUN = os.getenv("RUN_DB_E2E") == "1"
_REPO = Path(__file__).resolve().parents[1]
_ENV = _REPO / ".env.dev"

# LLM 질의 구조화를 건너뛰는 고정 질의(결정성). 비-emptiness 는 질의 내용과 무관하다 —
# ST 하이브리드는 컷오프 없이 해당 채널의 모든 후보를 점수순 상위 N 으로 돌려주므로, 그 채널에
# 임베딩이 있으면 후보가 반드시 반환된다(질의는 랭킹만 흔든다).
_QUERY = "영상 콘텐츠 요약"
_STRUCTURED = {"semantic_query": _QUERY, "semantic_query_en": ""}
# ST 텍스트 채널 검증 대상 버킷(text/audio/video). image 는 CLIP 채널이라 텍스트 채널 토글 무관.
_ST_BUCKETS = ("text_documents", "audio", "video")


@contextmanager
def _active_channel(channel: str):
    """``EMBED_ACTIVE_CHANNEL`` 을 격리 토글하고 settings 싱글톤을 재빌드한다(컨텍스트 한정).

    ``init_settings("dev")`` 가 ``_build_settings`` 로 현재 env 에서 settings 를 새로 만든다 —
    ``mock.patch.dict`` 로 ``EMBED_ACTIVE_CHANNEL`` 만 덮은 채 재빌드하면 활성 채널만 바뀐다.
    컨텍스트 종료(정상·예외 무관) 시 env 가 ``.env.dev`` 값('st')으로 복원된 뒤 다시 재빌드해
    settings 를 기본 활성으로 되돌린다(다른 테스트 오염 차단)."""
    from src.config.settings import init_settings

    try:
        with mock.patch.dict(os.environ, {"EMBED_ACTIVE_CHANNEL": channel}):
            init_settings("dev")
            yield
    finally:
        # patch.dict 가 env 를 .env.dev 기본으로 복원한 뒤 재빌드 → 활성 항상 'st' 로 원복.
        init_settings("dev")


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만 실행하는 실 DB e2e (017 st_bge 백필 데이터 필요)")
class TestEmbedChannelProfileE2E(unittest.TestCase):
    """활성 채널 토글·롤백·결정성을 실DB 검색으로 검증(SC-001·SC-003·SC-004).

    하나의 테스트로 묶는다 — 토글→롤백→결정성이 같은 데이터·연결 위에서 순차로 의존하고,
    실DB 검색 라운드트립을 최소화하기 위함이다(``test_embedding_ab_kpi`` 와 동일 1-메서드 패턴)."""

    db = None  # type: ignore[assignment]

    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv(_ENV, override=False)
        from src.config.settings import init_settings

        init_settings("dev")  # 기준선(활성 'st')
        from src.database.postgres_util import PostgresUtil

        try:
            cls.db = PostgresUtil()
            cls.db.__enter__()
        except Exception as exc:  # noqa: BLE001 — 접속 불가 시 skip
            raise unittest.SkipTest(f"DB 미접속: {type(exc).__name__}: {exc}") from None

        # 데이터 전제(FR-006): 두 채널이 text/audio/video 모달리티에 모두 백필돼 있어야 토글로
        # 채널 전환을 실측할 수 있다. 한쪽이라도 비면 e2e skip(전환 전 백필은 운영자 책임).
        cls.st_count = cls._pool_count(cls.db, "st")
        cls.bge_count = cls._pool_count(cls.db, "st_bge")
        if cls.st_count == 0 or cls.bge_count == 0:
            raise unittest.SkipTest(
                f"채널 데이터 부족 — st={cls.st_count}, st_bge={cls.bge_count} "
                "(017 백필 필요: RUN_DB_E2E 토글 검증은 두 채널 공존 전제)"
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.db is not None:
            cls.db.__exit__(None, None, None)
        # 활성 설정을 .env.dev 기본('st')으로 확실히 원복(방어적).
        from src.config.settings import init_settings

        init_settings("dev")

    @staticmethod
    def _pool_count(db: Any, channel: str) -> int:
        """``channel`` 임베딩 중 text/audio/video(ST 검색 대상) 모달리티 자산의 건수."""
        mods = sorted(MEDIA_TYPES_ST_CHUNK_SEARCH)
        with db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM asset_embedding ae "
                "JOIN asset a ON a.asset_id = ae.asset_id "
                "WHERE ae.channel = %s AND a.modality = ANY(%s)",
                (channel, mods),
            )
            return int(cur.fetchone()[0])

    @staticmethod
    def _embed_counts(db: Any) -> dict[str, int]:
        """채널별 ``asset_embedding`` 총 건수(데이터 불변 검증용 스냅샷)."""
        with db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT channel, count(*) FROM asset_embedding GROUP BY channel ORDER BY channel"
            )
            return {str(ch): int(n) for ch, n in cur.fetchall()}

    def _search_capture(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """``search_hybrid(text_channel=None)`` 을 실DB 로 호출하고, ST 경로에 전달된 채널·질의모델을
        스파이로 포착한다. 스파이는 실 ``search_media_all_grouped`` 로 위임 → 실제 검색이 수행된다."""
        captured: dict[str, Any] = {}

        def spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["channel"] = kwargs.get("channel")
            captured["query_model_name"] = kwargs.get("query_model_name")
            return search_media_all_grouped(*args, **kwargs)

        res = search_hybrid(
            _QUERY,
            modalities=["text", "audio", "video"],
            structured=_STRUCTURED,
            _grouped_fn=spy,
        )
        return captured, res

    @staticmethod
    def _ranked_ids(res: dict[str, Any]) -> dict[str, list[str]]:
        """결과를 버킷별 자산 id 순서로 환원한다(결정성 비교용)."""
        results = res["results"]
        return {b: [str(r["id"]) for r in results.get(b, [])] for b in _ST_BUCKETS}

    def _total_hits(self, res: dict[str, Any]) -> int:
        return sum(len(res["results"].get(b, [])) for b in _ST_BUCKETS)

    def test_active_channel_toggle_rollback_deterministic(self) -> None:
        from src.config.settings import active_embed_channel, active_embed_model

        before = self._embed_counts(self.db)

        # SC-001 ① 기준선 활성 'st' → ST 검색이 KoSimCSE('st') 채널로 해소되고 결과를 낸다.
        with self.subTest(scenario="active=st"), _active_channel("st"):
            self.assertEqual(active_embed_channel(), "st")
            cap, res = self._search_capture()
            self.assertEqual(cap["channel"], "st")
            self.assertEqual(cap["query_model_name"], active_embed_model())  # KoSimCSE
            self.assertGreater(self._total_hits(res), 0, "st 채널 검색이 결과를 내야 함")

        # SC-001 ② 활성 'st_bge' → ST 검색이 BGE('st_bge') 채널로 일제히 전환되고 결과를 낸다.
        with self.subTest(scenario="active=st_bge"), _active_channel("st_bge"):
            self.assertEqual(active_embed_channel(), "st_bge")
            bge_model = active_embed_model()  # BAAI/bge-m3
            cap, res = self._search_capture()
            self.assertEqual(cap["channel"], "st_bge")
            self.assertEqual(cap["query_model_name"], bge_model)
            self.assertGreater(
                self._total_hits(res), 0, "st_bge 채널 검색이 BGE 후보로 결과를 내야 함"
            )

        # SC-004 롤백: 'st_bge'→'st' 복귀 → 다시 KoSimCSE('st') 채널로 원복.
        with self.subTest(scenario="rollback st_bge->st"), _active_channel("st"):
            cap, _ = self._search_capture()
            self.assertEqual(cap["channel"], "st", "롤백 후 활성 채널이 st 로 원복돼야 함")
            self.assertEqual(cap["query_model_name"], active_embed_model())

        # SC-003 결정성: 동일 활성·질의 2회가 동일 랭킹(자산 id 순서).
        with self.subTest(scenario="determinism"), _active_channel("st_bge"):
            r1 = self._ranked_ids(self._search_capture()[1])
            r2 = self._ranked_ids(self._search_capture()[1])
            self.assertEqual(r1, r2, "동일 활성·질의 2회는 동일 랭킹이어야 함(결정성)")

        # SC-004 데이터 불변: 토글·검색은 읽기 전용 — 채널별 임베딩 건수가 전후로 동일.
        after = self._embed_counts(self.db)
        self.assertEqual(before, after, "토글·검색으로 asset_embedding 데이터가 변하면 안 됨")


if __name__ == "__main__":
    unittest.main()
