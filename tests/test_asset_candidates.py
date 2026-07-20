"""018 G3 — 관계 후보 검색 활성 임베딩 채널 프로파일(asset_candidates) 단위 테스트.

018 은 운영 텍스트 임베딩 채널(=모델)을 ``EMBED_ACTIVE_CHANNEL`` 설정으로 토글한다.
관계 후보(`find_embedding_candidates`)의 텍스트 채널 결정(`_channels_param`)을 'st' 하드코딩
대신 **활성 채널 단일 출처**(`active_embed_channel()`)로 치환한다 — 적재·검색과 동일 active.

테스트 전략(docs/테스트_가이드.md §2)
  - **settings 모킹**: 활성 해소가 ``get_current_settings`` 를 거치므로 가짜 설정을 주입해
    순수 단위로 채널을 검증한다(``init_settings``/DB 0).
  - **회귀 가드(최우선)**: 기본 active='st' 시 채널 ['st']/['st','clip'] 로 기존과 **동치**
    (헌법 8조, SC-002). settings 미초기화에서도 'st' 로 폴백(기존 관계 단위가 settings 없이 동작).
  - **CLIP 무변경**: 시각 채널('clip')은 active 와 무관하게 불변(텍스트 채널만 프로파일).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

from src.relations.asset_candidates import _channels_param, find_embedding_candidates


def _fake_cfg(active: str) -> SimpleNamespace:
    """active_embed_channel 이 읽는 필드만 가진 가짜 설정(실 .env·init_settings 없이)."""
    return SimpleNamespace(embed=SimpleNamespace(active_channel=active))


class TestChannelsParamActive(unittest.TestCase):
    """``_channels_param`` 의 텍스트 채널이 활성 채널을 반영한다."""

    def test_st_kind_uses_active_channel(self) -> None:
        # active='st_bge' → 텍스트 후보 채널이 ['st_bge'] (단일 출처).
        with mock.patch(
            "src.config.settings.get_current_settings",
            return_value=_fake_cfg("st_bge"),
        ):
            self.assertEqual(_channels_param("st"), ["st_bge"])

    def test_both_kind_uses_active_plus_clip(self) -> None:
        # both → [활성 텍스트 채널, 'clip']. CLIP 은 무변경.
        with mock.patch(
            "src.config.settings.get_current_settings",
            return_value=_fake_cfg("st_bge"),
        ):
            self.assertEqual(_channels_param("both"), ["st_bge", "clip"])

    def test_clip_kind_unchanged(self) -> None:
        # 'clip' 은 active 와 무관하게 ['clip'] (텍스트 채널만 프로파일).
        with mock.patch(
            "src.config.settings.get_current_settings",
            return_value=_fake_cfg("st_bge"),
        ):
            self.assertEqual(_channels_param("clip"), ["clip"])


class TestChannelsParamDefaultRegression(unittest.TestCase):
    """기본 active='st' 동치(회귀 가드)."""

    def test_st_active_is_st(self) -> None:
        with mock.patch(
            "src.config.settings.get_current_settings", return_value=_fake_cfg("st")
        ):
            self.assertEqual(_channels_param("st"), ["st"])
            self.assertEqual(_channels_param("both"), ["st", "clip"])

    def test_uninitialized_settings_falls_back_to_st(self) -> None:
        # settings 미초기화(RuntimeError)면 기존 기본 'st' 로 폴백 — 기존 관계 단위가
        # settings 없이 그대로 동작(회귀 0).
        with mock.patch(
            "src.config.settings.get_current_settings",
            side_effect=RuntimeError("settings 미초기화"),
        ):
            self.assertEqual(_channels_param("st"), ["st"])
            self.assertEqual(_channels_param("both"), ["st", "clip"])


class TestFindCandidatesActiveChannelReachesSql(unittest.TestCase):
    """활성 채널이 SQL 파라미터(channels)에 실린다(스파이)."""

    def test_active_channel_in_sql_params(self) -> None:
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchall.return_value = []
        conn.cursor.return_value = cur
        with mock.patch(
            "src.config.settings.get_current_settings",
            return_value=_fake_cfg("st_bge"),
        ):
            find_embedding_candidates(
                conn, source_asset_id="x", top_k=5, embedding_kind="both"
            )
        params = cur.execute.call_args[0][1]
        # 파라미터 순서: (source_asset_id, channels, source_asset_id, min_sim, top_k)
        self.assertIn(["st_bge", "clip"], params)


if __name__ == "__main__":
    unittest.main()
