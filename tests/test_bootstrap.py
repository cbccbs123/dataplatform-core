"""069 US-E FR-E2 — bootstrap_env 공용 부트스트랩 단위 테스트(헌법 8조).

7곳(6 CLI 진입점 + 포탈 lifespan)이 공유하는 부트스트랩을 직접 봉인한다: ``.env.{env}`` 유무 분기·
``init_settings`` 위임·반환값 동일성. load_dotenv/init_settings 를 patch 해 실 파일·실 설정 없이 검증.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.config import bootstrap


class TestBootstrapEnv(unittest.TestCase):
    def _run(self, *, env_file_exists: bool):
        sentinel = object()  # init_settings 반환 대역 — 반환 동일성 확인용
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            if env_file_exists:
                (root / ".env.dev").write_text("X=1\n", encoding="utf-8")
            with (
                mock.patch.object(bootstrap, "_REPO_ROOT", root),
                mock.patch.object(bootstrap, "load_dotenv") as m_load,
                mock.patch.object(bootstrap, "init_settings", return_value=sentinel) as m_init,
            ):
                out = bootstrap.bootstrap_env("dev")
        return out, sentinel, m_load, m_init

    def test_loads_dotenv_when_present_override_false_and_returns_settings(self) -> None:
        # .env.{env} 존재 → load_dotenv(override=False) 1회 + init_settings 위임 + 그 결과 반환.
        out, sentinel, m_load, m_init = self._run(env_file_exists=True)
        m_load.assert_called_once()
        self.assertIs(m_load.call_args.kwargs.get("override"), False)  # OS 기존 환경변수 우선 보존
        m_init.assert_called_once_with("dev")
        self.assertIs(out, sentinel)  # bootstrap_env 는 init_settings 결과를 그대로 돌려준다

    def test_skips_dotenv_when_absent_but_still_inits(self) -> None:
        # .env 부재(컨테이너 환경변수 직접 주입 등) → load_dotenv 미호출·init_settings 는 여전히 검증.
        out, sentinel, m_load, m_init = self._run(env_file_exists=False)
        m_load.assert_not_called()
        m_init.assert_called_once_with("dev")
        self.assertIs(out, sentinel)


if __name__ == "__main__":
    unittest.main()
