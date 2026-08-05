"""069 US-E FR-E2 — bootstrap_env 공용 부트스트랩 단위 테스트(헌법 8조).

여러 진입점이 공유하는 부트스트랩을 직접 봉인한다: ``.env.{env}`` **탐색 순서**·유무 분기·
``init_settings`` 위임·반환값 동일성. load_dotenv/init_settings 를 patch 해 실 파일·실 설정 없이 검증.

**2026-08-05 보강**: 탐색이 "작업 디렉터리 → 코어 레포 루트" 2곳이 됐다(비-editable 설치에서
``_REPO_ROOT`` 가 ``site-packages/`` 를 가리켜 사용자의 ``.env`` 를 놓치던 결함). 그래서 테스트는
**반드시 임시 디렉터리로 chdir 한다** — 그러지 않으면 실제 레포 루트의 ``.env.dev`` 가 cwd 후보로
잡혀 "파일 부재" 케이스가 성립하지 않는다(실제로 그렇게 깨졌다).
"""
from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.config import bootstrap


class TestBootstrapEnv(unittest.TestCase):
    def _run(self, *, cwd_has_env: bool, root_has_env: bool):
        """부트스트랩을 격리 실행하고 관찰 지점을 돌려준다.

        Args:
            cwd_has_env: 작업 디렉터리에 ``.env.dev`` 를 둘지.
            root_has_env: ``_REPO_ROOT`` 대역 디렉터리에 ``.env.dev`` 를 둘지.

        Returns:
            ``(반환값, sentinel, load_dotenv mock, init_settings mock, cwd 경로, root 경로)``.
        """
        sentinel = object()  # init_settings 반환 대역 — 반환 동일성 확인용
        with tempfile.TemporaryDirectory() as d_cwd, tempfile.TemporaryDirectory() as d_root:
            # macOS: /var → /private/var 심볼릭 링크. Path.cwd() 는 resolve 된 경로를
            # 돌려주므로 비교가 어긋난다 → 여기서 미리 resolve 해 둔다.
            cwd, root = Path(d_cwd).resolve(), Path(d_root).resolve()
            if cwd_has_env:
                (cwd / ".env.dev").write_text("X=1\n", encoding="utf-8")
            if root_has_env:
                (root / ".env.dev").write_text("Y=2\n", encoding="utf-8")
            with (
                contextlib.chdir(cwd),  # 실제 레포 루트의 .env.dev 가 cwd 후보로 잡히는 것을 막는다
                mock.patch.object(bootstrap, "_REPO_ROOT", root),
                mock.patch.object(bootstrap, "load_dotenv") as m_load,
                mock.patch.object(bootstrap, "init_settings", return_value=sentinel) as m_init,
            ):
                out = bootstrap.bootstrap_env("dev")
        return out, sentinel, m_load, m_init, cwd, root

    def test_loads_repo_root_dotenv_override_false_and_returns_settings(self) -> None:
        # 레포 루트에만 존재 → 그 파일을 load_dotenv(override=False) 1회 + init_settings 위임.
        out, sentinel, m_load, m_init, _cwd, root = self._run(cwd_has_env=False, root_has_env=True)
        m_load.assert_called_once()
        self.assertEqual(m_load.call_args.kwargs.get("dotenv_path"), root / ".env.dev")
        self.assertIs(m_load.call_args.kwargs.get("override"), False)  # OS 기존 환경변수 우선 보존
        m_init.assert_called_once_with("dev")
        self.assertIs(out, sentinel)  # bootstrap_env 는 init_settings 결과를 그대로 돌려준다

    def test_skips_dotenv_when_absent_but_still_inits(self) -> None:
        # 두 곳 모두 부재(컨테이너 환경변수 직접 주입 등) → load_dotenv 미호출·init_settings 는 여전히 검증.
        out, sentinel, m_load, m_init, _cwd, _root = self._run(cwd_has_env=False, root_has_env=False)
        m_load.assert_not_called()
        m_init.assert_called_once_with("dev")
        self.assertIs(out, sentinel)

    def test_cwd_dotenv_is_found_when_repo_root_has_none(self) -> None:
        # 🔴 이 케이스가 결함의 핵심이다 — 비-editable 설치는 _REPO_ROOT 가 site-packages 라
        # 거기엔 .env 가 없다. 그때 **작업 디렉터리**의 .env 를 읽어야 한다.
        _out, _s, m_load, m_init, cwd, _root = self._run(cwd_has_env=True, root_has_env=False)
        m_load.assert_called_once()
        self.assertEqual(m_load.call_args.kwargs.get("dotenv_path"), cwd / ".env.dev")
        m_init.assert_called_once_with("dev")

    def test_cwd_wins_when_both_exist_and_only_one_is_loaded(self) -> None:
        # 두 곳에 다 있으면 **작업 디렉터리 것 하나만** 쓴다(병합하지 않는다 — 어느 값이 이겼는지
        # 알 수 없는 상태를 만들지 않기 위해).
        _out, _s, m_load, _m_init, cwd, _root = self._run(cwd_has_env=True, root_has_env=True)
        m_load.assert_called_once()
        self.assertEqual(m_load.call_args.kwargs.get("dotenv_path"), cwd / ".env.dev")

    def test_candidate_order_is_cwd_then_repo_root(self) -> None:
        # 순서 자체를 봉인한다 — 후보 목록이 뒤바뀌면 위 우선순위 테스트만으로는 놓칠 수 있다.
        with tempfile.TemporaryDirectory() as d, contextlib.chdir(d):
            got = bootstrap._dotenv_candidates("prod")
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0], Path(d).resolve() / ".env.prod")
        self.assertEqual(got[1], bootstrap._REPO_ROOT / ".env.prod")


if __name__ == "__main__":
    unittest.main()
