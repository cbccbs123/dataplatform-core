"""진입점 공통 부트스트랩 (069 US-E FR-E2) — ``.env.{env}`` 로드 + ``init_settings`` 1회 호출.

종전 6개 CLI 진입점(run_ingest·run_relations·run_search·run_topic_backfill·run_about_backfill·
run_opensearch_resync)과 포탈 lifespan 이 각자 ``Path(__file__).resolve().parents[N]`` 로 레포 루트를
구해 ``.env.{env}`` 를 로드하고 ``init_settings`` 를 부르는 **동일한 5줄 블록을 복제**했다. 파일 위치마다
``parents[N]`` 의 N 이 달라 오프바이원 footgun 이 됐다(FR-E6 에서 포탈 이동 시 parents[2]→[3] 보정 필요).

여기로 모아 **레포 루트 계산을 한 곳(이 파일 기준)에** 고정한다 — 진입점은 ``bootstrap_env(env)`` 한 줄만
호출하면 된다. 동작은 종전과 동일: ``.env.{env}`` 가 있으면 ``override=False``(OS 기존 환경변수 우선)로
로드한 뒤 ``init_settings`` 로 필수 env 검증 + frozen 설정 생성(이후 ``get_current_settings`` 활성).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from src.config.settings import PipelineSettings, init_settings

# src/config/bootstrap.py → parents[2] = 레포 루트. **루트 계산은 이 한 줄이 유일 출처**(진입점별
# parents[N] 분산 제거) — 이 파일이 옮겨지지 않는 한 호출자 위치와 무관하게 항상 올바르다.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def bootstrap_env(env: Literal["dev", "prod"]) -> PipelineSettings:
    """``.env.{env}`` 로드 후 ``init_settings(env)`` 로 설정을 초기화하고 그 frozen 설정을 돌려준다.

    운영 진입점(CLI ``main()``·포탈 lifespan)의 표준 부트스트랩 순서다:
    1) ``.env.{env}`` 가 있으면 ``load_dotenv(override=False)`` — OS 기존 환경변수 우선(배포 override 존중).
    2) ``init_settings(env)`` — 필수 환경변수 검증 후 frozen ``PipelineSettings`` 생성(재현성·헌법 3조).

    ``.env`` 파일이 없어도(컨테이너에서 환경변수 직접 주입 등) init_settings 가 OS 환경변수로 검증하므로
    안전하다. 반환값은 ``init_settings`` 가 만든 설정(설정을 바로 쓰는 진입점 편의).
    """
    dotenv_path = _REPO_ROOT / f".env.{env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    return init_settings(env)
