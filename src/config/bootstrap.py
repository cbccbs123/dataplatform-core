"""코어 진입점 공통 부트스트랩 (069 US-E FR-E2) — ``.env.{env}`` 로드 + ``init_settings`` 1회 호출.

원래 여러 진입점(run_ingest·run_relations·run_search·백필·포탈 lifespan)이 각자
``Path(__file__).resolve().parents[N]`` 로 레포 루트를 구해 ``.env.{env}`` 로드 + ``init_settings`` 를
부르는 **동일한 5줄 블록을 복제**했다(파일 위치마다 N 이 달라 오프바이원 footgun). 그래서 루트 계산을
이 한 파일에 모아 고정했다 — 호출자는 ``bootstrap_env(env)`` 한 줄만 부르면 된다.

**077/078 레포 분리 반영**: 위 실행 진입점(run_ingest·run_relations·run_search·백필)은 파이프라인 레포
(``processing.*``)로, 포탈은 백엔드 레포(``service.*``)로 이관됐다. 원래 의도는 **각 소비 레포가 자체
부트스트랩을 소유**하는 것이었다(자기 레포 루트에서 자기 ``.env`` 로드) — 설치된 코어의 이 함수를 그대로
재사용하면 ``_REPO_ROOT`` 가 코어 위치를 가리켜 **자기 ``.env`` 가 아니라 코어의 ``.env`` 를 읽기 때문**이다.

⚠️ **현실은 반쪽이다(2026-08-05 실측·정정)**: 백엔드는 자체 ``service/bootstrap.py`` 를 갖고 규칙을 지키지만
**파이프라인 진입점 4개(run_ingest·run_relations·run_search·run_opensearch_resync)는 이 함수를 그대로
재사용한다.** 그래서 파이프 CLI 는 지금까지 코어 레포의 ``.env`` 를 읽어 왔다(파이프 레포에 ``.env`` 파일이
없어 그렇게 "동작"했다). 아래 ``_dotenv_candidates`` 가 **작업 디렉터리를 먼저 보므로** 파이프 레포에서
실행하면 그쪽 ``.env`` 가 우선해 실용상 문제는 없다. 다만 설계상 옳은 것은 파이프 자체 부트스트랩 신설이며,
그 변경은 "파이프에 ``.env`` 가 없으면 파일 로딩이 사라진다"는 동작 변화를 부르므로 **백로그로 둔다**
(`docs/과제_책무_KPI.md`). 이 주석은 그 어긋남을 숨기지 않기 위해 남긴다 — 문서가 금지한 것을 코드가 하고
있으면 다음 사람이 어느 쪽이 맞는지 알 수 없다.

**탐색 위치 2곳**(2026-08-05 추가): ``.env.{env}`` 를 **작업 디렉터리 → 코어 레포 루트** 순으로 찾는다.
작업 디렉터리를 앞에 둔 이유는 ``_REPO_ROOT`` 가 **비-editable 설치에서 레포 루트가 아니기 때문**이다 —
``pip install .`` 로 깔면 이 파일이 ``site-packages/src/config/bootstrap.py`` 가 되어 ``parents[2]`` 는
``site-packages/`` 를 가리킨다. 그러면 작업 디렉터리에 ``.env.dev`` 를 멀쩡히 둬도 읽히지 않고
``ValueError: 필수 환경변수 누락: …`` 로 죽는다(공개본 클린룸 테스트에서 실측). editable 설치에서는
두 경로가 같은 파일을 가리키는 경우가 대부분이라 동작 변화가 없다.

로드는 ``override=False``(OS 기존 환경변수 우선 — 배포 override 존중)이고, 그 뒤 ``init_settings`` 로
필수 env 검증 + frozen 설정 생성(이후 ``get_current_settings`` 활성).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from src.config.settings import PipelineSettings, init_settings

# src/config/bootstrap.py → parents[2] = **코어 레포 루트**. 코어 내부 호출자에겐 이 한 줄이 유일 출처
# (진입점별 parents[N] 분산 제거)다. ※ 소비 레포(파이프/백엔드)는 이 값을 재사용하지 말 것 — 설치된 코어
# 위치를 가리켜 코어의 .env 를 읽게 된다(자체 부트스트랩에서 자기 루트를 계산).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _dotenv_candidates(env: Literal["dev", "prod"]) -> tuple[Path, ...]:
    """``.env.{env}`` 탐색 후보를 **우선순위 순서**로 돌려준다.

    작업 디렉터리를 먼저 보는 이유는 모듈 docstring 참조(비-editable 설치에서 ``_REPO_ROOT`` 가
    ``site-packages/`` 가 되어 사용자가 둔 ``.env`` 를 놓친다).

    Args:
        env: 설정 프로파일(``dev``·``prod``). 파일명 접미사가 된다.

    Returns:
        탐색할 경로 튜플. 존재 여부는 호출자가 확인한다(첫 번째로 존재하는 것만 로드).
    """
    return (Path.cwd() / f".env.{env}", _REPO_ROOT / f".env.{env}")


def bootstrap_env(env: Literal["dev", "prod"]) -> PipelineSettings:
    """``.env.{env}`` 로드 후 ``init_settings(env)`` 로 설정을 초기화하고 그 frozen 설정을 돌려준다.

    운영 진입점(CLI ``main()``·포탈 lifespan)의 표준 부트스트랩 순서다:
    1) ``.env.{env}`` 를 **작업 디렉터리 → 코어 레포 루트** 순으로 찾아 **처음 발견한 하나만**
       ``load_dotenv(override=False)`` — OS 기존 환경변수 우선(배포 override 존중).
    2) ``init_settings(env)`` — 필수 환경변수 검증 후 frozen ``PipelineSettings`` 생성(재현성·헌법 3조).

    ``.env`` 파일이 없어도(컨테이너에서 환경변수 직접 주입 등) init_settings 가 OS 환경변수로 검증하므로
    안전하다. 반환값은 ``init_settings`` 가 만든 설정(설정을 바로 쓰는 진입점 편의).

    Args:
        env: 설정 프로파일(``dev``·``prod``).

    Returns:
        ``init_settings`` 가 만든 frozen 설정.
    """
    for dotenv_path in _dotenv_candidates(env):
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path, override=False)
            break  # 두 곳에 다 있으면 앞선 것(작업 디렉터리)만 쓴다 — 병합하지 않는다
    return init_settings(env)
