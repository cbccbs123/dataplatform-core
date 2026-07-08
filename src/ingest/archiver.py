"""061 — Airflow 인입 파일 아카이브 헬퍼 (감시 디렉터리 비우기).

Airflow 감시 수집(030)에서 인입(``WATCHER_INBOX_DIR``) 파일을 수명주기에 따라 ``archive/`` 로 옮겨
감시 대상을 비운다: **중복 파일**은 collect 시 즉시(``dag_collect``), **처리완료(registered)** 파일은
처리 후(``dag_process`` 꼬리 ``archive_processed``)에 이동한다. 이동 시 자산 ``fs_path`` 를 아카이브
경로로 갱신해 다운로드·썸네일·재처리가 유효하게 유지한다(호출부 책임).

이 모듈의 **경로 계산·하위 판정·이동 계획**은 순수·결정적이다(시각 ``when`` 주입·충돌 검사 ``exists``
주입 가능). IO 는 ``execute_move`` 하나로 분리하고 멱등(부분 실패 복구)하게 만든다. 수집/처리 순수
함수·상태전이·해시 dedup·마이그레이션은 건드리지 않는다(파일 수명주기만).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterable
from datetime import date


def is_under(path: str, root: str) -> bool:
    """정규화한 실경로 기준으로 ``path`` 가 ``root`` **하위**인지(순수).

    ``os.path.realpath`` 로 ``..``·심링크를 흡수해 우회를 막고, 구분자 경계로 비교해 형제 디렉터리
    prefix 오탐(``/inbox`` vs ``/inbox2``)을 배제한다. ``root`` 자신은 하위로 보지 않는다(파일이어야).
    """
    rp = os.path.realpath(root)
    pp = os.path.realpath(path)
    return pp.startswith(rp + os.sep)


def archive_dest(
    root: str,
    filename: str,
    *,
    when: date,
    subdir: str = "",
    exists: Callable[[str], bool] = os.path.exists,
) -> str:
    """아카이브 목적 경로 = ``root/[subdir/]YYYYMMDD/filename`` (순수·결정적).

    동명 파일이 이미 있으면 ``stem_1.ext``, ``stem_2.ext`` … 로 결정적으로 회피한다(``exists`` 주입 —
    같은 배치 내 예약 dest 도 함께 검사하려고 호출부가 래핑한다). ``when`` 주입으로 시각 비결정 제거.
    """
    day = when.strftime("%Y%m%d")
    parts = [root, subdir, day] if subdir else [root, day]
    base = os.path.join(*parts)
    candidate = os.path.join(base, filename)
    if not exists(candidate):
        return candidate
    stem, suffix = os.path.splitext(filename)
    n = 1
    while True:
        alt = os.path.join(base, f"{stem}_{n}{suffix}")
        if not exists(alt):
            return alt
        n += 1


def registered_dest(archive_root: str, asset_id: str, fs_path: str, *, when: date) -> str:
    """처리완료(registered) 자산의 아카이브 목적 경로 = ``archive_root/YYYYMMDD/{asset_id}__{name}`` (순수·결정적).

    **``asset_id`` 를 파일명에 넣어 전역 유일**하게 만든다 — 충돌 카운터(파일시스템 상태 의존)를 쓰지 않으므로
    같은 자산을 두 번 스윕해도(부분 실패 복구) **항상 같은 경로**를 재현한다(C4). ``when`` 은 자산 생성일
    (``created_at``)을 주입해 재스윕 날짜가 달라도 경로가 안 바뀌게 한다(dup 경로와 달리 today 를 쓰지 않음).
    """
    day = when.strftime("%Y%m%d")
    return os.path.join(archive_root, day, f"{asset_id}__{os.path.basename(fs_path)}")


def plan_archive_moves(
    rows: Iterable[tuple[str, str, date]],
    *,
    inbox_root: str,
    archive_root: str,
) -> list[tuple[str, str, str]]:
    """registered ``(asset_id, fs_path, created_at)`` 중 **인입 하위**인 것만 ``(asset_id, src, dest)`` 로(순수·결정적).

    이미 아카이브 경로(인입 밖)인 자산은 제외돼 스윕이 자기수렴한다(멱등). dest 는 ``registered_dest`` 로
    asset_id 키 결정적 경로라 파일시스템 상태·재스윕 타이밍에 무관하게 재현된다(C4 복구 안전).
    """
    moves: list[tuple[str, str, str]] = []
    for asset_id, fs_path, created_at in rows:
        if not is_under(fs_path, inbox_root):
            continue  # 이미 아카이브(인입 밖) → 스킵(멱등·자기수렴)
        dest = registered_dest(archive_root, str(asset_id), fs_path, when=created_at)
        moves.append((str(asset_id), fs_path, dest))
    return moves


def execute_move(src: str, dest: str) -> None:
    """부모 디렉터리 생성 후 ``src`` → ``dest`` **원자 이동**(IO).

    멱등·부분실패 복구: ``src`` 가 없고 ``dest`` 가 이미 있으면 no-op(이전 이동이 성공했으나 후속
    ``fs_path`` 갱신이 크래시로 누락된 경우, 다음 스윕이 갱신만 재시도하도록). 둘 다 없으면 오류.
    같은 파일시스템은 ``os.replace``(원자), 크로스-파일시스템은 ``shutil.move`` 로 폴백한다.
    """
    if not os.path.exists(src):
        if os.path.exists(dest):
            return  # 이미 이동됨 — 멱등(복구)
        raise FileNotFoundError(src)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        os.replace(src, dest)  # 동일 fs 원자 이동
    except OSError:
        shutil.move(src, dest)  # 크로스-fs 폴백
