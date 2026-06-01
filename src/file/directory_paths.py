"""디렉터리에서 파이프라인 입력으로 쓸 파일 경로 목록을 만든다."""

from __future__ import annotations

import random
from pathlib import Path


def _prefix_before_last_underscore(path: Path) -> str:
    stem = path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def list_file_paths_under_directory(
    directory: str | Path,
    *,
    recursive: bool = True,
    include_hidden: bool = False,
    dedup_by_prefix: bool = False,
    sample_seed: int | None = None,
) -> list[str]:
    """
    ``directory`` 아래 일반 파일만 모아 경로 문자열 리스트로 반환한다(경로순 정렬).

    - ``recursive=True``: 하위 디렉터리까지 탐색.
    - ``include_hidden=False``: 루트 기준 상대 경로에 ``.`` 로 시작하는 이름이 있으면 건너뜀.
    - ``dedup_by_prefix=True``: 파일명 stem 기준 ``마지막 '_'`` 앞까지를 키로 묶어 그룹당 1개만
      ``sample_seed`` 가 고정되면 재현 가능한 난수로 선택.

      예) ``PC_콘솔_게임_6ihGHlrfqM8.mp4`` 와 ``PC_콘솔_게임_xxxxx.mp4`` → 같은 그룹.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"디렉터리가 아니거나 없습니다: {root}")

    def _skip(path: Path) -> bool:
        if include_hidden:
            return False
        try:
            rel = path.relative_to(root)
        except ValueError:
            return False
        return any(part.startswith(".") for part in rel.parts)

    paths: list[Path] = []
    it = root.rglob("*") if recursive else root.iterdir()
    for p in it:
        if not p.is_file():
            continue
        if _skip(p):
            continue
        paths.append(p)

    if not dedup_by_prefix:
        paths.sort(key=lambda x: str(x))
        return [str(p) for p in paths]

    rng = random.Random(sample_seed)
    groups: dict[str, list[Path]] = {}
    for p in paths:
        key = _prefix_before_last_underscore(p)
        groups.setdefault(key, []).append(p)

    sampled = [rng.choice(group) for group in groups.values()]
    sampled.sort(key=lambda x: str(x))
    return [str(p) for p in sampled]
