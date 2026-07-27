"""디렉터리에서 파이프라인 입력으로 쓸 파일 경로 목록을 만든다."""

from __future__ import annotations

from pathlib import Path


def list_file_paths_under_directory(
    directory: str | Path,
    *,
    recursive: bool = True,
) -> list[str]:
    """
    ``directory`` 아래 일반 파일만 모아 경로 문자열 리스트로 반환한다(경로순 정렬).

    - ``recursive=True``: 하위 디렉터리까지 탐색.
    - 숨김 항목 제외: 루트 기준 상대 경로에 ``.`` 로 시작하는 이름이 있으면 건너뛴다.

    (069 US-F: ``include_hidden``·``dedup_by_prefix``·``sample_seed`` 옵션은 운영 소비 0 으로 제거.
    이제 항상 숨김 제외·중복샘플링 없음 = 종전 기본 동작.)
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"디렉터리가 아니거나 없습니다: {root}")

    def _skip(path: Path) -> bool:
        """수집에서 제외할 경로인지 판단한다(루트 밖 경로는 제외하지 않는다)."""
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

    paths.sort(key=lambda x: str(x))
    return [str(p) for p in paths]
