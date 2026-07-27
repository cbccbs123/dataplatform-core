"""디렉터리에서 파이프라인 입력으로 쓸 파일 경로 목록을 만든다."""

from __future__ import annotations

from pathlib import Path


def list_file_paths_under_directory(
    directory: str | Path,
    *,
    recursive: bool = True,
) -> list[str]:
    """디렉터리 아래의 **일반 파일만** 모아 경로 목록으로 돌려준다(경로순 정렬).

    Args:
        directory: 훑을 디렉터리.
        recursive: 하위 디렉터리까지 볼지.

    Returns:
        경로 문자열 목록(정렬 고정 — 같은 디렉터리면 항상 같은 순서). **숨김 항목은 항상
        제외한다** — 편집기 임시 파일·OS 메타 파일이 자산으로 잡히면 안 된다.

    Raises:
        NotADirectoryError: 경로가 디렉터리가 아니거나 없을 때.
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
