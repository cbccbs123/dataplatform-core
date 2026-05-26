"""파일 내용 해시(SHA-256) — 자산 중복 적재 방지(dedup)용."""

from __future__ import annotations

import hashlib
import os

_CHUNK = 1 << 20  # 1MiB


def sha256_file(path: str, *, chunk_size: int = _CHUNK) -> str:
    """파일을 스트리밍으로 읽어 SHA-256 hexdigest(64자) 반환."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hash_and_size(path: str) -> tuple[str, int]:
    """(sha256 hexdigest, 파일 크기 byte). 중복 판별 + asset.file_hash/file_size 적재용."""
    return sha256_file(path), os.path.getsize(path)
