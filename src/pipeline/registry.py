"""v2 전략 레지스트리 — 슬롯별 (이름 → 구현 + capability 태그)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class _Entry:
    fn: Callable
    tags: frozenset[str]


class StrategyRegistry:
    """슬롯(classify/extract/embed/persist/…)별로 이름 붙은 전략을 보관한다."""

    def __init__(self) -> None:
        self._slots: dict[str, dict[str, _Entry]] = {}

    def register(self, slot: str, name: str, fn: Callable, *, tags: Iterable[str] = ()) -> None:
        self._slots.setdefault(slot, {})[name] = _Entry(fn, frozenset(tags))

    def resolve(self, slot: str, name: str) -> Callable:
        try:
            return self._slots[slot][name].fn
        except KeyError as e:
            raise KeyError(f"미등록 전략: slot={slot!r} name={name!r}") from e

    def tags(self, slot: str, name: str) -> frozenset[str]:
        try:
            return self._slots[slot][name].tags
        except KeyError as e:
            raise KeyError(f"미등록 전략: slot={slot!r} name={name!r}") from e


# 프로세스 전역 기본 레지스트리(builtins 가 등록).
DEFAULT_REGISTRY = StrategyRegistry()
