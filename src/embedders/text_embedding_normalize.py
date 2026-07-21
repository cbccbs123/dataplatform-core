"""
임베딩 입력용 텍스트 정규화.

- NFKC로 호환 문자(전각 숫자·동그라미 숫자 일부 등)를 ASCII에 가깝게 맞춘다.
- zero-width·제어 문자 등 임베딩에 거의 도움이 되지 않는 코드포인트를 제거한다.
- 동그라미·괄호·딩뱃 계열 숫자를 일반 숫자(또는 ``(12)`` 형태)로 바꾼다.

모든 유니코드 기호를 커버하지는 않으며, 필요 시 ``_symbol_replacements()`` 에 항목을 추가하면 된다.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_NOISE_CHARS: frozenset[str] = frozenset(
    "\ufeff"
    "\u00ad"
    "\u200b\u200c\u200d\u200e\u200f"
    "\u202a\u202b\u202c\u202d\u202e"
    "\u2060\u2061\u2062\u2063\u2064"
)


def _symbol_replacements() -> dict[str, str]:
    """동그라미·괄호·딩뱃 등 단일 코드포인트 → 일반 문자열."""
    m: dict[str, str] = {}
    # \u2460 U+2460\u20132468(circled digit 1\u20139) \u2192 "1".."9", \u2469 U+2469 \u2192 "10"
    for i in range(1, 10):
        m[chr(0x245F + i)] = str(i)
    m["\u2469"] = "10"
    # \u246a U+246A\u20132473(circled 11\u201320) \u2192 "11".."20", \u24ea U+24EA \u2192 "0"
    for i in range(11, 21):
        m[chr(0x246A + (i - 11))] = str(i)
    m["\u24ea"] = "0"
    # \u2474 U+2474\u20132487(parenthesized digit 1\u201320) \u2192 "(1)".."(20)"
    for i in range(1, 21):
        m[chr(0x2473 + i)] = f"({i})"
    # \u2776 U+2776\u2013277E(dingbat negative circled 1\u20139) \u2192 "1".."9"
    for i in range(1, 10):
        m[chr(0x2775 + i)] = str(i)
    # \u2780 U+2780\u20132788(dingbat circled 1\u20139) \u2192 "1".."9"
    for i in range(1, 10):
        m[chr(0x277F + i)] = str(i)
    return m


@lru_cache(maxsize=1)
def _replacement_pairs() -> tuple[tuple[str, str], ...]:
    d = _symbol_replacements()
    return tuple(sorted(d.items(), key=lambda kv: len(kv[0]), reverse=True))


def _strip_control_and_noise(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if ch in _NOISE_CHARS:
            continue
        cat = unicodedata.category(ch)
        if cat == "Cc" and ch not in "\n\t\r":
            continue
        out.append(ch)
    return "".join(out)


def _collapse_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in s.split("\n"):
        line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
        lines.append(line)
    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_text_for_embedding(text: str | None) -> str:
    """
    임베딩 입력·검색 쿼리 전처리에 동일하게 사용한다.

    처리 요약: NFKC → 제어/무음 문자 제거 → 동그라미·괄호·딩뱃 숫자 치환 → 공백·빈 줄 정리.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = _strip_control_and_noise(s)
    for old, new in _replacement_pairs():
        s = s.replace(old, new)
    s = _collapse_whitespace(s)
    return s
