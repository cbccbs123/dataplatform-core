"""코퍼스-골든 정합 가드(025 G3, FR-004) — 순수·결정적.

운영 규칙: **코퍼스에 새 토픽 자산이 추가되면 골든 질의도 추가**되어야 한다(평가 계기판이
코퍼스를 따라가게). 토픽 키는 코퍼스 파일명에서 결정적으로 뽑는다 — 명명 규약 3종을 처리한다:
  - 신규 출처-prefix `youtube_<주제>_<id>` / `wikipedia_<주제>_<id>` → **2번째 토큰**이 주제
    (`youtube_사막_3bTA2c2n2QI.jpg` → `사막`). 출처 prefix(youtube/wikipedia)는 주제가 아니므로
    벗긴다 — 안 그러면 신규 대량 자산이 전부 `youtube`/`wikipedia` 한 토픽으로 뭉쳐 가드가 무력화.
  - 구형 `<주제>_<11자ID>_<제목>` → **첫 토큰**이 주제(`등산_입문_...` → `등산`, `무선_충전기_...`
    → `무선`). golden_os 의 topics 태그(build_golden_os `_QUERY_TOPICS`)와 같은 키 체계.
다른 규약 수집원은 골든 등재 시 topics 태그를 직접 부여하면 된다 — 가드는 집합 비교만 한다.

`uncovered_topics(corpus, golden)` 가 비어 있지 않으면 gated e2e(`RUN_OS_E2E`)가 실패해
골든 질의 추가를 강제한다. DB·OS 미접촉(호출부가 토픽 집합을 만들어 넘긴다 — 헌법 6조).
"""

from __future__ import annotations

import re

# 신규 출처-prefix(주제 아님 — 벗겨야 2번째 토큰이 진짜 주제). build_golden_ko_draft 와 동일 규약.
_SOURCE_PREFIXES = ("youtube", "wikipedia")
# 재수집 명명 `<uuid>__<제목>_(주제).ext` 의 **끝 괄호**가 주제 표식(정본). 제목에 밑줄·공백·특수문자가
# 많아 토큰 위치가 불안정하므로 이 괄호 표식을 토큰 분해보다 우선 신뢰한다.
_TRAILING_PAREN = re.compile(r"\(([^()]+)\)\s*$")


def topic_of_filename(file_name: str) -> str:
    """파일명에서 토픽 키를 뽑는다(순수·결정적, 명명 규약 4종 대응).

    - **신규(재수집)** `<uuid>__<제목>_(주제).ext` → **끝의 괄호 안**이 주제(`…_(전통주).svg` → `전통주`).
      제목에 밑줄·공백·특수문자가 많아 토큰 위치가 불안정하므로 괄호 표식을 우선 신뢰한다.
    - `youtube_사막_<id>.jpg`·`wikipedia_고려청자_<id>.txt` → 2번째 토큰(`사막`·`고려청자`).
    - `등산_입문_<id>_<제목>.mp4` → `등산`(구형: 첫 토큰). `<uuid>__` 접두가 있으면 벗기고 적용한다.
    - 언더스코어가 없으면 확장자 뗀 stem 전체가 토픽(예: `manifest.json` → `manifest` — 위생
      대상이 가드에 노출되도록 그대로 둔다)."""
    stem = (file_name.rsplit(".", 1)[0] if "." in file_name else file_name).strip()
    if not stem:
        return ""
    # 신규 명명: 끝의 (주제) 괄호가 정본(UUID 접두·자유 제목 뒤).
    m = _TRAILING_PAREN.search(stem)
    if m:
        return m.group(1).strip()
    # UUID 접두(`<uuid>__`) 제거 후 구형/출처-prefix 규약 적용.
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] in _SOURCE_PREFIXES:
        return parts[1]
    return parts[0]


def uncovered_topics(corpus_topics: set[str], golden_topics: set[str]) -> list[str]:
    """골든 질의가 커버하지 않는 코퍼스 토픽 목록(정렬·결정적). 비어 있어야 정상."""
    return sorted(corpus_topics - golden_topics)
