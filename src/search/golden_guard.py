"""코퍼스-골든 정합 가드(025 G3, FR-004) — 순수·결정적.

운영 규칙: **코퍼스에 새 토픽 자산이 추가되면 골든 질의도 추가**되어야 한다(평가 계기판이
코퍼스를 따라가게). 토픽 키는 코퍼스 파일명 슬러그의 첫 토큰(`등산_입문_...` → `등산`)으로,
수집 데이터셋의 명명 규약에 맞춘 결정적 휴리스틱이다(다른 규약 수집원은 골든 등재 시 topics
태그를 직접 부여하면 된다 — 가드는 집합 비교만 한다).

`uncovered_topics(corpus, golden)` 가 비어 있지 않으면 gated e2e(`RUN_OS_E2E`)가 실패해
골든 질의 추가를 강제한다. DB·OS 미접촉(호출부가 토픽 집합을 만들어 넘긴다 — 헌법 6조).
"""

from __future__ import annotations


def topic_of_filename(file_name: str) -> str:
    """파일명에서 토픽 키(슬러그 첫 토큰)를 뽑는다(순수·결정적).

    `등산_입문_<id>_<제목>.mp4` → `등산`. 언더스코어가 없으면 확장자 뗀 stem 전체가 토픽
    (예: `manifest.json` → `manifest` — 위생 대상이 가드에 노출되도록 그대로 둔다)."""
    stem = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
    return stem.split("_", 1)[0] if stem else ""


def uncovered_topics(corpus_topics: set[str], golden_topics: set[str]) -> list[str]:
    """골든 질의가 커버하지 않는 코퍼스 토픽 목록(정렬·결정적). 비어 있어야 정상."""
    return sorted(corpus_topics - golden_topics)
