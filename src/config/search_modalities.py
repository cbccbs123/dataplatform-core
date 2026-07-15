"""검색 모달리티 enum + CSV 파서 단일 출처(D5·P2-29 — 순수·의존 0).

리뷰 P2-29: 검색 진입점 3곳(``portal_api``·``run_search``·``sample_search_api``)이 각자 유효
모달리티 튜플(``("text","image","video","audio")``)과 콤마 구분 파싱을 복제하고 있었다. 이 모듈을
유일 출처로 두고 셋이 공유한다(하나만 고치면 세 곳이 함께 바뀜 — 드리프트 차단).

**책임 분리**: 이 모듈은 **파싱만** 한다(split → strip → 빈 토큰 스킵, 미지정=None).
유효값 밖 거부(검증)는 진입점마다 반환 계약이 달라(포탈 ``HTTPException(400)`` · sample
``200 + {"error"}`` · run_search argparse ``parser.error``) 여기서 통일하지 않고, 각 진입점이
``VALID_SEARCH_MODALITIES`` 로 수행한다. 순수 함수라 import 0·IO 0 — settings 미초기화 환경(순수
단위 테스트·CLI 인자 검증)에서도 안전하게 참조된다.

**대소문자**: 3진입점 원본 파서가 모두 소문자화하지 **않았으므로**(단순 split/strip) 여기서도 하지
않는다 — 대문자/혼합 입력(예: ``"TEXT"``)은 이전처럼 유효값 밖으로 거부된다(동작 불변·US-D 원칙).
"""

from __future__ import annotations

# 검색이 다루는 모달리티 버킷의 유효값(단일 출처). 진입점 3곳이 이 튜플로 미지 모달리티를 거부한다.
VALID_SEARCH_MODALITIES: tuple[str, ...] = ("text", "image", "video", "audio")


def parse_modalities_csv(raw: str | None) -> list[str] | None:
    """콤마 구분 모달리티 문자열 → 라벨 리스트. 미지정/공백이면 ``None``(전체 버킷).

    split → strip → 빈 토큰 스킵. **검증도 소문자화도 하지 않는다** — 유효값 밖 거부는 호출측 몫
    (진입점별 반환 계약이 다름), 대소문자 정규화는 원본 3파서가 하지 않았으므로 동작 보존 위해
    생략한다. 순수·결정적(같은 입력 → 같은 출력).
    """
    if not raw or not raw.strip():
        return None
    items = [m.strip() for m in raw.split(",") if m.strip()]
    return items or None


__all__ = ["VALID_SEARCH_MODALITIES", "parse_modalities_csv"]
