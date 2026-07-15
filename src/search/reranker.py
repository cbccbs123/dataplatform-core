"""Cross-encoder reranker(028 평가) — 질의-문서 쌍별 절대 적합도(추론만·헌법 1조).

운영 상태(2026-07-13): A/B 측정 결과 이득 0(nDCG 무개선·지연만 증가)으로 운영 **off**(기본
``OS_RERANK_ENABLED_DEFAULT=False``). 코드·주입 seam(``rerank_fn``)은 보존하며, GPU 이행 시 재평가한다.

bi-encoder 코사인(코퍼스 통계 의존)과 달리 cross-encoder 는 (질의, 문서) **쌍 단독**으로
적합도(sigmoid 0~1)를 출력한다 — 같은 쌍은 코퍼스가 커져도 같은 점수(결정적 forward)라
임계(τ)가 **모델 기준 1회 보정**으로 끝난다(코퍼스 추적 재보정 소멸 — 028 ADR).

모델 로드는 lazy(lru_cache·디바이스 mps→cpu 자동) — 모듈 import 만으로는 무거운 의존을
당기지 않는다(검색 경로의 주입 seam ``rerank_fn`` 기본값으로만 연결). 로드/추론 실패는
**명확한 예외로 전파**한다(silent 폴백 금지 — 평가 결과가 조용히 다른 체제로 바뀌지 않게).
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def get_reranker(model_name: str):
    """CrossEncoder 를 1회 로드해 재사용한다(평가 배치·검색 요청 간 공유).

    디바이스는 mps(맥북 GPU) 가용 시 mps, 아니면 기본(cpu) — 지연이 도입 판정의 관문이라
    가속 가능 환경에서는 자동으로 쓴다. sentence-transformers CrossEncoder 는 num_labels=1
    분류 head 에 기본 sigmoid 활성화를 적용해 predict 가 곧 0~1 확률이다.
    """
    import torch
    from sentence_transformers import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else None
    return CrossEncoder(model_name, max_length=512, device=device)


def score_pairs(query: str, texts: list[str], *, model_name: str) -> list[float]:
    """(질의, 문서) 쌍들의 절대 적합도(0~1)를 배치 채점한다(결정적 — 같은 입력=같은 출력)."""
    if not texts:
        return []
    model = get_reranker(model_name)
    scores = model.predict([(query, t) for t in texts])
    return [float(s) for s in scores]
