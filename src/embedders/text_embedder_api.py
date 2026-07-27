"""062 — API 텍스트 임베딩 seam (온프레미스 OpenAI 호환 ``/v1/embeddings``·bge-m3).

로컬 ``text_embedder.embed_texts`` 와 **대칭**인 저수준 seam — 모델 raw 차원(bge-m3=1024)의 (정규화)
벡터를 반환하고, DB ``vector(1536)`` 패딩은 하지 않는다(다운스트림 ``embedding_text_chunks``/``query_embed``
의 ``pad_embedding_to_storage_dim`` 이 단일 적용). 네트워크(``requests``)는 함수 내부에서 지연 import 한다
— 로컬 백엔드(기본) 환경의 순수성을 보존하기 위함. inference-only·온프레미스(사설망·외부 아님)·결정적
(bge-m3 동일 입력→동일 벡터; GPU 배치 부동소수 미세 비결정은 캐비엇).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

_LOG = logging.getLogger("meta_extract.text_embedder_api")


def _l2_normalize(vec: list[float]) -> list[float]:
    """벡터 길이를 1로 맞춘다(로컬 임베더의 정규화와 같은 의미).

    Args:
        vec: 원본 벡터.

    Returns:
        길이 1인 벡터. **길이가 0인 벡터는 그대로** 돌려준다(0으로 나눌 수 없다).
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _post_embeddings(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
    max_retries: int,
) -> list[dict[str, Any]]:
    """배치 하나를 원격 임베딩 API 로 보내 결과를 받는다.

    Args:
        url: 호출할 엔드포인트.
        headers: 요청 헤더(인증 포함).
        payload: 요청 본문(모델·입력).
        timeout_s: 요청 제한 시간.
        max_retries: 재시도 횟수. **네트워크·응답 형식 오류만 재시도**하고 프로그래밍 오류는
            그대로 올린다 — 코드 버그를 재시도로 덮으면 원인을 못 찾는다.

    Returns:
        응답의 데이터 배열.

    Raises:
        RuntimeError: 모든 재시도가 실패했을 때(마지막 예외를 원인으로 붙인다).
    """
    import requests  # 지연 import — 로컬 백엔드 환경 순수성 보존

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()["data"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            # 네트워크·응답형식(JSONDecodeError=ValueError)·data 키 누락만 재시도(프로그래밍 오류는 전파).
            last_exc = exc
            _LOG.warning("임베딩 API 호출 실패(재시도 %d/%d·url=%s): %r", attempt + 1, max_retries + 1, url, exc)
    raise RuntimeError(f"임베딩 API 호출 실패({max_retries + 1}회): {url}") from last_exc


def embed_texts_api(
    texts: Sequence[str],
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout_s: float = 30.0,
    batch_size: int = 32,
    max_retries: int = 2,
    normalize_embeddings: bool = True,
) -> list[list[float]]:
    """텍스트 배치를 OpenAI 호환 ``/embeddings`` 로 임베딩한다(raw·정규화·**패딩 없음**).

    ⚠️ **응답 순서를 믿지 않는다** — 응답에 담긴 색인 값으로 원래 순서를 복원한다. 서버가 순서를
    바꿔 보내도 청크와 벡터의 짝이 어긋나지 않게 하기 위해서다.

    Args:
        texts: 임베딩할 문자열들. **빈 목록이면 호출하지 않는다**.
        base_url: API 기준 주소.
        model: 쓸 모델 이름.
        api_key: 인증 키. 없으면 인증 헤더를 붙이지 않는다.
        timeout_s: 요청 제한 시간.
        batch_size: 한 요청에 담을 개수. 나눠 보내 요청 하나가 지나치게 커지는 것을 막는다.
        max_retries: 요청별 재시도 횟수.
        normalize_embeddings: 벡터 길이를 1로 맞출지. **서버가 정규화하지 않는 경우를 대비**해
            여기서 한 번 더 맞춘다(적재와 질의가 같은 설정이어야 한다).

    Returns:
        입력 순서대로의 벡터 목록(모델 원래 차원·패딩 없음).

    Raises:
        ValueError: 응답 개수가 요청 개수와 다를 때. **재시도로 삼키지 않는다** — 개수가 안 맞으면
            어느 청크의 벡터인지 알 수 없어, 조용히 넘기면 엉뚱한 짝이 저장된다.
    """
    if not texts:
        return []

    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    out: list[list[float]] = []
    items = list(texts)
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        data = _post_embeddings(url, headers, {"model": model, "input": batch}, timeout_s, max_retries)
        # 배치 내 순서를 index 로 복원(서버 무순서 안전). 개수 불일치는 계약 위반 → 즉시 오류.
        rows = sorted(data, key=lambda d: d.get("index", 0))
        if len(rows) != len(batch):
            raise ValueError(
                f"임베딩 API 응답 개수 불일치: 입력 {len(batch)} != 응답 {len(rows)} (url={url})"
            )
        for r in rows:
            emb = r.get("embedding")
            if not emb:  # 빈/누락 임베딩 → 조용한 0벡터 오염 방지(FR-105·SC-05). 정상 "빈 텍스트"와 구분.
                raise ValueError(f"임베딩 API 응답에 빈 embedding: index={r.get('index')} (url={url})")
            vec = [float(x) for x in emb]
            out.append(_l2_normalize(vec) if normalize_embeddings else vec)
    return out
