"""검색 결과 **상위 몇 건만** LLM 이 다시 읽어 무관한 것을 노출 직전에 걸러낸다.

**흐름에서의 위치**: 검색·융합·컷이 모두 끝난 뒤 마지막에 붙는 선택적 층이다. 기본은 꺼져 있다.

설계를 정한 두 가지
  ① **한 건씩 묻는다.** 여러 건을 한 프롬프트에 묶으면 판정이 흔들려 같은 자산이 실행마다 다르게
     분류된다.
  ② **상위 소수만 본다.** 전체를 검증하면 동시 검색 몇 건만으로 LLM 이 포화된다. 넓은 범위는
     적재 시점 필터가 맡고, 여기는 맨 앞자리 정밀도만 담당한다.

같은 (정규화 질의, 자산) 쌍은 **첫 판정을 캐시에 고정**한다 — 그래야 같은 검색을 반복해도 순위가
흔들리지 않는다. 마감을 넘기거나 판정이 실패하면 **전량 폴백**한다(끝난 것만 반영하면 타이밍에
따라 결과가 달라진다).
"""

from __future__ import annotations

import concurrent.futures as _futures
import threading
import time
from collections.abc import Callable
from typing import Any

from src.config.search_constants import (
    SEARCH_LLM_VERIFY_CACHE_MAX,
    SEARCH_LLM_VERIFY_DEADLINE_S,
    SEARCH_LLM_VERIFY_TOP_N,
)

# 판정 프롬프트. 날짜·경로·난수 같은 환경 의존 입력을 넣지 않는다 — 넣으면 같은 질의가
# 실행할 때마다 다르게 판정된다.
_JUDGE_PROMPT = """검색어 "{q}" 로 자산을 찾는 사용자에게 아래 자산이 관련 있나? 주제가 맞으면 관련.
자산: {s}
JSON 하나만: {{"related": true 또는 false}}"""

# 판정 캐시: (정규화 질의, 자산) → 관련 여부. temperature 0 이라 같은 쌍은 늘 같은 답이므로
# 만료 시각 없이 개수 상한만 둔다. 반복되는 질의는 LLM 호출이 0으로 수렴한다.
_VERDICT_CACHE: dict[tuple[str, str], bool] = {}
# ⚠️ 캐시 락은 필수다 — 검색 요청이 스레드로 병렬 처리되는데, 락 없이 캐시를 정리하면
# 순회 도중 dict 가 바뀌어 실제로 예외가 났다. 조회·쓰기·정리를 모두 이 락으로 감싼다.
_CACHE_LOCK = threading.Lock()


def default_judge(query: str, asset_id: str, summary: str, *, client: Any | None = None) -> bool:
    """자산 1건이 질의와 관련 있는지 LLM 에 묻는다(단일 seam·temperature 0).

    **외부 호출**이 한 번 일어난다(단일 seam·temperature=0).

    Args:
        query: 사용자 원문 질의(의도가 담긴 쪽을 그대로 쓴다).
        asset_id: 판정 대상 자산. **프롬프트에는 쓰지 않는다** — 주입 seam 시그니처를 맞추기
            위한 자리다(캐시 키는 호출부가 만든다).
        summary: 자산 요약. 앞 120자만 프롬프트에 넣는다.
        client: **테스트용 LLM 클라이언트 주입 seam** — 미주입이면 운영 LLM.

    Returns:
        관련 있으면 True. **스키마 위반·빈 응답도 True** — 판정 실패가 자산을 지우는 쪽으로
        작동하면 안 되기 때문이다(드롭은 명시적 false 일 때만).
    """
    _ = asset_id  # 주입 seam 시그니처를 맞추려는 자리 — 프롬프트에는 쓰지 않는다.
    from src.llm.client import complete_json

    out = complete_json(_JUDGE_PROMPT.format(q=query, s=(summary or "")[:120]), client=client)
    if isinstance(out, dict) and isinstance(out.get("related"), bool):
        return out["related"]
    return True  # 판정 불명이면 남긴다 — 검증 실패가 결과를 지우면 안 된다


def _top_candidates(
    buckets: dict[str, list[dict[str, Any]]], top_n: int
) -> list[tuple[str, str]]:
    """모든 버킷의 행을 점수순으로 합쳐 상위 top_n 개를 뽑는다(중복 자산은 한 번만).

    모달리티 구분과 무관하게 "점수 기준으로 가장 먼저 보일 자산"을 검증 대상으로 삼는다.

    Args:
        buckets: 모달리티별 행 목록.
        top_n: 뽑을 최대 자산 수.

    Returns:
        ``[(asset_id, summary)]`` 최대 top_n 개. 같은 자산이 여러 버킷에 있어도 **한 번만** 넣는다.
    """
    rows = sorted(
        (r for rs in buckets.values() for r in (rs or [])),
        key=lambda r: (-float(r.get("similarity") or 0.0), str(r.get("id") or "")),
    )
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in rows:
        aid = str(r.get("id") or "")
        if not aid or aid in seen:
            continue
        seen.add(aid)
        out.append((aid, str(r.get("summary") or "")))
        if len(out) >= top_n:
            break
    return out


def verify_top_assets(
    buckets: dict[str, list[dict[str, Any]]],
    query: str,
    *,
    norm_query: str,
    judge_fn: Callable[[str, str, str], bool] | None = None,
    cache: dict[tuple[str, str], bool] | None = None,
    top_n: int = SEARCH_LLM_VERIFY_TOP_N,
    deadline_s: float = SEARCH_LLM_VERIFY_DEADLINE_S,
    cache_max: int = SEARCH_LLM_VERIFY_CACHE_MAX,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """상위 top_n 자산을 병렬로 판정해 무관한 것을 모든 버킷에서 제거한다.

    판정에는 **사용자 원문**을, 캐시 키에는 **정규화 질의**를 쓴다 — 판정은 의도가 담긴 원문이
    정확하고, 캐시는 표현이 조금 달라도 같은 것으로 묶여야 적중률이 오른다.

    Args:
        buckets: 모달리티별 행 목록(이 함수는 **드롭만** 하고 순서·점수는 건드리지 않는다).
        query: 판정 프롬프트에 쓸 사용자 원문 질의.
        norm_query: **캐시 키**로 쓸 정규화 질의 — 표현이 조금 달라도 같은 판정을 재사용한다.
        judge_fn: 판정 함수 주입 seam. ``None`` 이면 ``default_judge``(운영 LLM).
        cache: 판정 캐시. ``None`` 이면 프로세스 전역 캐시를 쓴다(테스트는 주입해 격리).
            temperature=0 이라 같은 쌍은 늘 같은 판정이므로 TTL 없이 상한만 관리한다.
        top_n: 검증할 상위 자산 수. 전수 검증은 동시 검색 몇 건만으로 포화된다.
        deadline_s: 전체 판정 마감(초). **넘기면 전량 폴백** — 일부만 적용하면 타이밍에 따라
            결과가 달라져 재현성이 깨지기 때문이다.
        cache_max: 캐시 상한. 넘으면 오래된 항목부터 지운다.

    Returns:
        ``(buckets, meta)``. meta 는 ``{verified, dropped, cache_hits, fallback, latency_ms}``.
        폴백이면 buckets 는 **입력 그대로**(드롭 0).
    """
    t0 = time.perf_counter()
    jf = judge_fn or default_judge
    vc = _VERDICT_CACHE if cache is None else cache
    meta: dict[str, Any] = {"verified": 0, "dropped": 0, "cache_hits": 0, "fallback": False}

    candidates = _top_candidates(buckets, top_n)
    if not candidates:
        meta["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return buckets, meta

    # 캐시에 없는 것만 병렬로 묻는다. **전부 제때 끝나야** 결과를 반영한다 —
    # 끝난 것만 적용하면 같은 검색이 실행할 때마다 다른 결과를 낸다.
    verdicts: dict[str, bool] = {}
    misses: list[tuple[str, str]] = []
    with _CACHE_LOCK:
        cached = {aid: vc.get((norm_query, aid)) for aid, _summ in candidates}
    for aid, summ in candidates:
        hit = cached[aid]
        if hit is None:
            misses.append((aid, summ))
        else:
            verdicts[aid] = hit
            meta["cache_hits"] += 1

    if misses:
        # ⚠️ **``with`` 블록으로 감싸면 안 된다.** ThreadPoolExecutor 의 종료가 실행 중인 작업을
        # 기다리기 때문에, 마감을 넘겨 폴백하려는 순간에도 느린 판정이 끝날 때까지 반환이 막힌다
        # (마감 1.5초인데 3초 걸리는 판정에서 실제로 3초가 걸렸다).
        # 그래서 기다리지 않고 종료한다 — 남은 스레드는 백그라운드에서 끝나고 그 결과는 버려진다
        # (이미 폴백이 확정됐으므로 부분 반영은 일어나지 않는다).
        ex = _futures.ThreadPoolExecutor(max_workers=len(misses))
        try:
            futs = {ex.submit(jf, query, aid, summ): aid for aid, summ in misses}
            done, not_done = _futures.wait(futs, timeout=deadline_s)
            if not_done or any(f.exception() is not None for f in done):
                # 하나라도 못 끝냈거나 실패하면 검증 자체를 포기하고 원본을 그대로 돌려준다.
                meta["fallback"] = True
                meta["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                return buckets, meta
            for f, aid in futs.items():
                verdicts[aid] = bool(f.result())
        except Exception:  # noqa: BLE001 — executor 자체 실패도 폴백(검증이 검색을 깨지 않는다)
            meta["fallback"] = True
            meta["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            return buckets, meta
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        # 끝까지 완주한 판정만 캐시에 남긴다(폴백 경로는 기록하지 않는다 — 불완전한 배치를
        # 저장하면 다음 요청이 그 값을 재사용해 버린다).
        with _CACHE_LOCK:
            for aid, _summ in misses:
                vc[(norm_query, aid)] = verdicts[aid]
            while len(vc) > cache_max:  # 넣은 순서대로 제거한다(사용 빈도는 보지 않는다)
                try:
                    vc.pop(next(iter(vc)))
                except (KeyError, StopIteration, RuntimeError):  # 축출 경합 최후 방어(락 밖 주입 캐시 대비)
                    break

    drop = {aid for aid, ok in verdicts.items() if not ok}
    meta["verified"] = len(candidates)
    meta["dropped"] = len(drop)
    if drop:
        buckets = {
            label: [r for r in (rows or []) if str(r.get("id") or "") not in drop]
            for label, rows in buckets.items()
        }
    meta["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return buckets, meta


__all__ = ["default_judge", "verify_top_assets"]
