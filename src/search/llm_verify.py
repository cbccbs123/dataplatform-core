"""검색시점 top-3 개별 LLM 검증 (spec 074 — 자연어 무관 L2·노출 직전 정밀 판정).

운영 상태(2026-07-14): 독립지표 재평가로 운영에서 **off** 로 되돌렸다(코드·배선은 보존·L3 재접근
예정 — ADR ``docs/decisions/2026-07-14-llm-verify-off-reeval.md``). 아래는 074 채택 근거 서사이며,
현행 운영 기본값(``SEARCH_LLM_VERIFY_ENABLED_DEFAULT=False``)이 이 재평가를 반영한다.

073(L1·적재시점 aboutness 필터) 후에도 남는 자연어 상위3 무관(실측 34~55%)을, **상위 3 자산만
gemma 가 1건씩 개별 정독 판정**해 제거한다. 측정(2026-07-13)의 결정적 발견 두 가지가 설계를 정한다:

  ① **개별 판정만 작동한다** — 자산당 1건씩 물으면 무관 44.7→0%·연관유지 1.36→1.60(잡음 자리에
     하위 관련이 승격). 10건을 한 프롬프트에 묶는 배치는 판정이 흔들려(개별과 일치율 66~77%) 기각.
  ② **top-3 이 유일한 실용점** — top3×병렬은 동시 30검색 p95 1.06s(실측)로 감당되지만, 전수(20건)
     검증은 동시 4~5검색에 gemma 포화. 전체 깊이는 L1(073) 몫·상위 정밀은 L2 몫(계층 분담).

헌법 정합: 검색시점 LLM 은 021 FR-004 의 029 거버넌스 토글 개정 선례를 따른다(기본 off·opt-in·
온프레미스 단일 seam·temp=0·프롬프트 env 입력 0). 라이브 읽기 경로의 near-tie 섭동은 판정 캐시
(같은 (정규화질의, 자산) 쌍은 첫 판정으로 고정)로 완화한다 — spec §헌법 정합.

폴백(FR-003): 데드라인 초과·judge 예외 시 **전량 폴백**(미검증 원 버킷 그대로·드롭 0·meta 표식).
부분 적용(끝난 판정만 반영)은 타이밍 의존 비결정이라 금지한다(헌법 §3).
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

# 074 판정 프롬프트 — 측정 하니스와 **동일 문면**(측정-구현 일치·"주제가 맞으면 관련" 기준).
# env 의존 입력(날짜·경로·랜덤) 0 — 029 가 021 에서 제거한 비결정성을 재도입하지 않는다.
_JUDGE_PROMPT = """검색어 "{q}" 로 자산을 찾는 사용자에게 아래 자산이 관련 있나? 주제가 맞으면 관련.
자산: {s}
JSON 하나만: {{"related": true 또는 false}}"""

# 프로세스 내 판정 캐시: (norm_query, asset_id) → bool. temp=0 결정적이라 TTL 불요·상한만 관리.
# 반복·인기 질의는 LLM 0회로 수렴한다(FR-004). 테스트는 cache= 주입으로 격리.
_VERDICT_CACHE: dict[tuple[str, str], bool] = {}
# 캐시 동시성 락(리뷰 🔴 — 포탈 /search 는 스레드풀 병렬이라 무락 트리밍이 iter 중 dict 변형
# RuntimeError 를 실제 유발·재현됨). 조회·쓰기·트리밍을 한 락으로 보호한다(임계구역 극소·µs 단위).
_CACHE_LOCK = threading.Lock()


def default_judge(query: str, asset_id: str, summary: str, *, client: Any | None = None) -> bool:
    """자산 1건 개별 판정(측정 프롬프트 그대로·단일 seam·temp=0).

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
    _ = asset_id  # 시그니처 계약(judge_fn(query, asset_id, summary)) — 프롬프트엔 미사용.
    from src.llm.client import complete_json

    out = complete_json(_JUDGE_PROMPT.format(q=query, s=(summary or "")[:120]), client=client)
    if isinstance(out, dict) and isinstance(out.get("related"), bool):
        return out["related"]
    return True  # fail-safe: 불명이면 유지


def _top_candidates(
    buckets: dict[str, list[dict[str, Any]]], top_n: int
) -> list[tuple[str, str]]:
    """전 버킷 행을 (-similarity, id) 합산 정렬해 **중복 제거 상위 top_n** (asset_id, summary).

    측정 하니스와 동일 형상(flat 합산 상위) — 모달리티 섹션 표시와 무관하게 "융합 점수 기준
    가장 먼저 보일 자산"을 검증 대상으로 삼는다.

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
    """상위 top_n 자산을 개별 병렬 판정해 무관을 전 버킷에서 제거한다(FR-002·003·004·006).

    - ``query`` = 판정 프롬프트용 **사용자 원문**(의도 정보 보존 — 측정과 동일).
      ``norm_query`` = 캐시 키(072 정규화 질의 — 표현 변형을 한 키로 흡수).
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

    # 캐시 조회(락 보호) → 미스만 병렬 판정. 데드라인 내 전부 완료돼야 적용(부분 적용 금지 — 결정성).
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
        # ⚠️ with-블록 금지(리뷰 🔴 재현): ThreadPoolExecutor.__exit__ 는 shutdown(wait=True) 라
        # 실행 중 judge 가 끝날 때까지 **폴백 반환 자체가 블록**돼 데드라인이 무력화된다(느린 judge
        # 3s 주입 시 실반환 3s 실측). 비대기 종료로 즉시 반환한다 — 미완 judge 스레드는 백그라운드
        # 에서 소진 후 종료(잔여 위험: gemma 무응답 시 SDK 기본 타임아웃까지 워커 점유 — ADR 후속:
        # llm client 명시 timeout). 결과는 이미 폴백 확정이라 버려진다(부분 적용 없음·결정성 유지).
        ex = _futures.ThreadPoolExecutor(max_workers=len(misses))
        try:
            futs = {ex.submit(jf, query, aid, summ): aid for aid, summ in misses}
            done, not_done = _futures.wait(futs, timeout=deadline_s)
            if not_done or any(f.exception() is not None for f in done):
                # 전량 폴백(FR-003): 미검증 원 버킷 그대로 — 서비스 무중단·드롭 0.
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
        # 완주한 판정만 캐시에 동결(폴백 경로는 캐시 미기록 — 불완전 배치 저장 방지). 락 보호(리뷰 🔴).
        with _CACHE_LOCK:
            for aid, _summ in misses:
                vc[(norm_query, aid)] = verdicts[aid]
            while len(vc) > cache_max:  # FIFO 근사(삽입순 제거·리뷰 지적으로 명칭 정정 — LRU 아님)
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
