"""영상 장면별 요약 결과 → 영상 전체 요약(reduce).

per-asset 파이프라인에서 ``video_skill`` 이 호출한다. 영상 extract 단계가 장면별 대표프레임을
``image_summarizer`` 로 요약해 둔 ``scene_results`` 를 받아, 시간 흐름(장면 순서)을 보존한 타임라인으로
직렬화한 뒤 단일 LLM 호출(``src.llm.client.complete_json`` seam·temperature=0·온프레미스)로 종합한다.

산출 ``summary`` 는 임베딩·BM25 의 단일 원천이라 image/text summarizer 와 동일하게 매체 문형을 금지하고
내용·주제 중심으로 강제한다(spec 026 FR-001 토픽화).
"""

from __future__ import annotations

from typing import TypedDict

from src.config.settings import get_current_settings
from src.llm.client import complete_json
from src.llm.summary_postprocess import promote_objects_to_keywords


class VideoSummaryResult(TypedDict):
    """영상 전체 요약 결과.

    ``objects`` 는 영상 전반에 걸쳐 등장한 주요 객체를 중복 없이 나열한 것이다.
    이미지 ImageSummaryResult 와 동일한 구조를 공유해 skill 레이어에서 일관되게 처리된다.
    """

    summary: str
    keywords: list[str]
    objects: list[str]


# 요약 토픽화 지시(spec 026 FR-001) — image_summarizer 와 동일 의도. video summary 도 임베딩·BM25
# 의 단일 원천이라 '~영상입니다' 매체 문형이 들어가면 토픽이 아닌 포맷으로 군집된다(F1). 산출 JSON
# 키·파싱은 불변, 이 지시 문구만 추가한다.
_SUMMARY_TOPIC_INSTRUCTION = (
    "- summary 는 매체 자체를 언급하지 말 것 — '이미지입니다/사진입니다/영상입니다/썸네일' 같은 "
    "매체 단어·문형 금지\n"
    "- 담긴 내용·주제·개체를 명사구 중심으로 서술\n"
)


# 049 FR-301/302: v2 reduce 추가 지시(종합). v1 본문 위에 **덧붙이기만** 하므로 v1 경로는 불변
# (FR-102 바이트 동일·회귀 안전판). 장면 단순 나열 대신 영상 전체 주제 + 2~3개 하위 주제를 종합하고,
# 각 장면의 두드러진 키워드를 누락 없이 통합(중복 제거)하도록 지시한다. 키워드 정규화·objects
# 승격은 summarize_video_from_scene_results 의 후처리(summary_postprocess)가 v2 일 때만 추가로 한다.
_V2_REDUCE_EXTRA_INSTRUCTION = (
    "- 장면을 단순 나열하지 말고 영상 전체 주제 + 2~3개 하위 주제를 종합\n"
    "- 각 장면의 두드러진(고유한) 키워드를 누락 없이 통합(중복 제거)\n"
)


def _build_video_summary_prompt(
    timeline_lines: list[str], *, summary_max_chars: int, top_k_keywords: int, v2: bool = False
) -> str:
    """영상 장면 타임라인을 종합하는 reduce 프롬프트를 만든다(순수 — LLM·settings 미접촉).

    함수 내 f-string 에 숨어 있던 프롬프트를 순수 빌더로 노출해 토픽화 지시(FR-001)를 단위로
    가드한다(동작 불변 — 이 빌더 출력이 그대로 LLM 입력). timeline_lines 는 scene 순서를 보존한다.

    049: ``v2=False``(기본) 면 현행 v1 프롬프트를 **바이트 동일**하게 반환한다(FR-102 회귀 안전판).
    ``v2=True`` 면 흐름 중심 요약 지시 직전에 종합 추가 지시(_V2_REDUCE_EXTRA_INSTRUCTION)를 끼워
    넣는다 — v1 본문·장면 결과 꼬리는 그대로 두고 덧붙이기만 한다.
    """
    v2_extra = _V2_REDUCE_EXTRA_INSTRUCTION if v2 else ""
    return (
        "아래는 영상의 장면별 대표프레임 분석 결과다. 전체를 종합해 반드시 JSON만 출력해.\n"
        "형식:\n"
        '{ "summary": "내용·주제 중심 요약", "keywords": ["키워드1"], "objects": ["객체1"] }\n'
        f"- summary는 {summary_max_chars}자 이내\n"
        f"- keywords는 핵심 키워드 최대 {top_k_keywords}개 (한국어)\n"
        "- objects는 영상 전반의 주요 객체를 일반 명사 형태 한국어로 중복 없이 나열\n"
        + _SUMMARY_TOPIC_INSTRUCTION
        + v2_extra
        + "장면 순서를 반영해 흐름 중심으로 요약하고, 개수/비율/합계 같은 통계 표현은 금지.\n\n"
        + "장면 결과:\n"
        + "\n".join(timeline_lines)
    )


def summarize_video_from_scene_results(
    scene_results: list[dict],
) -> VideoSummaryResult:
    """장면별 이미지 요약 결과를 종합해 영상 전체 요약을 생성한다.

    scene_results 각 항목은 최소한 아래 키를 포함해야 한다.
    - scene_index, start_sec, end_sec, frame_sec
    - summary: {"summary": str, "keywords": list[str], "objects": list[str]}

    각 장면을 타임라인 텍스트로 직렬화한 뒤 단일 LLM 호출로 reduce 한다.
    장면 순서(시간 흐름)를 반영한 요약을 위해 timeline_lines 는 scene_results 순서를 보존한다.
    ``jpeg_bytes`` 는 이미 extract 단계에서 제거되어 있어야 하며, 여기서는 참조하지 않는다.
    """
    cfg = get_current_settings()
    if not scene_results:
        return {"summary": "", "keywords": [], "objects": []}

    timeline_lines: list[str] = []
    for item in scene_results:
        si = item.get("scene_index", 0)
        ss = item.get("start_sec", 0.0)
        es = item.get("end_sec", 0.0)
        fs = item.get("frame_sec", 0.0)
        s = item.get("summary", {})
        if not isinstance(s, dict):
            s = {}
        summary = str(s.get("summary", "")).strip()
        keywords = s.get("keywords", [])
        objects = s.get("objects", [])
        if not isinstance(keywords, list):
            keywords = []
        if not isinstance(objects, list):
            objects = []
        kw_text = ", ".join(str(k).strip() for k in keywords if str(k).strip())
        obj_text = ", ".join(str(o).strip() for o in objects if str(o).strip())
        timeline_lines.append(
            f"[scene {si}] {ss:.2f}s~{es:.2f}s (대표 {fs:.2f}s): "
            f"summary={summary} | keywords={kw_text} | objects={obj_text}"
        )

    # 049: 토글(vlm_summary_prompt_v2)을 빌더에 v2= 로 전달한다. False(기본)면 v1 프롬프트·v1 키워드
    # 루프가 그대로 돌아 출력이 현행과 바이트 동일하다(FR-102 회귀 안전판). 빌더는 settings 미접촉 순수
    # 함수라 토글 해소는 여기에서만 한다(image_summarizer 와 동형·plan P2).
    v2 = cfg.vlm.summary_prompt_v2
    prompt = _build_video_summary_prompt(
        timeline_lines,
        summary_max_chars=cfg.summary_max_chars,
        top_k_keywords=cfg.top_k_keywords,
        v2=v2,
    )

    data = complete_json(prompt)

    summary = str(data.get("summary", "")).strip()[: cfg.summary_max_chars]

    keywords_raw = data.get("keywords", [])
    if not isinstance(keywords_raw, list):
        keywords_raw = []

    objects_raw = data.get("objects", [])
    if not isinstance(objects_raw, list):
        objects_raw = []
    objects: list[str] = []
    for obj in objects_raw:
        o = str(obj).strip()
        if o and o not in objects:
            objects.append(o)

    if v2:
        # v2: 결정적 후처리 — 키워드 정규화(generic 제거) + objects 를 검색 키워드로 승격(top_k cap).
        keywords = promote_objects_to_keywords(
            keywords_raw, objects, limit=cfg.top_k_keywords
        )
    else:
        # v1: 현행 inline 루프(dedup·top_k cap·objects 미승격) — 바이트 동일 보존.
        keywords = []
        for kw in keywords_raw:
            k = str(kw).strip()
            if k and k not in keywords:
                keywords.append(k)
            if len(keywords) >= cfg.top_k_keywords:
                break

    return {"summary": summary, "keywords": keywords, "objects": objects}

