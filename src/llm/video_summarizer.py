from __future__ import annotations

import json
from typing import TypedDict

from openai import OpenAI

from src.config.settings import get_current_settings


class VideoSummaryResult(TypedDict):
    summary: str
    keywords: list[str]
    objects: list[str]


def _call_openai_json(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float = 0.0,
) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": raw, "keywords": [], "objects": []}


def summarize_video_from_scene_results(
    scene_results: list[dict],
) -> VideoSummaryResult:
    """
    장면별 이미지 요약 결과를 종합해 영상 전체 요약을 생성한다.

    scene_results 각 항목은 최소한 아래 키를 포함해야 한다.
    - scene_index, start_sec, end_sec, frame_sec
    - summary: {"summary": str, "keywords": list[str], "objects": list[str]}
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

    prompt = (
        "아래는 영상의 장면별 대표프레임 분석 결과다. 전체를 종합해 반드시 JSON만 출력해.\n"
        "형식:\n"
        '{ "summary": "영상 전체 요약", "keywords": ["키워드1"], "objects": ["객체1"] }\n'
        f"- summary는 {cfg.summary_max_chars}자 이내\n"
        f"- keywords는 핵심 키워드 최대 {cfg.top_k_keywords}개 (한국어)\n"
        "- objects는 영상 전반의 주요 객체를 일반 명사 형태 한국어로 중복 없이 나열\n"
        "장면 순서를 반영해 흐름 중심으로 요약하고, 개수/비율/합계 같은 통계 표현은 금지.\n\n"
        f"장면 결과:\n" + "\n".join(timeline_lines)
    )

    client = OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)
    data = _call_openai_json(client, model=cfg.meta_model, prompt=prompt)

    summary = str(data.get("summary", "")).strip()[: cfg.summary_max_chars]

    keywords_raw = data.get("keywords", [])
    if not isinstance(keywords_raw, list):
        keywords_raw = []
    keywords: list[str] = []
    for kw in keywords_raw:
        k = str(kw).strip()
        if k and k not in keywords:
            keywords.append(k)
        if len(keywords) >= cfg.top_k_keywords:
            break

    objects_raw = data.get("objects", [])
    if not isinstance(objects_raw, list):
        objects_raw = []
    objects: list[str] = []
    for obj in objects_raw:
        o = str(obj).strip()
        if o and o not in objects:
            objects.append(o)

    return {"summary": summary, "keywords": keywords, "objects": objects}

