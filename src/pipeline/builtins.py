"""v2 기본 전략을 DEFAULT_REGISTRY 에 등록(import 시 부수효과).

태그 규약: 'onprem_llm'=온프레미스 LLM 사용, 'deterministic'=결정적.
by_modality 는 외부 LLM 을 쓰지 않으므로 'external_llm' 태그가 없다(의료 정책 통과).
"""
from __future__ import annotations

from src.classify import cascade
from src.dispatch.dispatcher import dispatch_embed, dispatch_extract_meta
from src.dispatch.types import ExtractContext
from src.pipeline.registry import DEFAULT_REGISTRY, StrategyRegistry
from src.registry.asset_persist import finalize_asset


def _classify_cascade_v1(ctx: ExtractContext):
    """ClassifyStage 어댑터 — cascade.classify(file_path, modality) 를 ctx 기반으로 감쌈."""
    return cascade.classify(ctx.file_path, ctx.modality)


def register_defaults(registry: StrategyRegistry) -> None:
    registry.register("classify", "cascade_v1", _classify_cascade_v1, tags={"onprem_llm"})
    registry.register("extract", "by_modality", dispatch_extract_meta, tags={"onprem_llm"})
    registry.register("embed", "by_modality", dispatch_embed, tags={"deterministic"})
    registry.register("persist", "asset_upsert", finalize_asset)


register_defaults(DEFAULT_REGISTRY)
