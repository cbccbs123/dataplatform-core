"""임베딩 저장 차원·CLIP 모델 등 파이프라인 공통 상수."""

# pgvector ``vector(N)`` 목표 차원. SentenceTransformer·CLIP 패딩 모두 이 길이로 맞춘다.
FIX_EMBEDDING_DIMENSION: int = 1536

# 인덱싱·검색이 동일 체크포인트를 써야 벡터 공간이 일치한다.
DEFAULT_CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"

__all__ = ["DEFAULT_CLIP_MODEL_NAME", "FIX_EMBEDDING_DIMENSION"]
