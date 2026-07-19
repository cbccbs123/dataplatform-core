"""임베딩 저장 차원·CLIP 모델 등 파이프라인 공통 상수."""

# pgvector ``vector(N)`` 목표 차원. SentenceTransformer·CLIP 패딩 모두 이 길이로 맞춘다.
FIX_EMBEDDING_DIMENSION: int = 1536

# 인덱싱·검색이 동일 체크포인트를 써야 벡터 공간이 일치한다.
DEFAULT_CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"

# ``asset_embedding.channel`` 채널 식별자(검색·관계 후보가 채널을 고를 때 쓰는 상수).
# 'st'=SentenceTransformer(KoSimCSE 등 텍스트), 'clip'=CLIP(이미지·키프레임).
# 채널=모델(공간)이고 백엔드(local/api)는 settings.backend_for_channel 이 직교로 고른다.
# (참고) API 서빙 bge-m3 채널 문자열 "st_api" 는 settings._API_EMBED_CHANNELS 가 단일 출처다
#   — 069 US-F 에서 여기 있던 동값 EMBEDDING_KIND_ST_API 상수(소비 0)를 제거했다.
EMBEDDING_KIND_ST = "st"
EMBEDDING_KIND_CLIP = "clip"

__all__ = [
    "DEFAULT_CLIP_MODEL_NAME",
    "EMBEDDING_KIND_CLIP",
    "EMBEDDING_KIND_ST",
    "FIX_EMBEDDING_DIMENSION",
]
