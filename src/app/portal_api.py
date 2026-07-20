"""일반 도메인 포탈 API 진입점 (얇은 심 — 069 US-E FR-E6·A).

종전 이 파일 하나에 1689줄로 있던 포탈 API 를 ``src/app/portal/`` 패키지로 분할했다(인프라·검색·자산·
관리자·관계검토). 하위호환을 위해 이 모듈은 패키지의 ``app`` 을 **재export** 한다 —
``from src.app.portal_api import app`` 진입점과 ``python -m src.app.portal_api`` 실행이 그대로 동작한다.

분할 상세: ``src/app/portal/__init__.py``(app 조립)·``_infra``(DB풀·lifespan·미들웨어)·
``routes_search``·``routes_assets``·``routes_admin``·``routes_review``. 테스트 patch 정본은 각 라우터/인프라
모듈(``src.app.portal.<module>.<name>``)로 이관됐다.
"""

from __future__ import annotations

from src.app.portal import app

__all__ = ["app"]


if __name__ == "__main__":
    # python -m src.app.portal_api 로도 띄울 수 있게(개발 편의).
    import os

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORTAL_API_PORT", "8001")))
