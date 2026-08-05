from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth.config import get_auth_settings
from app.auth.middleware import AuthenticationMiddleware
from app.auth.router import router as auth_router
from app.core.config import get_settings
from app.core.lifespan import lifespan
from app.dashboard.router import router as dashboard_router
from app.health.router import router as health_router
from app.market.api import router as market_router
from app.ml.protected.router import router as protected_ml_router
from app.optimization.router import router as optimization_router
from app.paper.api import router as paper_router
from app.paper.deployment_router import router as paper_deployment_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = get_settings()
    auth_settings = get_auth_settings()
    application = FastAPI(
        title=settings.service_name,
        version=settings.version,
        lifespan=lifespan,
        docs_url=None if auth_settings.enforcement_enabled else "/docs",
        redoc_url=None if auth_settings.enforcement_enabled else "/redoc",
        openapi_url=None if auth_settings.enforcement_enabled else "/openapi.json",
    )
    application.add_middleware(AuthenticationMiddleware, settings=auth_settings)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "X-CSRF-Token"],
        expose_headers=["X-CSRF-Token"],
    )
    application.include_router(auth_router)
    application.include_router(health_router)
    application.include_router(market_router)
    application.include_router(paper_router)
    application.include_router(paper_deployment_router)
    application.include_router(dashboard_router)
    application.include_router(optimization_router)
    application.include_router(protected_ml_router)
    return application


app = create_app()
