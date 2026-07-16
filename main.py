from fastapi import FastAPI

from app.core.config import get_settings
from app.core.lifespan import lifespan
from app.health.router import router as health_router
from app.market.api import router as market_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = get_settings()
    application = FastAPI(
        title=settings.service_name,
        version=settings.version,
        lifespan=lifespan,
    )
    application.include_router(health_router)
    application.include_router(market_router)
    return application


app = create_app()
