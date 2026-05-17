import structlog
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles
from pathlib import Path

from fastapi import FastAPI
from app.config import get_settings

from app.routers.estimations_route import router as router
from app.routers.sessions_route import router as sessions_router

def _prefix_module(logger, method, event_dict):
    module = event_dict.pop("module", None)
    if module:
        event_dict["event"] = f"[{module}] {event_dict.get('event', '')}"
    return event_dict


def configure_logging() -> None:
    """Set up structlog: JSON in production, human-readable in development."""
    settings = get_settings()

    if settings.APP_ENV == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.processors.CallsiteParameterAdder(
                [structlog.processors.CallsiteParameter.MODULE],
            ),
            _prefix_module,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    configure_logging()
    log = structlog.get_logger()
    settings = get_settings()
    log.info("application_started", environment=settings.APP_ENV)
    yield
    log.info("application_shutdown")



app = FastAPI(
    title="Software Estimation CAG Service",
    description="AI-powered software estimation service using Cache Augmented Generation architecture",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(prefix="/api/v1", router=router)
app.include_router(prefix="/api/v1", router=sessions_router)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

@app.get("/health")
def health():
    #logging for debugging
    print("Health check endpoint was called.")
    return {"status": "healthy"}

@app.get("/")
def main():
    # logging for debugging
    print("Root endpoint was called.")
    return {"message": "Please check the API documentation at endpoint /docs."}
