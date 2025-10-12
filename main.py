"""
Experfolio AI Service - Main Application
FastAPI 애플리케이션 엔트리포인트
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routers import search, health
from app.api.dependencies import (
    startup_dependencies, 
    shutdown_dependencies,
    # --- get_batch_service를 직접 사용하지 않고, 개별 컴포넌트를 가져옴 ---
    get_portfolio_repository,
    get_embedding_service,
    get_ocr_processor,
    get_file_handler,
    get_retry_executor,
    get_portfolio_processor,
    get_mongodb_client_cached
)
from app.services.batch_service import BatchService
from app.services.portfolio_processor import PortfolioProcessor # import 추가
from app.scheduler.batch_scheduler import initialize_batch_scheduler
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 라이프사이클 관리
    """
    logger.info("=" * 70)
    logger.info("Starting Experfolio AI Service...")
    logger.info("=" * 70)
    
    try:
        logger.info("Initializing dependencies...")
        await startup_dependencies()
        
        # --- 배치 스케줄러 초기화 로직 수정 ---
        # BatchService를 구성하는 부품들을 명시적으로 주입
        logger.info("Initializing batch components for scheduler...")
        db_client = get_mongodb_client_cached()
        portfolio_repo = get_portfolio_repository(mongodb_client=db_client)
        
        processor = get_portfolio_processor(
            embedding_service=get_embedding_service(),
            portfolio_repo=portfolio_repo,
            ocr_processor=get_ocr_processor(),
            file_handler=get_file_handler()
        )
        
        executor = get_retry_executor()
        
        batch_service = BatchService(
            portfolio_repo=portfolio_repo,
            processor=processor,
            executor=executor
        )
        
        scheduler = initialize_batch_scheduler(batch_service)
        scheduler.start()
        # ------------------------------------
        
        logger.info("=" * 70)
        logger.info("✓ Experfolio AI Service started successfully!")
        logger.info(f"✓ API Server: http://{settings.API_HOST}:{settings.API_PORT}")
        logger.info(f"✓ Swagger UI: http://{settings.API_HOST}:{settings.API_PORT}/docs")
        logger.info(f"✓ Next batch run: {scheduler.next_run_time}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info("=" * 70)
    logger.info("Shutting down Experfolio AI Service...")
    logger.info("=" * 70)
    
    try:
        from app.scheduler.batch_scheduler import get_batch_scheduler
        scheduler = get_batch_scheduler()
        scheduler.stop()
        
        await shutdown_dependencies()
        
        logger.info("=" * 70)
        logger.info("✓ Experfolio AI Service shutdown complete")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="...", # 설명 생략
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "https://experfolio.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(search.router)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc) if settings.LOG_LEVEL == "DEBUG" else "An unexpected error occurred."
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )