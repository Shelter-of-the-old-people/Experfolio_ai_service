"""
Experfolio AI Service - Main Application
FastAPI 애플리케이션 엔트리포인트
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routers import search, health
from app.api.dependencies import startup_dependencies, shutdown_dependencies, get_batch_service
from app.scheduler.batch_scheduler import initialize_batch_scheduler
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 라이프사이클 관리
    
    Startup:
        - 의존성 초기화 (MongoDB, 모델 로드)
        - 배치 스케줄러 시작
    
    Shutdown:
        - 배치 스케줄러 중지
        - 의존성 정리 (MongoDB 연결 종료)
    """
    # ========== Startup ==========
    logger.info("=" * 70)
    logger.info("Starting Experfolio AI Service...")
    logger.info("=" * 70)
    
    try:
        # 1. 의존성 초기화
        logger.info("Initializing dependencies...")
        await startup_dependencies()
        
        # 2. 배치 스케줄러 초기화 및 시작
        logger.info("Initializing batch scheduler...")
        batch_service = get_batch_service()
        scheduler = initialize_batch_scheduler(batch_service)
        scheduler.start()
        
        logger.info("=" * 70)
        logger.info("✓ Experfolio AI Service started successfully!")
        logger.info(f"✓ API Server: http://{settings.API_HOST}:{settings.API_PORT}")
        logger.info(f"✓ Swagger UI: http://{settings.API_HOST}:{settings.API_PORT}/docs")
        logger.info(f"✓ Next batch run: {scheduler.next_run_time}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
    
    yield
    
    # ========== Shutdown ==========
    logger.info("=" * 70)
    logger.info("Shutting down Experfolio AI Service...")
    logger.info("=" * 70)
    
    try:
        # 1. 배치 스케줄러 중지
        from app.scheduler.batch_scheduler import get_batch_scheduler
        scheduler = get_batch_scheduler()
        scheduler.stop()
        
        # 2. 의존성 정리
        await shutdown_dependencies()
        
        logger.info("=" * 70)
        logger.info("✓ Experfolio AI Service shutdown complete")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


# ============================================
# FastAPI 애플리케이션 생성
# ============================================

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="""
    ## Experfolio AI Service
    
    AI 기반 포트폴리오 검색 및 임베딩 서비스입니다.
    
    ### 주요 기능:
    - **자연어 검색**: GPT-4 기반 의미 검색
    - **벡터 검색**: KURE-v1 임베딩 + MongoDB Vector Search
    - **자동 배치**: 매일 새벽 2시 자동 임베딩 처리
    
    ### 기술 스택:
    - **임베딩**: KURE-v1 (1024차원)
    - **LLM**: GPT-4
    - **재순위**: BGE Reranker v2
    - **데이터베이스**: MongoDB Atlas
    - **OCR**: Tesseract
    
    ### API 사용법:
    1. `/health` - 서비스 상태 확인
    2. `/ai/search` - 포트폴리오 검색
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================
# CORS 설정
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React 개발 서버
        "http://localhost:8080",  # Spring Boot
        "https://experfolio.com",  # 프로덕션 도메인 (예시)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# 라우터 등록
# ============================================

app.include_router(health.router)
app.include_router(search.router)


# ============================================
# 전역 예외 핸들러
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    전역 예외 핸들러
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Request: {request.method} {request.url}")
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc) if settings.LOG_LEVEL == "DEBUG" else "An error occurred"
        }
    )


# ============================================
# 개발 서버 실행
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,  # 개발 모드에서만 사용
        log_level=settings.LOG_LEVEL.lower()
    )
