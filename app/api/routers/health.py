"""
Health Check Router.
헬스 체크 라우터.
"""
from fastapi import APIRouter, Depends
from app.schemas.response import HealthResponse
from app.infrastructure.mongodb_client import MongoDBClient
from app.api.dependencies import get_mongodb_client_cached
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="헬스 체크",
    description="서비스 상태를 확인합니다."
)
async def health_check(
    mongodb_client: MongoDBClient = Depends(get_mongodb_client_cached)
) -> HealthResponse:
    """
    서비스 헬스 체크 API
    
    ## 확인 항목:
    - 서비스 실행 상태
    - MongoDB 연결 상태
    - API 버전 정보
    
    ## 응답 예시:
    ```json
    {
        "status": "healthy",
        "version": "1.0.0",
        "mongodb": "connected"
    }
    ```
    
    Args:
        mongodb_client: MongoDB 클라이언트 (의존성 주입)
    
    Returns:
        HealthResponse: 헬스 체크 결과
    """
    logger.debug("Health check requested")
    
    # MongoDB 연결 상태 확인
    mongodb_status = "connected" if await mongodb_client.ping() else "disconnected"
    
    # 전체 상태 판단
    overall_status = "healthy" if mongodb_status == "connected" else "unhealthy"
    
    response = HealthResponse(
        status=overall_status,
        version=settings.API_VERSION,
        mongodb=mongodb_status
    )
    
    logger.debug(f"Health check result: {overall_status}")
    
    return response


@router.get(
    "/",
    summary="루트 엔드포인트",
    description="API 정보를 반환합니다."
)
async def root():
    """
    루트 엔드포인트
    
    Returns:
        Dict: API 기본 정보
    """
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }
