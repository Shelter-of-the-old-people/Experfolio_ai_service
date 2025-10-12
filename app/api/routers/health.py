"""
Health Check Router.
헬스 체크 라우터.
"""
from fastapi import APIRouter, Depends, Response, status
from app.core.config import settings
from app.core.logging import get_logger

# --- 의존성 및 모델 변경 ---
from app.services.health_aggregator import HealthAggregator
from app.api.dependencies import get_health_aggregator
from app.schemas.health_status import Status
# -------------------------

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

@router.get(
    "/health",
    summary="종합 헬스 체크",
    description="서비스의 모든 핵심 구성 요소(DB, AI 모델, 외부 API)의 상태를 종합적으로 확인합니다."
)
async def health_check(
    response: Response,
    health_aggregator: HealthAggregator = Depends(get_health_aggregator)
):
    """
    서비스의 종합 건강 상태를 확인하는 API.

    - **healthy**: 모든 구성 요소가 정상 작동 중입니다. (HTTP 200 OK)
    - **unhealthy**: 하나 이상의 구성 요소에 문제가 있습니다. (HTTP 503 Service Unavailable)

    응답의 `details` 필드에서 각 구성 요소의 상세 상태를 확인할 수 있습니다.
    """
    logger.debug("Comprehensive health check requested.")
    
    details = await health_aggregator.check_all()
    
    overall_status = Status.OK
    for result in details.values():
        if result.status == Status.UNHEALTHY:
            overall_status = Status.UNHEALTHY
            break
    
    if overall_status == Status.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning(f"Health check resulted in UNHEALTHY state. Details: {details}")
    else:
        logger.info("Health check successful: all components are healthy.")

    return {
        "overall_status": overall_status,
        "details": details
    }


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