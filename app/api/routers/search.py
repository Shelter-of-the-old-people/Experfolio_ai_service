"""
Search Router for AI-powered portfolio search.
AI 기반 포트폴리오 검색 라우터.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.request import SearchRequest
from app.schemas.response import SearchResponse, ErrorResponse
from app.services.search_service import SearchService
from app.api.dependencies import get_search_service
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ai", tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="포트폴리오 검색",
    description="자연어 쿼리로 포트폴리오를 검색합니다. AI 기반 의미 검색을 수행합니다."
)
async def search_portfolios(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service)
) -> SearchResponse:
    """
    AI 기반 포트폴리오 검색 API
    
    ## 검색 프로세스:
    1. 검색 의도 분석 (GPT-4)
    2. 쿼리 임베딩 (KURE-v1)
    3. 벡터 유사도 검색 (MongoDB)
    4. 결과 재순위 (CrossEncoder)
    5. 매칭 분석 (GPT-4)
    
    ## 요청 예시:
    ```json
    {
        "query": "React와 TypeScript 경험이 있는 프론트엔드 개발자"
    }
    ```
    
    ## 응답 예시:
    ```json
    {
        "status": "success",
        "candidates": [
            {
                "userId": "550e8400-e29b-41d4-a716-446655440000",
                "matchScore": 0.92,
                "matchReason": "React 프로젝트 3개 경험, TypeScript 능숙",
                "keywords": ["React", "TypeScript", "프론트엔드"]
            }
        ],
        "searchTime": "1.23s",
        "totalResults": 10
    }
    ```
    
    Args:
        request: 검색 요청 (query 포함)
        search_service: 검색 서비스 (의존성 주입)
    
    Returns:
        SearchResponse: 검색 결과
    
    Raises:
        HTTPException: 검색 실패 시
    """
    try:
        logger.info(f"Search request received: {request.query[:50]}...")
        
        # 검색 실행
        response = await search_service.search_portfolios(request.query)
        
        logger.info(f"Search completed: {response.totalResults} results in {response.searchTime}")
        
        return response
        
    except ValueError as e:
        logger.error(f"Invalid search request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed. Please try again later."
        )
