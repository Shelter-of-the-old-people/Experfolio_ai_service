"""
Response schemas for API endpoints.
API 엔드포인트의 응답 스키마.
"""
from typing import List
from pydantic import BaseModel, Field


class CandidateResult(BaseModel):
    """
    검색된 후보자 결과
    """
    userId: str = Field(..., description="사용자 ID (UUID)")
    matchScore: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="매칭 점수 (0.0~1.0)"
    )
    matchReason: str = Field(..., description="매칭 이유/분석 근거")
    keywords: List[str] = Field(..., description="추출된 주요 키워드")
    
    class Config:
        json_schema_extra = {
            "example": {
                "userId": "550e8400-e29b-41d4-a716-446655440000",
                "matchScore": 0.92,
                "matchReason": "React 프로젝트 3개 경험, TypeScript 능숙",
                "keywords": ["React", "TypeScript", "프론트엔드", "프로젝트"]
            }
        }


class SearchResponse(BaseModel):
    """
    검색 API 응답 스키마
    """
    status: str = Field(..., description="응답 상태 (success/failed)")
    candidates: List[CandidateResult] = Field(
        default_factory=list,
        description="검색된 후보자 목록"
    )
    searchTime: str = Field(..., description="검색 소요 시간 (예: '1.23s')")
    totalResults: int = Field(default=0, description="검색된 총 결과 수")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "candidates": [
                    {
                        "userId": "550e8400-e29b-41d4-a716-446655440000",
                        "matchScore": 0.92,
                        "matchReason": "React 프로젝트 3개 경험, TypeScript 능숙",
                        "keywords": ["React", "TypeScript", "프론트엔드"]
                    },
                    {
                        "userId": "550e8400-e29b-41d4-a716-446655440001",
                        "matchScore": 0.85,
                        "matchReason": "React 사용 경험 2년, 포트폴리오 우수",
                        "keywords": ["React", "프론트엔드", "웹개발"]
                    }
                ],
                "searchTime": "1.23s",
                "totalResults": 2
            }
        }


class HealthResponse(BaseModel):
    """
    Health Check 응답 스키마
    """
    status: str = Field(..., description="서비스 상태")
    version: str = Field(..., description="API 버전")
    mongodb: str = Field(..., description="MongoDB 연결 상태")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "mongodb": "connected"
            }
        }


class ErrorResponse(BaseModel):
    """
    에러 응답 스키마
    """
    status: str = Field(default="failed", description="응답 상태")
    error: str = Field(..., description="에러 메시지")
    detail: str = Field(default="", description="상세 에러 정보")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "failed",
                "error": "Invalid query",
                "detail": "Query cannot be empty"
            }
        }
