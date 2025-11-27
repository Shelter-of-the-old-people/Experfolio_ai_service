"""
Response schemas for API endpoints.
API 엔드포인트의 응답 스키마.
"""
from typing import List, Optional
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


class AlternativeCandidate(BaseModel):
    """
    대안 후보자 (Reranker threshold 통과, LLM 분석 없음)
    """
    userId: str = Field(..., description="사용자 ID (UUID)")
    vector_score: float = Field(..., description="벡터 검색 점수 (0.0-1.0)")
    rerank_score: float = Field(..., description="Reranker 점수")
    
    class Config:
        json_schema_extra = {
            "example": {
                "userId": "550e8400-e29b-41d4-a716-446655440011",
                "vector_score": 0.68,
                "rerank_score": 0.72
            }
        }


class RewrittenQueryInfo(BaseModel):
    """
    쿼리 재작성 정보 (Phase 8)
    """
    original: str = Field(..., description="원본 쿼리")
    rewritten: str = Field(..., description="재작성된 쿼리")
    strategy: str = Field(..., description="사용된 전략 (natural_generation, skip, failed)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="재작성 신뢰도")
    cached: bool = Field(default=False, description="캐시 히트 여부")
    
    class Config:
        json_schema_extra = {
            "example": {
                "original": "블록체인 백엔드 개발자",
                "rewritten": "블록체인 관련 기술을 사용한 백엔드 서버 개발 경험",
                "strategy": "natural_generation",
                "confidence": 0.85,
                "cached": False
            }
        }


class SearchResponse(BaseModel):
    """
    검색 API 응답 스키마
    """
    status: str = Field(..., description="응답 상태 (success/failed)")
    candidates: List[CandidateResult] = Field(
        default_factory=list,
        description="LLM 분석 완료된 상위 후보자 목록"
    )
    alternativeCandidates: List[AlternativeCandidate] = Field(
        default_factory=list,
        description="Reranker threshold 통과한 나머지 후보자 목록 (LLM 분석 없음)"
    )
    searchTime: str = Field(..., description="검색 소요 시간 (예: '1.23s')")
    totalResults: int = Field(default=0, description="검색된 총 결과 수 (candidates + alternativeCandidates)")
    queryRewrite: Optional[RewrittenQueryInfo] = Field(
        default=None, 
        description="쿼리 재작성 정보 (Phase 8)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "queryRewrite": {
                    "original": "블록체인 백엔드 개발자",
                    "rewritten": "블록체인 관련 기술을 사용한 백엔드 서버 개발 경험",
                    "strategy": "natural_generation",
                    "confidence": 0.85,
                    "cached": False
                },
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
                "alternativeCandidates": [
                    {
                        "userId": "550e8400-e29b-41d4-a716-446655440011",
                        "vector_score": 0.68,
                        "rerank_score": 0.72
                    },
                    {
                        "userId": "550e8400-e29b-41d4-a716-446655440012",
                        "vector_score": 0.65,
                        "rerank_score": 0.70
                    }
                ],
                "searchTime": "1.23s",
                "totalResults": 4
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