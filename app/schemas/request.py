"""
Request schemas for API endpoints.
API 엔드포인트의 요청 스키마.
"""
from pydantic import BaseModel, Field, validator


class SearchRequest(BaseModel):
    """
    검색 API 요청 스키마
    """
    query: str = Field(
        ..., 
        min_length=1,
        max_length=500,
        description="자연어 검색 쿼리",
        examples=["React 잘하는 신입 개발자", "백엔드 3년 이상 경력자"]
    )
    
    @validator('query')
    def validate_query(cls, v):
        """
        쿼리 검증: 공백 제거 및 최소 길이 확인
        """
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or only whitespace")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "React와 TypeScript 경험이 있는 프론트엔드 개발자"
            }
        }
