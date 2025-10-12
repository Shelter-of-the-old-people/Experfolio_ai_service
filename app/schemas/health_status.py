"""
Defines the data models for health check statuses.
헬스 체크 상태에 대한 데이터 모델을 정의합니다.
"""
from enum import Enum
from pydantic import BaseModel, Field

class Status(str, Enum):
    """
    각 구성 요소의 건강 상태를 나타내는 열거형.
    """
    OK = "ok"
    UNHEALTHY = "unhealthy"

class HealthStatus(BaseModel):
    """
    개별 건강 검진 항목의 결과를 담는 모델.
    """
    status: Status = Field(..., description="구성 요소의 건강 상태")
    message: str = Field(default="", description="상태에 대한 추가적인 메시지")