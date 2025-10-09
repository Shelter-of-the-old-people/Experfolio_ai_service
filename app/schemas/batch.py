"""
Batch processing schemas.
배치 처리 관련 스키마.
"""
from typing import List
from pydantic import BaseModel, Field


class BatchResult(BaseModel):
    """
    배치 처리 결과 스키마
    """
    total: int = Field(..., ge=0, description="전체 처리 대상 개수")
    success: int = Field(..., ge=0, description="성공한 개수")
    failed: int = Field(..., ge=0, description="실패한 개수")
    failedIds: List[str] = Field(
        default_factory=list,
        description="실패한 포트폴리오 ID 목록"
    )
    processingTime: str = Field(..., description="총 처리 시간 (예: '15m 30s')")
    
    @property
    def success_rate(self) -> float:
        """성공률 계산 (0.0~1.0)"""
        if self.total == 0:
            return 0.0
        return self.success / self.total
    
    class Config:
        json_schema_extra = {
            "example": {
                "total": 100,
                "success": 95,
                "failed": 5,
                "failedIds": [
                    "64f8a5b2c1d2e3f4a5b6c7d8",
                    "64f8a5b2c1d2e3f4a5b6c7d9"
                ],
                "processingTime": "15m 30s"
            }
        }


class BatchStatus(BaseModel):
    """
    배치 처리 상태 스키마
    """
    isRunning: bool = Field(..., description="배치 실행 중 여부")
    lastRunTime: str = Field(default="", description="마지막 실행 시간")
    nextRunTime: str = Field(default="", description="다음 실행 예정 시간")
    lastResult: BatchResult = Field(default=None, description="마지막 실행 결과")
    
    class Config:
        json_schema_extra = {
            "example": {
                "isRunning": False,
                "lastRunTime": "2024-01-15T02:00:00Z",
                "nextRunTime": "2024-01-16T02:00:00Z",
                "lastResult": {
                    "total": 100,
                    "success": 95,
                    "failed": 5,
                    "failedIds": ["64f8a5b2c1d2e3f4a5b6c7d8"],
                    "processingTime": "15m 30s"
                }
            }
        }
