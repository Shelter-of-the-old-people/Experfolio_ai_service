"""
Defines individual health check strategies for different components of the service.
서비스의 여러 구성 요소에 대한 개별 헬스 체크 전략을 정의합니다.
"""
import asyncio
from abc import ABC, abstractmethod
from openai import OpenAI
from app.schemas.health_status import HealthStatus, Status
from app.core.config import settings

# --- 의존성 주입을 위해 필요한 타입들을 직접 임포트 ---
from app.infrastructure.mongodb_client import MongoDBClient
from app.services.embedding_service import EmbeddingService
from app.infrastructure.reranker_client import RerankerClient
# ---------------------------------------------------

class HealthCheckStrategy(ABC):
    """
    모든 헬스 체크 전략이 구현해야 하는 추상 베이스 클래스(인터페이스).
    """
    @abstractmethod
    async def check(self) -> HealthStatus:
        """
        구성 요소의 건강 상태를 확인하고 HealthStatus를 반환합니다.
        """
        pass

class MongoDBHealthCheck(HealthCheckStrategy):
    """MongoDB 연결 상태를 확인하는 전략."""
    def __init__(self, client: MongoDBClient):
        self.client = client

    async def check(self) -> HealthStatus:
        try:
            if await self.client.ping():
                return HealthStatus(status=Status.OK, message="Connection successful.")
            else:
                return HealthStatus(status=Status.UNHEALTHY, message="Ping to database failed.")
        except Exception as e:
            return HealthStatus(status=Status.UNHEALTHY, message=f"An exception occurred: {e}")

class KUREModelHealthCheck(HealthCheckStrategy):
    """KURE 임베딩 모델의 로드 상태를 확인하는 전략."""
    def __init__(self, service: EmbeddingService):
        self.service = service

    async def check(self) -> HealthStatus:
        try:
            if self.service and self.service._model is not None:
                return HealthStatus(status=Status.OK, message="KURE model is loaded.")
            else:
                return HealthStatus(status=Status.UNHEALTHY, message="KURE model object is None or service not available.")
        except Exception as e:
            return HealthStatus(status=Status.UNHEALTHY, message=f"An exception occurred: {e}")

class RerankerModelHealthCheck(HealthCheckStrategy):
    """Reranker 모델의 로드 상태를 확인하는 전략."""
    def __init__(self, client: RerankerClient):
        self.client = client

    async def check(self) -> HealthStatus:
        try:
            if self.client and self.client._model is not None:
                return HealthStatus(status=Status.OK, message="Reranker model is loaded.")
            else:
                return HealthStatus(status=Status.UNHEALTHY, message="Reranker model object is None or client not available.")
        except Exception as e:
            return HealthStatus(status=Status.UNHEALTHY, message=f"An exception occurred: {e}")

class OpenAIHealthCheck(HealthCheckStrategy):
    """OpenAI API 연결 및 인증 상태를 확인하는 전략."""
    async def check(self) -> HealthStatus:
        try:
            client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=5.0)
            await asyncio.to_thread(client.models.list)
            return HealthStatus(status=Status.OK, message="API is reachable and authenticated.")
        except Exception as e:
            error_message = f"An exception occurred: {type(e).__name__}"
            if "authentication" in str(e).lower():
                error_message = "Authentication failed. Check your OPENAI_API_KEY."
            return HealthStatus(status=Status.UNHEALTHY, message=error_message)