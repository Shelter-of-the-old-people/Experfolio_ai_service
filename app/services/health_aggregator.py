"""
Aggregates results from multiple health check strategies.
여러 헬스 체크 전략의 결과들을 종합합니다.
"""
import asyncio
from typing import List, Dict
from app.services.health_checks import (
    HealthCheckStrategy,
    MongoDBHealthCheck,
    KUREModelHealthCheck,
    RerankerModelHealthCheck,
    OpenAIHealthCheck
)
from app.schemas.health_status import HealthStatus, Status
from app.core.logging import get_logger

# --- 의존성 주입을 위해 필요한 타입들을 직접 임포트 ---
from app.infrastructure.mongodb_client import MongoDBClient
from app.services.embedding_service import EmbeddingService
from app.infrastructure.reranker_client import RerankerClient
# ---------------------------------------------------

logger = get_logger(__name__)

class HealthAggregator:
    """
    등록된 모든 HealthCheckStrategy를 실행하고 결과를 종합하는 서비스.
    '어떤 검사들을 수행할 것인가'와 '어떻게 구성할 것인가'에 대한 책임을 가진다.
    """
    def __init__(
        self,
        mongodb_client: MongoDBClient,
        embedding_service: EmbeddingService,
        reranker_client: RerankerClient
    ):
        """
        HealthAggregator 초기화.
        필요한 의존성을 주입받아 각 헬스 체크 전략을 생성하고 등록합니다.
        """
        self._strategies: List[HealthCheckStrategy] = [
            MongoDBHealthCheck(client=mongodb_client),
            KUREModelHealthCheck(service=embedding_service),
            RerankerModelHealthCheck(client=reranker_client),
            OpenAIHealthCheck(), # OpenAIHealthCheck는 외부 의존성이 없음
        ]
        logger.info(f"HealthAggregator initialized with {len(self._strategies)} check strategies.")

    async def check_all(self) -> Dict[str, HealthStatus]:
        """
        등록된 모든 헬스 체크를 병렬로 실행하고 결과를 딕셔너리로 반환합니다.
        """
        tasks = {
            strategy.__class__.__name__.replace("HealthCheck", ""): strategy.check()
            for strategy in self._strategies
        }
        
        logger.info(f"Running {len(tasks)} health checks in parallel...")
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        final_results = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, HealthStatus):
                final_results[name] = result
            else:
                logger.error(f"Health check '{name}' raised an unexpected exception: {result}")
                final_results[name] = HealthStatus(
                    status=Status.UNHEALTHY,
                    message=f"Checker failed with exception: {type(result).__name__}"
                )
        
        logger.info("All health checks completed.")
        return final_results