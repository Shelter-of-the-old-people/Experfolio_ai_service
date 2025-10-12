"""
Dependency Injection for FastAPI.
FastAPI 의존성 주입 관리.
"""
from functools import lru_cache
from fastapi import Depends

# --- 신규 클래스 import ---
from app.services.portfolio_processor import PortfolioProcessor
from app.services.retry_executor import RetryExecutor
from app.services.health_aggregator import HealthAggregator
# -------------------------

from app.services.embedding_service import EmbeddingService
from app.services.analysis_service import AnalysisService
from app.services.search_service import SearchService
from app.services.batch_service import BatchService
from app.repositories.portfolio_repository import PortfolioRepository
from app.infrastructure.mongodb_client import MongoDBClient, get_mongodb_client
from app.infrastructure.ocr_processor import OCRProcessor
from app.infrastructure.file_handler import FileHandler
from app.infrastructure.reranker_client import RerankerClient
from app.core.logging import get_logger

logger = get_logger(__name__)

# ============================================
# Infrastructure Layer Dependencies
# ============================================

@lru_cache()
def get_mongodb_client_cached() -> MongoDBClient:
    return get_mongodb_client()

@lru_cache()
def get_ocr_processor() -> OCRProcessor:
    return OCRProcessor()

@lru_cache()
def get_file_handler() -> FileHandler:
    return FileHandler()

@lru_cache()
def get_reranker_client() -> RerankerClient:
    return RerankerClient()

@lru_cache()
def get_retry_executor() -> RetryExecutor:
    """RetryExecutor 싱글톤 반환"""
    return RetryExecutor()

# ============================================
# Repository Layer Dependencies
# ============================================

def get_portfolio_repository(
    mongodb_client: MongoDBClient = Depends(get_mongodb_client_cached)
) -> PortfolioRepository:
    return PortfolioRepository(mongodb_client)

# ============================================
# Service Layer Dependencies
# ============================================

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    logger.info("Creating EmbeddingService instance (KURE model loading...)")
    return EmbeddingService()

@lru_cache()
def get_analysis_service() -> AnalysisService:
    return AnalysisService()

# --- Health Aggregator 의존성 주입 방식 수정 ---
def get_health_aggregator(
    mongodb_client: MongoDBClient = Depends(get_mongodb_client_cached),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    reranker_client: RerankerClient = Depends(get_reranker_client)
) -> HealthAggregator:
    """HealthAggregator 인스턴스를 생성하고 의존성을 주입합니다."""
    # @lru_cache를 제거하여 매번 새로운 인스턴스를 만들지 않도록 함
    # (FastAPI의 Depends가 캐싱을 관리하므로 lru_cache 불필요)
    logger.debug("Creating HealthAggregator instance.")
    return HealthAggregator(
        mongodb_client=mongodb_client,
        embedding_service=embedding_service,
        reranker_client=reranker_client
    )
# ------------------------------------

def get_search_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    analysis_service: AnalysisService = Depends(get_analysis_service),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    reranker: RerankerClient = Depends(get_reranker_client)
) -> SearchService:
    return SearchService(
        embedding_service=embedding_service,
        analysis_service=analysis_service,
        portfolio_repo=portfolio_repo,
        reranker=reranker
    )

def get_portfolio_processor(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    ocr_processor: OCRProcessor = Depends(get_ocr_processor),
    file_handler: FileHandler = Depends(get_file_handler)
) -> PortfolioProcessor:
    """PortfolioProcessor 인스턴스 생성"""
    return PortfolioProcessor(
        embedding_service=embedding_service,
        portfolio_repo=portfolio_repo,
        ocr_processor=ocr_processor,
        file_handler=file_handler
    )

def get_batch_service(
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    processor: PortfolioProcessor = Depends(get_portfolio_processor),
    executor: RetryExecutor = Depends(get_retry_executor)
) -> BatchService:
    """BatchService 인스턴스 생성"""
    return BatchService(
        portfolio_repo=portfolio_repo,
        processor=processor,
        executor=executor
    )

# ============================================
# Lifespan Management
# ============================================

async def startup_dependencies():
    logger.info("Initializing dependencies...")
    mongodb_client = get_mongodb_client_cached()
    await mongodb_client.connect()
    await mongodb_client.create_indexes()
    get_embedding_service()
    get_reranker_client()
    # Health Aggregator는 요청 시점에 생성되므로 여기서 미리 호출할 필요 없음
    logger.info("Dependencies initialized successfully")

async def shutdown_dependencies():
    logger.info("Shutting down dependencies...")
    mongodb_client = get_mongodb_client_cached()
    await mongodb_client.disconnect()
    logger.info("Dependencies shutdown complete")