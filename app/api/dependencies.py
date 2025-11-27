"""
Dependency Injection for FastAPI.
FastAPI 의존성 주입 관리.
"""
from functools import lru_cache
from fastapi import Depends
from openai import AsyncOpenAI

# --- 신규 클래스 import ---
from app.services.portfolio_processor import PortfolioProcessor
from app.services.retry_executor import RetryExecutor
from app.services.health_aggregator import HealthAggregator
from app.services.query_rewrite_service import QueryRewriteService
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
from app.core.config import settings
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

@lru_cache()
def get_openai_client() -> AsyncOpenAI:
    """OpenAI 클라이언트 싱글톤"""
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

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

@lru_cache()
def get_query_rewrite_service() -> QueryRewriteService:
    """QueryRewriteService 싱글톤"""
    logger.info("Creating QueryRewriteService instance")
    return QueryRewriteService()  # ← 수정: 인자 없이 호출!

# --- Health Aggregator 의존성 주입 방식 수정 ---
def get_health_aggregator(
    mongodb_client: MongoDBClient = Depends(get_mongodb_client_cached),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    reranker_client: RerankerClient = Depends(get_reranker_client)
) -> HealthAggregator:
    """HealthAggregator 인스턴스를 생성하고 의존성을 주입합니다."""
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
    reranker: RerankerClient = Depends(get_reranker_client),
    query_rewrite_service: QueryRewriteService = Depends(get_query_rewrite_service)
) -> SearchService:
    return SearchService(
        embedding_service=embedding_service,
        analysis_service=analysis_service,
        portfolio_repo=portfolio_repo,
        reranker=reranker,
        query_rewrite_service=query_rewrite_service
    )

def get_portfolio_processor(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    ocr_processor: OCRProcessor = Depends(get_ocr_processor)
) -> PortfolioProcessor:
    """PortfolioProcessor 인스턴스 생성"""
    return PortfolioProcessor(
        embedding_service=embedding_service,
        portfolio_repo=portfolio_repo,
        ocr_processor=ocr_processor
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
    get_query_rewrite_service()
    logger.info("Dependencies initialized successfully")

async def shutdown_dependencies():
    logger.info("Shutting down dependencies...")
    mongodb_client = get_mongodb_client_cached()
    await mongodb_client.disconnect()
    logger.info("Dependencies shutdown complete")