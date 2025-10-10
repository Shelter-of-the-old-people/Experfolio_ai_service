"""
Dependency Injection for FastAPI.
FastAPI 의존성 주입 관리.
"""
from functools import lru_cache
from fastapi import Depends
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
    """
    MongoDB 클라이언트 싱글톤 반환
    
    Returns:
        MongoDBClient: MongoDB 클라이언트 인스턴스
    """
    return get_mongodb_client()


@lru_cache()
def get_ocr_processor() -> OCRProcessor:
    """
    OCR Processor 싱글톤 반환
    
    Returns:
        OCRProcessor: OCR 처리기 인스턴스
    """
    logger.debug("Creating OCRProcessor instance")
    return OCRProcessor()


@lru_cache()
def get_file_handler() -> FileHandler:
    """
    File Handler 싱글톤 반환
    
    Returns:
        FileHandler: 파일 핸들러 인스턴스
    """
    logger.debug("Creating FileHandler instance")
    return FileHandler()


@lru_cache()
def get_reranker_client() -> RerankerClient:
    """
    Reranker Client 싱글톤 반환
    
    Returns:
        RerankerClient: Reranker 인스턴스
    """
    logger.debug("Creating RerankerClient instance")
    return RerankerClient()


# ============================================
# Repository Layer Dependencies
# ============================================

def get_portfolio_repository(
    mongodb_client: MongoDBClient = Depends(get_mongodb_client_cached)
) -> PortfolioRepository:
    """
    Portfolio Repository 인스턴스 생성
    
    Args:
        mongodb_client: MongoDB 클라이언트 (Depends로 주입)
    
    Returns:
        PortfolioRepository: 포트폴리오 저장소 인스턴스
    """
    logger.debug("Creating PortfolioRepository instance")
    return PortfolioRepository(mongodb_client)


# ============================================
# Service Layer Dependencies
# ============================================

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """
    Embedding Service 싱글톤 반환
    
    Returns:
        EmbeddingService: 임베딩 서비스 인스턴스
    """
    logger.info("Creating EmbeddingService instance (KURE model loading...)")
    return EmbeddingService()


@lru_cache()
def get_analysis_service() -> AnalysisService:
    """
    Analysis Service 싱글톤 반환
    
    Returns:
        AnalysisService: 분석 서비스 인스턴스
    """
    logger.debug("Creating AnalysisService instance")
    return AnalysisService()


def get_search_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    analysis_service: AnalysisService = Depends(get_analysis_service),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    reranker: RerankerClient = Depends(get_reranker_client)
) -> SearchService:
    """
    Search Service 인스턴스 생성 (의존성 주입)
    
    Args:
        embedding_service: 임베딩 서비스
        analysis_service: 분석 서비스
        portfolio_repo: 포트폴리오 저장소
        reranker: 재순위 클라이언트
    
    Returns:
        SearchService: 검색 서비스 인스턴스
    """
    logger.debug("Creating SearchService instance")
    return SearchService(
        embedding_service=embedding_service,
        analysis_service=analysis_service,
        portfolio_repo=portfolio_repo,
        reranker=reranker
    )


def get_batch_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    ocr_processor: OCRProcessor = Depends(get_ocr_processor),
    file_handler: FileHandler = Depends(get_file_handler)
) -> BatchService:
    """
    Batch Service 인스턴스 생성 (의존성 주입)
    
    Args:
        embedding_service: 임베딩 서비스
        portfolio_repo: 포트폴리오 저장소
        ocr_processor: OCR 처리기
        file_handler: 파일 핸들러
    
    Returns:
        BatchService: 배치 서비스 인스턴스
    """
    logger.debug("Creating BatchService instance")
    return BatchService(
        embedding_service=embedding_service,
        portfolio_repo=portfolio_repo,
        ocr_processor=ocr_processor,
        file_handler=file_handler
    )


# ============================================
# Lifespan Management
# ============================================

async def startup_dependencies():
    """
    애플리케이션 시작 시 실행되는 의존성 초기화
    """
    logger.info("Initializing dependencies...")
    
    # MongoDB 연결
    mongodb_client = get_mongodb_client_cached()
    await mongodb_client.connect()
    await mongodb_client.create_indexes()
    
    # 필수 서비스 로드 (싱글톤 캐싱)
    get_embedding_service()  # KURE 모델 사전 로드
    get_reranker_client()    # Reranker 모델 사전 로드
    
    logger.info("Dependencies initialized successfully")


async def shutdown_dependencies():
    """
    애플리케이션 종료 시 실행되는 정리 작업
    """
    logger.info("Shutting down dependencies...")
    
    # MongoDB 연결 종료
    mongodb_client = get_mongodb_client_cached()
    await mongodb_client.disconnect()
    
    logger.info("Dependencies shutdown complete")