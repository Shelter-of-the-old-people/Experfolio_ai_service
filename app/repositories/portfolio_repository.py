"""
Portfolio Repository for MongoDB operations.
포트폴리오 데이터 접근을 위한 Repository 계층.
"""
from typing import List, Dict, Optional
from datetime import datetime
from bson import ObjectId
from pymongo.errors import PyMongoError
from app.infrastructure.mongodb_client import MongoDBClient
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class PortfolioRepository:
    """
    포트폴리오 데이터 접근을 담당하는 Repository 클래스
    """

    def __init__(self, mongodb_client: MongoDBClient):
        """
        Repository 초기화
        """
        self._mongodb_client = mongodb_client
        self._db = mongodb_client.get_database()
        self._collection = self._db.portfolios
        self._vector_index_name = "kure_vector_index"
        logger.info("PortfolioRepository initialized")

    async def find_needing_embedding(self) -> List[Dict]:
        """
        [기존 메소드 유지]
        임베딩이 필요한 포트폴리오 목록만을 조회합니다.
        (특정 목적을 위해 재사용될 수 있으므로 유지합니다.)
        """
        try:
            query = {"processingStatus.needsEmbedding": True}
            cursor = self._collection.find(query)
            portfolios = await cursor.to_list(length=None)
            logger.info(f"Found {len(portfolios)} portfolios needing embedding")
            return portfolios
        except PyMongoError as e:
            logger.error(f"Error finding portfolios needing embedding: {str(e)}")
            raise

    async def find_portfolios_to_process(self) -> List[Dict]:
        """
        [신규 메소드 추가]
        일일 배치에서 처리해야 할 모든 포트폴리오를 조회합니다.
        (임베딩 필요 또는 OCR 재처리 대상)
        """
        try:
            query = {
                "$or": [
                    {"processingStatus.needsEmbedding": True},
                    {"portfolioItems.attachments.extractionStatus": "failed"}
                ]
            }
            cursor = self._collection.find(query)
            portfolios = await cursor.to_list(length=None)
            logger.info(f"Found {len(portfolios)} portfolios needing processing (new embedding or OCR retry).")
            return portfolios
        except PyMongoError as e:
            logger.error(f"Error finding portfolios to process: {str(e)}")
            raise

    async def find_by_id(self, portfolio_id: str) -> Optional[Dict]:
        """
        ID로 단일 포트폴리오를 조회합니다.
        """
        try:
            portfolio = await self._collection.find_one({"_id": ObjectId(portfolio_id)})
            if portfolio:
                logger.debug(f"Found portfolio: {portfolio_id}")
            else:
                logger.warning(f"Portfolio not found: {portfolio_id}")
            return portfolio
        except Exception as e:
            logger.error(f"Error finding portfolio by ID {portfolio_id}: {str(e)}")
            return None

    async def vector_search(self, query_vector: List[float], limit: int = 50) -> List[Dict]:
        """
        벡터 유사도 검색을 수행합니다.
        """
        try:
            pipeline = self._build_vector_search_pipeline(query_vector, limit)
            cursor = self._collection.aggregate(pipeline)
            results = await cursor.to_list(length=limit)
            logger.info(f"Vector search returned {len(results)} results after filtering")
            return results
        except PyMongoError as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise

    async def update_embeddings_and_status(
        self,
        portfolio_id: str,
        searchable_text: str,
        kure_vector: List[float],
        portfolio_items: List[Dict]
    ) -> bool:
        """
        [신규 메소드] 임베딩, portfolioItems(OCR 상태 포함), 처리 상태를 모두 업데이트합니다.
        """
        try:
            update_data = {
                "$set": {
                    "embeddings.searchableText": searchable_text,
                    "embeddings.kureVector": kure_vector,
                    "embeddings.lastUpdated": datetime.utcnow(),
                    "portfolioItems": portfolio_items, # OCR 상태가 변경되었을 수 있으므로 덮어쓰기
                    "processingStatus.needsEmbedding": False,
                    "processingStatus.lastProcessed": datetime.utcnow(),
                    "updatedAt": datetime.utcnow()
                }
            }
            result = await self._collection.update_one(
                {"_id": ObjectId(portfolio_id)},
                update_data
            )
            if result.modified_count > 0:
                logger.info(f"Successfully updated embeddings and status for portfolio: {portfolio_id}")
                return True
            logger.warning(f"No document modified for portfolio: {portfolio_id}")
            return False
        except PyMongoError as e:
            logger.error(f"Error updating embeddings and status for {portfolio_id}: {str(e)}")
            return False

    async def mark_as_processed(self, portfolio_id: str) -> bool:
        """
        [신규 메소드] 임베딩할 텍스트가 없는 경우, 처리 완료 상태로만 변경합니다.
        """
        try:
            update_data = {
                "$set": {
                    "processingStatus.needsEmbedding": False,
                    "processingStatus.lastProcessed": datetime.utcnow(),
                    "updatedAt": datetime.utcnow()
                }
            }
            result = await self._collection.update_one(
                {"_id": ObjectId(portfolio_id)},
                update_data
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Error marking portfolio {portfolio_id} as processed: {str(e)}")
            return False

    def _build_vector_search_pipeline(
        self,
        query_vector: List[float],
        limit: int
    ) -> List[Dict]:
        """
        MongoDB Vector Search aggregation pipeline을 생성합니다.
        """
        return [
            {
                "$vectorSearch": {
                    "index": self._vector_index_name,
                    "path": "embeddings.kureVector",
                    "queryVector": query_vector,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "score": {"$meta": "vectorSearchScore"},
                    "userId": 1,
                    "embeddings.searchableText": 1,
                    "basicInfo": 1
                }
            },
            {
                "$match": {
                    "score": {"$gte": settings.VECTOR_SEARCH_SCORE_THRESHOLD}
                }
            }
        ]