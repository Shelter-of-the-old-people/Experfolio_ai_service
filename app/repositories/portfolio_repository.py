"""
Portfolio Repository for MongoDB operations.
포트폴리오 데이터 접근을 위한 Repository 계층.
"""
from typing import List, Dict, Optional
from datetime import datetime
from bson import ObjectId
from pymongo.errors import DuplicateKeyError, PyMongoError
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
        
        Args:
            mongodb_client: MongoDB 클라이언트 인스턴스
        """
        self._mongodb_client = mongodb_client
        self._db = mongodb_client.get_database()
        self._collection = self._db.portfolios
        self._vector_index_name = "kure_vector_index"
        
        logger.info("PortfolioRepository initialized")
    
    async def find_needing_embedding(self) -> List[Dict]:
        """
        임베딩이 필요한 포트폴리오 목록을 조회합니다.
        
        Returns:
            List[Dict]: 임베딩 필요한 포트폴리오 목록
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
    
    async def find_by_id(self, portfolio_id: str) -> Optional[Dict]:
        """
        ID로 단일 포트폴리오를 조회합니다.
        
        Args:
            portfolio_id: 포트폴리오 ID (ObjectId string)
        
        Returns:
            Optional[Dict]: 포트폴리오 문서 (없으면 None)
        """
        try:
            portfolio = await self._collection.find_one(
                {"_id": ObjectId(portfolio_id)}
            )
            
            if portfolio:
                logger.debug(f"Found portfolio: {portfolio_id}")
            else:
                logger.warning(f"Portfolio not found: {portfolio_id}")
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Error finding portfolio by ID {portfolio_id}: {str(e)}")
            return None
    
    async def vector_search(
        self, 
        query_vector: List[float], 
        limit: int = 50
    ) -> List[Dict]:
        """
        벡터 유사도 검색을 수행합니다.
        
        Args:
            query_vector: 쿼리 임베딩 벡터 (1024차원)
            limit: 반환할 최대 결과 수
        
        Returns:
            List[Dict]: 검색 결과 포트폴리오 목록
        """
        try:
            pipeline = self._build_vector_search_pipeline(query_vector, limit)
            
            cursor = self._collection.aggregate(pipeline)
            results = await cursor.to_list(length=limit)
            
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except PyMongoError as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise
    
    async def update_embeddings(
        self,
        portfolio_id: str,
        searchable_text: str,
        kure_vector: List[float]
    ) -> bool:
        """
        포트폴리오의 임베딩 데이터를 업데이트합니다.
        
        Args:
            portfolio_id: 포트폴리오 ID
            searchable_text: 검색 가능한 텍스트
            kure_vector: KURE 임베딩 벡터
        
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            update_data = {
                "$set": {
                    "embeddings.searchableText": searchable_text,
                    "embeddings.kureVector": kure_vector,
                    "embeddings.lastUpdated": datetime.utcnow(),
                    "processingStatus.needsEmbedding": False,
                    "processingStatus.lastProcessed": datetime.utcnow()
                }
            }
            
            result = await self._collection.update_one(
                {"_id": ObjectId(portfolio_id)},
                update_data
            )
            
            if result.acknowledged and result.modified_count > 0:
                logger.info(f"Successfully updated embeddings for portfolio: {portfolio_id}")
                return True
            else:
                logger.warning(f"No document modified for portfolio: {portfolio_id}")
                return False
                
        except PyMongoError as e:
            logger.error(f"Error updating embeddings for {portfolio_id}: {str(e)}")
            return False
    
    async def mark_for_reprocessing(self, portfolio_id: str) -> bool:
        """
        포트폴리오를 재처리 대상으로 표시합니다.
        
        Args:
            portfolio_id: 포트폴리오 ID
        
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            result = await self._collection.update_one(
                {"_id": ObjectId(portfolio_id)},
                {
                    "$set": {
                        "processingStatus.needsEmbedding": True
                    }
                }
            )
            
            if result.acknowledged and result.modified_count > 0:
                logger.info(f"Marked portfolio for reprocessing: {portfolio_id}")
                return True
            else:
                logger.warning(f"Failed to mark portfolio: {portfolio_id}")
                return False
                
        except PyMongoError as e:
            logger.error(f"Error marking portfolio for reprocessing: {str(e)}")
            return False
    
    async def delete(self, portfolio_id: str) -> bool:
        """
        포트폴리오를 삭제합니다.
        
        Args:
            portfolio_id: 포트폴리오 ID
        
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            result = await self._collection.delete_one(
                {"_id": ObjectId(portfolio_id)}
            )
            
            if result.deleted_count > 0:
                logger.info(f"Deleted portfolio: {portfolio_id}")
                return True
            else:
                logger.warning(f"Portfolio not found for deletion: {portfolio_id}")
                return False
                
        except PyMongoError as e:
            logger.error(f"Error deleting portfolio: {str(e)}")
            return False
    
    def _build_vector_search_pipeline(
        self, 
        query_vector: List[float], 
        limit: int
    ) -> List[Dict]:
        """
        MongoDB Vector Search aggregation pipeline을 생성합니다.
        
        Args:
            query_vector: 쿼리 벡터
            limit: 결과 제한 수
        
        Returns:
            List[Dict]: Aggregation pipeline
        """
        return [
            {
                "$vectorSearch": {
                    "index": self._vector_index_name,
                    "path": "embeddings.kureVector",
                    "queryVector": query_vector,
                    "numCandidates": limit * 10,  # 후보 수 (limit의 10배)
                    "limit": limit
                }
            },
            {
                "$project": {
                    "userId": 1,
                    "embeddings.searchableText": 1,
                    "basicInfo": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
