"""
MongoDB Client using Motor (async driver).
Motor를 사용한 비동기 MongoDB 클라이언트.
"""
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class MongoDBClient:
    """
    MongoDB 비동기 클라이언트 클래스
    Motor를 사용하여 MongoDB와 비동기로 통신합니다.
    """
    
    def __init__(
        self, 
        connection_string: Optional[str] = None,
        database_name: Optional[str] = None
    ):
        """
        MongoDB 클라이언트 초기화
        
        Args:
            connection_string: MongoDB 연결 URI (기본값: settings에서 로드)
            database_name: 데이터베이스 이름 (기본값: settings에서 로드)
        """
        self._connection_string = connection_string or settings.MONGODB_URI
        self._database_name = database_name or settings.MONGODB_DATABASE
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        
        logger.info(f"MongoDB client initialized for database: {self._database_name}")
    
    async def connect(self) -> None:
        """
        MongoDB에 연결합니다.
        
        Raises:
            ConnectionFailure: 연결 실패 시
        """
        try:
            logger.info("Connecting to MongoDB...")
            
            self._client = AsyncIOMotorClient(
                self._connection_string,
                serverSelectionTimeoutMS=5000,  # 5초 타임아웃
                connectTimeoutMS=10000,  # 10초 연결 타임아웃
                maxPoolSize=50,  # 최대 연결 풀 크기
                minPoolSize=10,  # 최소 연결 풀 크기
            )
            
            self._db = self._client[self._database_name]
            
            # 연결 테스트
            await self._client.admin.command('ping')
            
            logger.info("Successfully connected to MongoDB")
            logger.info(f"Database: {self._database_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise ConnectionFailure(f"MongoDB connection failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during MongoDB connection: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """
        MongoDB 연결을 종료합니다.
        """
        if self._client:
            logger.info("Disconnecting from MongoDB...")
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed")
    
    def get_database(self) -> AsyncIOMotorDatabase:
        """
        데이터베이스 인스턴스를 반환합니다.
        
        Returns:
            AsyncIOMotorDatabase: 데이터베이스 인스턴스
        
        Raises:
            RuntimeError: 연결되지 않은 상태에서 호출 시
        """
        if self._db is None:
            raise RuntimeError(
                "Database not connected. Call connect() first."
            )
        return self._db
    
    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """
        컬렉션 인스턴스를 반환합니다.
        
        Args:
            collection_name: 컬렉션 이름
        
        Returns:
            AsyncIOMotorCollection: 컬렉션 인스턴스
        
        Raises:
            RuntimeError: 연결되지 않은 상태에서 호출 시
        """
        db = self.get_database()
        return db[collection_name]
    
    async def ping(self) -> bool:
        """
        MongoDB 연결 상태를 확인합니다.
        
        Returns:
            bool: 연결 성공 시 True
        """
        try:
            if self._client is None:
                return False
            
            await self._client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB ping failed: {str(e)}")
            return False
    
    async def create_indexes(self) -> None:
        """
        필요한 인덱스를 생성합니다.
        """
        try:
            collection = self.get_collection("portfolios")
            
            # userId 유니크 인덱스
            await collection.create_index("userId", unique=True)
            logger.info("Created unique index on userId")
            
            # processingStatus.needsEmbedding 인덱스 (배치 처리용)
            await collection.create_index("processingStatus.needsEmbedding")
            logger.info("Created index on processingStatus.needsEmbedding")
            
            # basicInfo.gpa 인덱스 (필터링용)
            await collection.create_index("basicInfo.gpa")
            logger.info("Created index on basicInfo.gpa")
            
            logger.info("All indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            # 인덱스 생성 실패는 치명적이지 않으므로 예외를 다시 발생시키지 않음
    
    @property
    def is_connected(self) -> bool:
        """
        연결 상태를 반환합니다.
        
        Returns:
            bool: 연결된 상태면 True
        """
        return self._client is not None and self._db is not None


# 싱글톤 인스턴스
_mongodb_client: Optional[MongoDBClient] = None


def get_mongodb_client() -> MongoDBClient:
    """
    MongoDB 클라이언트 싱글톤 인스턴스를 반환합니다.
    
    Returns:
        MongoDBClient: MongoDB 클라이언트 인스턴스
    """
    global _mongodb_client
    
    if _mongodb_client is None:
        _mongodb_client = MongoDBClient()
    
    return _mongodb_client
