"""
Embedding Service using KURE-v1 model.
KURE-v1 모델을 사용한 임베딩 서비스.
"""
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import get_logger
from app.core.result import Result, Ok, Err, InvalidDataError, SystemError

logger = get_logger(__name__)


class EmbeddingService:
    """
    텍스트 임베딩을 생성하는 서비스
    KURE-v1 모델을 사용하여 1024차원 벡터를 생성합니다.
    """
    
    def __init__(self, model_name: str = None):
        """
        EmbeddingService 초기화
        
        Args:
            model_name: 임베딩 모델 이름 (기본값: settings에서 로드)
        """
        self._model_name = model_name or settings.KURE_MODEL_NAME
        self._dimension = 1024
        self._model = None
        
        logger.info(f"EmbeddingService initializing with model: {self._model_name}")
        self._load_model()
    
    def _load_model(self) -> None:
        """
        KURE 모델을 로드합니다.
        
        Raises:
            Exception: 모델 로드 실패 시
        """
        try:
            logger.info("Loading KURE model... (This may take a few minutes on first run)")
            
            self._model = SentenceTransformer(
                self._model_name,
                device='cpu'
            )
            
            logger.info(f"KURE model loaded successfully: {self._model_name}")
            logger.info(f"Model dimension: {self._dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load KURE model: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> Result:
        """
        검색 쿼리를 임베딩합니다.
        
        Args:
            text: 검색 쿼리 텍스트
        
        Returns:
            Result:
                - Ok(List[float]): 1024차원 임베딩 벡터
                - Err: 에러 정보
        """
        if not text or not text.strip():
            return Err(InvalidDataError(
                error=ValueError("Text cannot be empty"),
                context={"text_length": len(text) if text else 0}
            ))
        
        try:
            logger.debug(f"Embedding query (length: {len(text)})")
            
            embedding = self._model.encode(
                text.strip(),
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            embedding_list = embedding.tolist()
            
            logger.debug(f"Query embedding generated: {len(embedding_list)} dimensions")
            
            return Ok(embedding_list)
            
        except MemoryError as e:
            logger.error(f"Query embedding failed (memory): {str(e)}")
            return Err(SystemError(
                error=e,
                context={"text_length": len(text)}
            ))
            
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            return Err(SystemError(
                error=e,
                context={"text_length": len(text)}
            ))
    
    def embed_passage(self, text: str) -> Result:
        """
        문서(passage)를 임베딩합니다.
        배치 처리 시 포트폴리오 텍스트를 임베딩할 때 사용합니다.
        
        Args:
            text: 문서 텍스트
        
        Returns:
            Result:
                - Ok(List[float]): 1024차원 임베딩 벡터
                - Err: 에러 정보
        """
        if not text or not text.strip():
            return Err(InvalidDataError(
                error=ValueError("Text cannot be empty"),
                context={"text_length": len(text) if text else 0}
            ))
        
        try:
            logger.debug(f"Embedding passage (length: {len(text)})")
            
            if len(text) > 20000:
                logger.warning(f"Text too long ({len(text)} chars), truncating to 20000")
                text = text[:20000]
            
            embedding = self._model.encode(
                text.strip(),
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            embedding_list = embedding.tolist()
            
            logger.debug(f"Passage embedding generated: {len(embedding_list)} dimensions")
            
            return Ok(embedding_list)
            
        except MemoryError as e:
            logger.error(f"Passage embedding failed (memory): {str(e)}")
            return Err(SystemError(
                error=e,
                context={"text_length": len(text)}
            ))
            
        except Exception as e:
            logger.error(f"Passage embedding failed: {str(e)}")
            return Err(SystemError(
                error=e,
                context={"text_length": len(text)}
            ))
    
    def embed_batch(self, texts: List[str]) -> Result:
        """
        여러 텍스트를 한 번에 임베딩합니다 (배치 처리).
        
        Args:
            texts: 텍스트 리스트
        
        Returns:
            Result:
                - Ok(List[List[float]]): 임베딩 벡터 리스트
                - Err: 에러 정보
        """
        if not texts:
            return Err(InvalidDataError(
                error=ValueError("Texts list cannot be empty"),
                context={"texts_count": 0}
            ))
        
        try:
            logger.info(f"Batch embedding {len(texts)} texts")
            
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            
            if not valid_texts:
                return Err(InvalidDataError(
                    error=ValueError("No valid texts after preprocessing"),
                    context={"original_count": len(texts), "valid_count": 0}
                ))
            
            embeddings = self._model.encode(
                valid_texts,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=32
            )
            
            embeddings_list = [emb.tolist() for emb in embeddings]
            
            logger.info(f"Batch embedding complete: {len(embeddings_list)} embeddings generated")
            
            return Ok(embeddings_list)
            
        except MemoryError as e:
            logger.error(f"Batch embedding failed (memory): {str(e)}")
            return Err(SystemError(
                error=e,
                context={"texts_count": len(texts)}
            ))
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            return Err(SystemError(
                error=e,
                context={"texts_count": len(texts)}
            ))
    
    @property
    def model_name(self) -> str:
        """모델 이름을 반환합니다."""
        return self._model_name
    
    @property
    def dimension(self) -> int:
        """임베딩 차원을 반환합니다."""
        return self._dimension
