"""
Reranker Client using CrossEncoder.
CrossEncoder를 사용한 검색 결과 재순위 클라이언트.
"""
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RerankerClient:
    """
    검색 결과를 재순위하는 클래스
    CrossEncoder 모델을 사용하여 쿼리-문서 쌍의 관련성을 평가합니다.
    """
    
    def __init__(self, model_name: str = None):
        """
        RerankerClient 초기화
        
        Args:
            model_name: CrossEncoder 모델 이름 (기본값: settings에서 로드)
        """
        self._model_name = model_name or settings.RERANKER_MODEL_NAME
        self._model = None
        
        logger.info(f"RerankerClient initializing with model: {self._model_name}")
        self._load_model()
    
    def _load_model(self) -> None:
        """
        CrossEncoder 모델을 로드합니다.
        
        Raises:
            Exception: 모델 로드 실패 시
        """
        try:
            logger.info("Loading CrossEncoder model...")
            
            self._model = CrossEncoder(
                self._model_name,
                max_length=512,  # 최대 토큰 길이
                device='cpu'  # GPU 사용 시 'cuda'로 변경
            )
            
            logger.info(f"CrossEncoder model loaded successfully: {self._model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {str(e)}")
            raise
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict], 
        top_k: int = 10
    ) -> List[Dict]:
        """
        검색 결과를 재순위합니다.
        
        Args:
            query: 검색 쿼리
            candidates: 후보 문서 리스트 (각 문서는 Dict)
                       'embeddings.searchableText' 키가 있어야 함
            top_k: 반환할 상위 결과 수
        
        Returns:
            List[Dict]: 재순위된 상위 top_k 후보 리스트
        """
        if not candidates:
            logger.warning("No candidates to rerank")
            return []
        
        if len(candidates) <= top_k:
            logger.info(f"Candidates ({len(candidates)}) <= top_k ({top_k}), returning all")
            return candidates
        
        try:
            logger.info(f"Reranking {len(candidates)} candidates to top {top_k}")
            
            # 쿼리-문서 쌍 생성
            pairs = self._prepare_pairs(query, candidates)
            
            # CrossEncoder로 점수 계산
            scores = self._model.predict(pairs)
            
            # 점수와 함께 후보 정렬
            scored_candidates = [
                {**candidate, 'rerank_score': float(score)}
                for candidate, score in zip(candidates, scores)
            ]
            
            # 점수 기준 내림차순 정렬
            scored_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # 상위 top_k 반환
            top_candidates = scored_candidates[:top_k]
            
            logger.info(f"Reranking complete. Top score: {top_candidates[0]['rerank_score']:.4f}")
            
            return top_candidates
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # 실패 시 원본 순서대로 top_k 반환
            return candidates[:top_k]
    
    def _prepare_pairs(
        self, 
        query: str, 
        candidates: List[Dict]
    ) -> List[Tuple[str, str]]:
        """
        쿼리와 문서의 쌍을 생성합니다.
        
        Args:
            query: 검색 쿼리
            candidates: 후보 문서 리스트
        
        Returns:
            List[Tuple[str, str]]: (쿼리, 문서) 쌍의 리스트
        """
        pairs = []
        
        for candidate in candidates:
            # searchableText 추출 (중첩 구조 고려)
            if 'embeddings' in candidate and 'searchableText' in candidate['embeddings']:
                text = candidate['embeddings']['searchableText']
            elif 'searchableText' in candidate:
                text = candidate['searchableText']
            else:
                logger.warning(f"No searchableText found in candidate: {candidate.get('_id', 'unknown')}")
                text = ""
            
            # 텍스트가 너무 길면 잘라내기 (512 토큰 제한)
            if len(text) > 2000:  # 대략 512 토큰
                text = text[:2000]
            
            pairs.append((query, text))
        
        return pairs
    
    @property
    def model_name(self) -> str:
        """
        모델 이름을 반환합니다.
        
        Returns:
            str: 모델 이름
        """
        return self._model_name
