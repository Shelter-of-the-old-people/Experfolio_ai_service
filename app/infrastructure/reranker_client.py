"""
Reranker Client using CrossEncoder.
CrossEncoder를 사용한 검색 결과 재순위 클라이언트.
"""
import torch
from typing import List, Dict, Tuple, Union
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
        GPU가 사용 가능한 경우 자동으로 사용합니다.
        
        Raises:
            Exception: 모델 로드 실패 시
        """
        try:
            device = self._select_device()
            
            logger.info(f"Loading CrossEncoder model on device: {device}...")
            
            self._model = CrossEncoder(
                self._model_name,
                max_length=512,
                device=device
            )
            
            logger.info(
                f"CrossEncoder model loaded successfully: {self._model_name} "
                f"on {device}"
            )
            
            if device == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(
                    f"GPU Info: {gpu_name}, "
                    f"Total Memory: {gpu_memory:.2f} GB"
                )
            
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {str(e)}")
            raise
    
    def _select_device(self) -> str:
        """
        사용할 디바이스를 선택합니다.
        
        Returns:
            str: 'cuda' 또는 'cpu'
        """
        if settings.FORCE_CPU:
            logger.info("FORCE_CPU=True: Using CPU")
            return 'cpu'
        
        if not settings.USE_GPU:
            logger.info("USE_GPU=False: Using CPU")
            return 'cpu'
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU available: {gpu_count} device(s) detected")
            return 'cuda'
        else:
            logger.warning(
                "GPU not available. Falling back to CPU. "
                "This may result in slower performance."
            )
            return 'cpu'
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict], 
        top_k: int = 10,
        return_all_filtered: bool = False
    ) -> Union[List[Dict], Tuple[List[Dict], List[Dict]]]:
        """
        검색 결과를 재순위하고 점수 임계값으로 필터링합니다.
        
        Args:
            query: 검색 쿼리
            candidates: 후보 문서 리스트 (각 문서는 Dict)
                       'embeddings.searchableText' 키가 있어야 함
            top_k: 반환할 상위 결과 수
            return_all_filtered: True일 경우 (top_k, alternative) 튜플 반환
        
        Returns:
            List[Dict] 또는 Tuple[List[Dict], List[Dict]]:
                - return_all_filtered=False: 상위 top_k 후보 리스트
                - return_all_filtered=True: (상위 top_k, 나머지 필터링된 후보) 튜플
        """
        if not candidates:
            logger.warning("No candidates to rerank")
            if return_all_filtered:
                return [], []
            return []
        
        try:
            logger.info(f"Reranking {len(candidates)} candidates...")
            
            # 1. 쿼리-문서 쌍 생성
            pairs = self._prepare_pairs(query, candidates)
            
            # 2. CrossEncoder로 점수 계산
            scores = self._model.predict(pairs)
            
            # 3. 점수와 함께 후보 리스트 생성
            scored_candidates = [
                {**candidate, 'rerank_score': float(score)}
                for candidate, score in zip(candidates, scores)
            ]
            
            # 4. 점수 임계값으로 필터링
            initial_count = len(scored_candidates)
            filtered_candidates = [
                cand for cand in scored_candidates 
                if cand['rerank_score'] >= settings.RERANKER_SCORE_THRESHOLD
            ]
            
            filtered_count = len(filtered_candidates)
            logger.info(
                f"Reranker filtering: {initial_count} -> {filtered_count} "
                f"(threshold: {settings.RERANKER_SCORE_THRESHOLD})"
            )

            # 5. 점수 기준 내림차순 정렬
            filtered_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # 6. 상위 top_k와 나머지 분리
            top_candidates = filtered_candidates[:top_k]
            
            if top_candidates:
                logger.info(f"Reranking complete. Top score: {top_candidates[0]['rerank_score']:.4f}")
            else:
                logger.info("Reranking complete. No candidates passed the threshold.")
            
            # 7. 반환 형태 결정
            if return_all_filtered:
                alternative_candidates = filtered_candidates[top_k:]
                logger.info(
                    f"Returning top {len(top_candidates)} + alternative {len(alternative_candidates)} candidates"
                )
                return top_candidates, alternative_candidates
            
            return top_candidates
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            if return_all_filtered:
                return candidates[:top_k], []
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
            if 'embeddings' in candidate and 'searchableText' in candidate['embeddings']:
                text = candidate['embeddings']['searchableText']
            elif 'searchableText' in candidate:
                text = candidate['searchableText']
            else:
                logger.warning(f"No searchableText found in candidate: {candidate.get('_id', 'unknown')}")
                text = ""
            
            if len(text) > 2000:
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