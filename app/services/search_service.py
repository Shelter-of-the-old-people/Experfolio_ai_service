"""
Search Service for orchestrating the search process.
검색 프로세스를 오케스트레이션하는 서비스.
"""
import time
from typing import List
from app.services.embedding_service import EmbeddingService
from app.services.analysis_service import AnalysisService
from app.repositories.portfolio_repository import PortfolioRepository
from app.infrastructure.reranker_client import RerankerClient
from app.schemas.response import SearchResponse, CandidateResult
from app.core.config import settings
from app.core.logging import get_logger
from app.core.result import Result, Ok, Err, RateLimitError

logger = get_logger(__name__)


class SearchService:
    """
    검색 비즈니스 로직을 담당하는 서비스
    임베딩, 벡터 검색, 재순위, LLM 분석을 통합합니다.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        analysis_service: AnalysisService,
        portfolio_repo: PortfolioRepository,
        reranker: RerankerClient
    ):
        """
        SearchService 초기화
        
        Args:
            embedding_service: 임베딩 서비스
            analysis_service: LLM 분석 서비스
            portfolio_repo: 포트폴리오 저장소
            reranker: 재순위 클라이언트
        """
        self._embedding_service = embedding_service
        self._analysis_service = analysis_service
        self._portfolio_repo = portfolio_repo
        self._reranker = reranker
        
        logger.info("SearchService initialized")
    
    async def search_portfolios(self, query: str) -> Result:
        """
        포트폴리오 검색 전체 프로세스를 실행합니다.
        
        Args:
            query: 자연어 검색 쿼리
        
        Returns:
            Result:
                - Ok(SearchResponse): 검색 결과
                - Err: 에러 정보
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting search for query: {query[:50]}...")
            
            logger.info("Step 1: Analyzing search intent...")
            intent_result = self._analysis_service.analyze_search_intent(query)
            
            match intent_result:
                case Ok(intent):
                    logger.debug(f"Intent: {intent}")
                case Err():
                    logger.warning(
                        f"Intent analysis failed: {intent_result.error_message}, "
                        "continuing without intent"
                    )
            
            logger.info("Step 2: Embedding query...")
            embedding_result = self._embedding_service.embed_query(query)
            
            match embedding_result:
                case Ok(query_vector):
                    logger.debug(f"Query vector dimension: {len(query_vector)}")
                
                case Err():
                    logger.error(f"Query embedding failed: {embedding_result.error_message}")
                    return embedding_result
            
            logger.info("Step 3: Performing vector search...")
            vector_limit = settings.VECTOR_SEARCH_LIMIT
            
            try:
                search_results = await self._portfolio_repo.vector_search(
                    query_vector, 
                    limit=vector_limit
                )
            except Exception as e:
                logger.error(f"Vector search failed: {str(e)}")
                from app.core.result import NetworkError
                return Err(NetworkError(
                    error=e,
                    context={"query": query[:50]}
                ))
            
            logger.info(f"Vector search returned {len(search_results)} results")
            
            if not search_results:
                logger.warning("No search results found")
                elapsed = time.time() - start_time
                return Ok(SearchResponse(
                    status="success",
                    candidates=[],
                    searchTime=f"{elapsed:.2f}s",
                    totalResults=0
                ))
            
            logger.info("Step 4: Reranking results...")
            top_k = min(settings.RERANK_TOP_K, len(search_results))
            reranked_results = self._reranker.rerank(
                query, 
                search_results[:20],
                top_k=top_k
            )
            logger.info(f"Reranked to top {len(reranked_results)} candidates")
            
            logger.info("Step 5: Analyzing candidate matches...")
            candidates = await self._analyze_candidates(query, reranked_results)
            
            elapsed = time.time() - start_time
            
            response = SearchResponse(
                status="success",
                candidates=candidates,
                searchTime=f"{elapsed:.2f}s",
                totalResults=len(candidates)
            )
            
            logger.info(f"Search completed in {elapsed:.2f}s with {len(candidates)} results")
            
            return Ok(response)
            
        except Exception as e:
            logger.error(f"Unexpected search error: {str(e)}")
            from app.core.result import NetworkError
            return Err(NetworkError(
                error=e,
                context={"query": query[:50]}
            ))
    
    async def _analyze_candidates(
        self, 
        query: str, 
        results: List[dict]
    ) -> List[CandidateResult]:
        """
        각 후보자의 매칭도를 분석합니다.
        실패한 후보자는 제외하고, 성공한 후보자만 반환합니다.
        
        Args:
            query: 검색 쿼리
            results: 재순위된 검색 결과
        
        Returns:
            List[CandidateResult]: 분석된 후보자 목록
        """
        candidates = []
        rate_limit_count = 0
        other_error_count = 0
        
        for i, result in enumerate(results):
            try:
                logger.debug(f"Analyzing candidate {i+1}/{len(results)}")
                
                portfolio_text = result.get('embeddings', {}).get('searchableText', '')
                
                if not portfolio_text:
                    logger.warning(f"No searchable text for candidate: {result.get('userId', 'unknown')}")
                    continue
                
                analysis_result = self._analysis_service.analyze_candidate_match(
                    query, 
                    portfolio_text
                )
                
                match analysis_result:
                    case Ok(analysis):
                        candidate = CandidateResult(
                            userId=result.get('userId', ''),
                            matchScore=float(analysis.get('matchScore', 0.5)),
                            matchReason=analysis.get('matchReason', ''),
                            keywords=analysis.get('keywords', [])
                        )
                        candidates.append(candidate)
                    
                    case Err(error_type=RateLimitError()):
                        rate_limit_count += 1
                        logger.warning(
                            f"Rate limit hit for candidate {i+1}, "
                            f"retry after {analysis_result.retry_delay}s"
                        )
                    
                    case Err():
                        other_error_count += 1
                        logger.error(
                            f"Analysis failed for candidate {i+1}: "
                            f"{analysis_result.error_message}"
                        )
                
            except Exception as e:
                logger.error(f"Unexpected error analyzing candidate {i+1}: {str(e)}")
                continue
        
        logger.info(f"Successfully analyzed {len(candidates)} candidates")
        
        if rate_limit_count > 0:
            logger.warning(f"{rate_limit_count} candidates skipped due to rate limit")
        
        if other_error_count > 0:
            logger.warning(f"{other_error_count} candidates skipped due to errors")
        
        return candidates
