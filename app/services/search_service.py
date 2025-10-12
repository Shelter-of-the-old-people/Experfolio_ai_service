"""
Search Service for orchestrating the search process.
검색 프로세스를 오케스트레이션하는 서비스.
"""
import asyncio
import time
from typing import List, Tuple
from app.services.embedding_service import EmbeddingService
from app.services.analysis_service import AnalysisService
from app.repositories.portfolio_repository import PortfolioRepository
from app.infrastructure.reranker_client import RerankerClient
from app.schemas.response import SearchResponse, CandidateResult
from app.core.config import settings
from app.core.logging import get_logger
from app.core.result import Result, Ok, Err, RateLimitError, InvalidDataError, NetworkError, SystemError

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
        self._embedding_service = embedding_service
        self._analysis_service = analysis_service
        self._portfolio_repo = portfolio_repo
        self._reranker = reranker
        
        logger.info("SearchService initialized")
    
    async def search_portfolios(self, query: str) -> Result:
        start_time = time.time()
        
        try:
            logger.info(f"Search request received for query: '{query[:50]}...'")
            
            embedding_result = self._embedding_service.embed_query(query)
            if isinstance(embedding_result, Err):
                logger.error(f"Query embedding failed: {embedding_result.error_message}")
                return embedding_result
            query_vector = embedding_result.value

            try:
                search_results = await self._portfolio_repo.vector_search(
                    query_vector, 
                    limit=settings.VECTOR_SEARCH_LIMIT
                )
            except Exception as e:
                logger.error(f"Vector search failed: {str(e)}", exc_info=True)
                return Err(NetworkError(error=e, context={"query": query[:50]}))

            logger.info(f"Step 1 (Vector Search): Found {len(search_results)} candidates passing threshold.")
            
            if not search_results:
                elapsed = time.time() - start_time
                logger.info(f"Search completed in {elapsed:.2f}s, no results found at vector search stage.")
                return Ok(SearchResponse(status="success", candidates=[], searchTime=f"{elapsed:.2f}s", totalResults=0))
            
            reranked_results = self._reranker.rerank(
                query, 
                search_results,
                top_k=settings.RERANK_TOP_K 
            )
            logger.info(f"Step 2 (Reranker): Filtered to {len(reranked_results)} candidates.")

            if not reranked_results:
                elapsed = time.time() - start_time
                logger.info(f"Search completed in {elapsed:.2f}s, no results found after reranking.")
                return Ok(SearchResponse(status="success", candidates=[], searchTime=f"{elapsed:.2f}s", totalResults=0))

            final_candidates = await self._analyze_candidates(query, reranked_results)
            logger.info(f"Step 3 (LLM Analysis): Analyzed and finalized {len(final_candidates)} candidates.")

            elapsed = time.time() - start_time
            
            response = SearchResponse(
                status="success",
                candidates=final_candidates,
                searchTime=f"{elapsed:.2f}s",
                totalResults=len(final_candidates)
            )
            
            logger.info(f"Search completed successfully in {elapsed:.2f}s with {len(final_candidates)} results.")
            
            return Ok(response)
            
        except Exception as e:
            logger.error(f"An unexpected error occurred during search: {str(e)}", exc_info=True)
            return Err(SystemError(error=e, context={"query": query[:50]}))

    async def _analyze_candidates(
        self, 
        query: str, 
        results: List[dict]
    ) -> List[CandidateResult]:
        """
        제한된 병렬성(Semaphore)을 사용하여 최종 후보자 목록에 대해 LLM 분석을 수행합니다.
        """
        semaphore = asyncio.Semaphore(3)
        
        async def analyze_with_semaphore(result: dict) -> CandidateResult | None:
            user_id = result.get('userId', 'unknown')
            
            # === 검증용 로그 추가 (시작) ===
            start_mono = time.monotonic()
            logger.info(f"[{start_mono:.2f}s] START analysis for '{user_id}' (waiting for semaphore).")
            # ==============================

            async with semaphore:
                # === 검증용 로그 추가 (실행) ===
                acquired_mono = time.monotonic()
                logger.info(f"[{acquired_mono:.2f}s] RUNNING analysis for '{user_id}' (semaphore acquired).")
                # ==============================

                portfolio_text = result.get('embeddings', {}).get('searchableText', '')
                if not portfolio_text:
                    logger.warning(f"No text for candidate '{user_id}', skipping.")
                    return None

                analysis_result = self._analysis_service.analyze_candidate_match(query, portfolio_text)
                
                # === 검증용 로그 추가 (종료) ===
                end_mono = time.monotonic()
                logger.info(f"[{end_mono:.2f}s] END analysis for '{user_id}' (duration: {end_mono - acquired_mono:.2f}s).")
                # ==============================

                match analysis_result:
                    case Ok(analysis):
                        return CandidateResult(
                            userId=user_id,
                            matchScore=float(analysis.get('matchScore', 0.0)),
                            matchReason=analysis.get('matchReason', 'N/A'),
                            keywords=analysis.get('keywords', [])
                        )
                    case Err():
                        logger.warning(f"Analysis failed for '{user_id}': {analysis_result.error_message}")
                        return None

        tasks = [analyze_with_semaphore(result) for result in results]
        candidate_results = await asyncio.gather(*tasks)
        
        final_candidates = [res for res in candidate_results if res is not None]
        
        return final_candidates