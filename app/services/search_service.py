"""
Search Service for orchestrating the search process.
검색 프로세스를 오케스트레이션하는 서비스.
"""
import asyncio
import time
from typing import List
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
    
    Rate Limit 안정성을 위한 다층 방어 시스템:
    1. Semaphore를 이용한 동시성 제어
    2. 배치 처리 방식
    3. Rate Limit 에러 시 재시도 로직
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
        
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_ANALYSIS)
        
        logger.info(f"SearchService initialized with:")
        logger.info(f"  - Max concurrent analysis: {settings.MAX_CONCURRENT_ANALYSIS}")
        logger.info(f"  - Batch size: {settings.ANALYSIS_BATCH_SIZE}")
        logger.info(f"  - Rate limit retries: {settings.RATE_LIMIT_MAX_RETRIES}")
    
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
        배치 처리 방식으로 후보자 목록을 분석합니다.
        
        전략:
        1. 전체 후보를 배치 크기로 분할
        2. 각 배치를 순차적으로 처리
        3. 배치 내에서는 Semaphore로 제한된 병렬 처리
        4. Rate Limit 에러 발생 시 재시도
        """
        total_candidates = len(results)
        batch_size = settings.ANALYSIS_BATCH_SIZE
        
        logger.info(
            f"Starting batch analysis for {total_candidates} candidates "
            f"(batch_size={batch_size}, max_concurrent={settings.MAX_CONCURRENT_ANALYSIS})"
        )
        
        all_valid_candidates = []
        total_failed = 0
        
        for batch_idx in range(0, total_candidates, batch_size):
            batch_end = min(batch_idx + batch_size, total_candidates)
            batch_results = results[batch_idx:batch_end]
            
            logger.info(
                f"Processing batch {batch_idx // batch_size + 1} "
                f"(candidates {batch_idx + 1}-{batch_end} of {total_candidates})"
            )
            
            batch_candidates, batch_failed = await self._analyze_batch(
                query=query,
                results=batch_results,
                start_index=batch_idx
            )
            
            all_valid_candidates.extend(batch_candidates)
            total_failed += batch_failed
            
            if batch_end < total_candidates:
                await asyncio.sleep(0.5)
        
        logger.info(
            f"Batch analysis complete: "
            f"success={len(all_valid_candidates)}, failed={total_failed}"
        )
        
        return all_valid_candidates
    
    async def _analyze_batch(
        self,
        query: str,
        results: List[dict],
        start_index: int
    ) -> tuple[List[CandidateResult], int]:
        """
        단일 배치를 분석합니다 (Semaphore로 동시성 제어).
        """
        tasks = [
            self._analyze_single_candidate_with_semaphore(
                query=query,
                result=result,
                index=start_index + idx
            )
            for idx, result in enumerate(results)
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_candidates = []
        failed_count = 0
        
        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Candidate {start_index + idx} analysis raised exception: "
                    f"{type(result).__name__}: {str(result)}"
                )
                failed_count += 1
            elif result is None:
                failed_count += 1
            else:
                valid_candidates.append(result)
        
        return valid_candidates, failed_count
    
    async def _analyze_single_candidate_with_semaphore(
        self,
        query: str,
        result: dict,
        index: int
    ) -> CandidateResult | None:
        """
        Semaphore를 사용하여 단일 후보자를 분석합니다 (동시성 제어).
        """
        async with self._semaphore:
            return await self._analyze_single_candidate_with_retry(
                query=query,
                result=result,
                index=index
            )
    
    async def _analyze_single_candidate_with_retry(
        self,
        query: str,
        result: dict,
        index: int
    ) -> CandidateResult | None:
        """
        단일 후보자를 분석하며, Rate Limit 에러 발생 시 재시도합니다.
        """
        user_id = result.get('userId', 'unknown')
        
        portfolio_text = result.get('embeddings', {}).get('searchableText', '')
        if not portfolio_text:
            logger.warning(f"No text for candidate '{user_id}' (index {index}), skipping.")
            return None
        
        for attempt in range(settings.RATE_LIMIT_MAX_RETRIES):
            try:
                analysis_result = await asyncio.wait_for(
                    self._analysis_service.analyze_candidate_match_async(
                        query,
                        portfolio_text
                    ),
                    timeout=settings.CANDIDATE_ANALYSIS_TIMEOUT
                )
                
                match analysis_result:
                    case Ok(analysis):
                        if attempt > 0:
                            logger.info(
                                f"Candidate '{user_id}' (index {index}) succeeded after {attempt + 1} attempts."
                            )
                        else:
                            logger.debug(f"Successfully analyzed candidate '{user_id}' (index {index}).")
                        
                        return CandidateResult(
                            userId=user_id,
                            matchScore=float(analysis.get('matchScore', 0.0)),
                            matchReason=analysis.get('matchReason', 'N/A'),
                            keywords=analysis.get('keywords', [])
                        )
                    
                    case Err(error_type=RateLimitError()) if attempt < settings.RATE_LIMIT_MAX_RETRIES - 1:
                        wait_time = settings.RATE_LIMIT_INITIAL_DELAY * (settings.RATE_LIMIT_BACKOFF_MULTIPLIER ** attempt)
                        logger.warning(
                            f"Rate limit hit for candidate '{user_id}' (index {index}), "
                            f"attempt {attempt + 1}/{settings.RATE_LIMIT_MAX_RETRIES}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    
                    case Err(error_type=RateLimitError()):
                        logger.error(
                            f"Rate limit hit for candidate '{user_id}' (index {index}) "
                            f"after {settings.RATE_LIMIT_MAX_RETRIES} attempts, giving up."
                        )
                        return None
                    
                    case Err():
                        logger.error(
                            f"Analysis failed for candidate '{user_id}' (index {index}). "
                            f"Error: {analysis_result.error_message}"
                        )
                        return None
            
            except asyncio.TimeoutError:
                logger.warning(
                    f"Candidate {index} analysis timeout after {settings.CANDIDATE_ANALYSIS_TIMEOUT}s "
                    f"(userId: {user_id})"
                )
                return None
            
            except Exception as e:
                logger.error(
                    f"Candidate {index} analysis unexpected error: "
                    f"{type(e).__name__}: {str(e)}"
                )
                return None
        
        return None