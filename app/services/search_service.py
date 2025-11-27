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
from app.schemas.response import SearchResponse, CandidateResult, AlternativeCandidate
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
        reranker: RerankerClient,
        query_rewrite_service
    ):
        self._embedding_service = embedding_service
        self._analysis_service = analysis_service
        self._portfolio_repo = portfolio_repo
        self._reranker = reranker
        self._query_rewrite_service = query_rewrite_service
        
        logger.info("SearchService initialized with QueryRewriteService")
    
    async def search_portfolios(self, query: str, enable_rewrite: bool = True) -> Result:
        start_time = time.time()
        
        try:
            logger.info(f"Search request received for query: '{query[:50]}...'")
            
            # 1. 쿼리 재작성
            search_query = query
            query_rewrite_info = None
            
            if enable_rewrite and settings.QUERY_REWRITE_ENABLED:
                rewrite_result = await self._query_rewrite_service.rewrite_query(
                    query, 
                    {},
                    enable=True
                )
                
                if isinstance(rewrite_result, Ok):
                    rewritten = rewrite_result.value
                    search_query = rewritten.rewritten
                    query_rewrite_info = rewritten
                    
                    logger.info(
                        f"Query rewritten: '{query[:50]}...' → '{search_query[:50]}...' "
                        f"(strategy: {rewritten.strategy}, cached: {rewritten.cached})"
                    )
                else:
                    logger.warning(f"Query rewrite failed, using original: {rewrite_result.error_message}")
            else:
                logger.debug("Query rewrite disabled")
            
            # 2. 쿼리 임베딩
            embedding_result = self._embedding_service.embed_query(search_query)
            if isinstance(embedding_result, Err):
                logger.error(f"Query embedding failed: {embedding_result.error_message}")
                return embedding_result
            query_vector = embedding_result.value

            # 3. 벡터 검색
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
                return Ok(SearchResponse(
                    status="success", 
                    candidates=[], 
                    alternativeCandidates=[],
                    searchTime=f"{elapsed:.2f}s", 
                    totalResults=0,
                    queryRewrite=query_rewrite_info
                ))
            
            # 4. Reranker 단계 (threshold 통과한 전체 후보 받기)
            rerank_result = self._reranker.rerank(
                search_query,
                search_results,
                top_k=settings.RERANK_TOP_K,
                return_all_filtered=True
            )
            
            # Tuple 언패킹
            if isinstance(rerank_result, tuple):
                reranked_results, alternative_results = rerank_result
            else:
                reranked_results = rerank_result
                alternative_results = []
            
            logger.info(
                f"Step 2 (Reranker): Top {len(reranked_results)} candidates + "
                f"{len(alternative_results)} alternative candidates."
            )

            if not reranked_results:
                elapsed = time.time() - start_time
                logger.info(f"Search completed in {elapsed:.2f}s, no results found after reranking.")
                return Ok(SearchResponse(
                    status="success", 
                    candidates=[], 
                    alternativeCandidates=[],
                    searchTime=f"{elapsed:.2f}s", 
                    totalResults=0,
                    queryRewrite=query_rewrite_info
                ))

            # 5. LLM 분석 (Top 10만)
            final_candidates = await self._analyze_candidates(search_query, reranked_results)
            logger.info(f"Step 3 (LLM Analysis): Analyzed and finalized {len(final_candidates)} candidates.")

            # 6. AlternativeCandidates 생성
            alternative_candidates = [
                AlternativeCandidate(
                    userId=cand['userId'],
                    vector_score=cand.get('score', 0.0),
                    rerank_score=cand['rerank_score']
                )
                for cand in alternative_results
                if ((cand['score'] >= 0.75 and cand['rerank_score'] >= 0.8) or cand['score'] >= 0.75 or cand['rerank_score'] >= 0.8)
            ]
            
            logger.info(f"Created {len(alternative_candidates)} alternative candidates.")

            elapsed = time.time() - start_time
            
            total_results = len(final_candidates) + len(alternative_candidates)
            
            response = SearchResponse(
                status="success",
                candidates=final_candidates,
                alternativeCandidates=alternative_candidates,
                searchTime=f"{elapsed:.2f}s",
                totalResults=total_results,
                queryRewrite=query_rewrite_info
            )
            
            logger.info(
                f"Search completed successfully in {elapsed:.2f}s with {total_results} total results "
                f"({len(final_candidates)} main + {len(alternative_candidates)} alternative). "
                f"(original: '{query[:30]}...', rewritten: '{search_query[:30]}...')"
            )
            
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
        병렬로 최종 후보자 목록에 대해 LLM 분석을 수행합니다.
        """
        logger.info(f"Starting parallel analysis for {len(results)} candidates")
        
        tasks = [
            self._analyze_single_candidate_with_timeout(
                query=query,
                result=result,
                index=idx
            )
            for idx, result in enumerate(results)
        ]
        
        candidate_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_candidates = []
        failed_count = 0
        
        for idx, result in enumerate(candidate_results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Candidate {idx} analysis failed: "
                    f"{type(result).__name__}: {str(result)}"
                )
                failed_count += 1
            elif result is None:
                logger.warning(f"Candidate {idx} returned None")
                failed_count += 1
            else:
                valid_candidates.append(result)
        
        logger.info(
            f"Parallel analysis complete: "
            f"success={len(valid_candidates)}, failed={failed_count}"
        )
        
        return valid_candidates
    
    async def _analyze_single_candidate_with_timeout(
        self,
        query: str,
        result: dict,
        index: int,
        timeout: float = 10.0
    ) -> CandidateResult | None:
        """
        단일 후보자를 분석합니다 (타임아웃 포함).
        """
        user_id = result.get('userId', 'unknown')
        
        try:
            portfolio_text = result.get('embeddings', {}).get('searchableText', '')
            if not portfolio_text:
                logger.warning(f"No text for candidate '{user_id}' (index {index}), skipping.")
                return None
            
            analysis_result = await asyncio.wait_for(
                self._analysis_service.analyze_candidate_match_async(
                    query,
                    portfolio_text
                ),
                timeout=timeout
            )
            
            match analysis_result:
                case Ok(analysis):
                    logger.debug(f"Successfully analyzed candidate '{user_id}' (index {index}).")
                    return CandidateResult(
                        userId=user_id,
                        matchScore=float(analysis.get('matchScore', 0.0)),
                        matchReason=analysis.get('matchReason', 'N/A'),
                        keywords=analysis.get('keywords', [])
                    )
                case Err(error_type=RateLimitError()):
                    logger.warning(
                        f"Rate limit hit for candidate '{user_id}' (index {index}), skipping. "
                        f"Details: {analysis_result.error_message}"
                    )
                    return None
                case Err():
                    logger.error(
                        f"Analysis failed for candidate '{user_id}' (index {index}), skipping. "
                        f"Error: {analysis_result.error_message}"
                    )
                    return None
        
        except asyncio.TimeoutError:
            logger.warning(
                f"Candidate {index} analysis timeout after {timeout}s "
                f"(userId: {user_id})"
            )
            return None
        
        except Exception as e:
            logger.error(
                f"Candidate {index} analysis unexpected error: "
                f"{type(e).__name__}: {str(e)}"
            )
            return None