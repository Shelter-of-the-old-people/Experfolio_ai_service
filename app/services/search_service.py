"""
Search Service for orchestrating the search process.
검색 프로세스를 오케스트레이션하는 서비스.
"""
import asyncio
import time
from typing import List, Optional
from app.services.embedding_service import EmbeddingService
from app.services.analysis_service import AnalysisService
from app.repositories.portfolio_repository import PortfolioRepository
from app.services.llm_reranker_service import LLMRerankerService
from app.schemas.response import SearchResponse, CandidateResult, AlternativeCandidate
from app.core.config import settings
from app.core.logging import get_logger
from app.core.result import Result, Ok, Err, RateLimitError, InvalidDataError, NetworkError, SystemError

logger = get_logger(__name__)


class SearchService:
    """
    검색 비즈니스 로직을 담당하는 서비스
    
    Phase 8.5: 
    - BGE Reranker 제거, LLM Reranker로 완전 대체
    - Threshold 기반 필터링
    - Top 10개만 상세 분석
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        analysis_service: AnalysisService,
        portfolio_repo: PortfolioRepository,
        query_rewrite_service,
        llm_reranker_service: LLMRerankerService
    ):
        self._embedding_service = embedding_service
        self._analysis_service = analysis_service
        self._portfolio_repo = portfolio_repo
        self._query_rewrite_service = query_rewrite_service
        self._llm_reranker_service = llm_reranker_service
        
        logger.info("SearchService initialized (LLM Reranker only, Top 10 analysis)")
    
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
                        f"Query rewritten: '{query[:30]}...' → '{search_query[:30]}...' "
                        f"(strategy: {rewritten.strategy})"
                    )
                else:
                    logger.warning(f"Query rewrite failed: {rewrite_result.error_message}")
            
            # 2. 쿼리 임베딩
            embedding_result = self._embedding_service.embed_query(search_query)
            if isinstance(embedding_result, Err):
                logger.error(f"Query embedding failed: {embedding_result.error_message}")
                return embedding_result
            query_vector = embedding_result.value

            # 3. 벡터 검색 (100개)
            try:
                search_results = await self._portfolio_repo.vector_search(
                    query_vector, 
                    limit=settings.VECTOR_SEARCH_LIMIT
                )
            except Exception as e:
                logger.error(f"Vector search failed: {str(e)}", exc_info=True)
                return Err(NetworkError(error=e, context={"query": query[:50]}))

            logger.info(f"Step 1 (Vector Search): Found {len(search_results)} candidates")
            
            if not search_results:
                elapsed = time.time() - start_time
                return Ok(self._empty_response(elapsed, query_rewrite_info))
            
            # 4. LLM Reranker (Threshold 기반)
            llm_rerank_result = await self._llm_reranker_service.rerank(
                search_query,
                search_results,
                threshold=settings.LLM_RERANKER_THRESHOLD
            )
            
            if isinstance(llm_rerank_result, Err):
                logger.error(f"LLM Reranker failed: {llm_rerank_result.error_message}")
                llm_reranked = sorted(
                    search_results, 
                    key=lambda x: x.get('score', 0.0), 
                    reverse=True
                )[:settings.LLM_RERANKER_MIN_CANDIDATES]
                
                for c in llm_reranked:
                    c['llm_rerank_score'] = c.get('score', 0.0)
            else:
                llm_reranked = llm_rerank_result.value
            
            logger.info(
                f"Step 2 (LLM Reranker): {len(llm_reranked)} candidates passed threshold "
                f"(score range: {llm_reranked[0].get('llm_rerank_score', 0):.3f} - "
                f"{llm_reranked[-1].get('llm_rerank_score', 0):.3f})"
            )

            if not llm_reranked:
                elapsed = time.time() - start_time
                return Ok(self._empty_response(elapsed, query_rewrite_info))

            # 5. LLM 상세 분석 (Top 10개만!)
            analysis_top_k = min(settings.LLM_ANALYSIS_TOP_K, len(llm_reranked))
            top_for_analysis = llm_reranked[:analysis_top_k]
            
            analyzed_candidates = await self._analyze_candidates(
                search_query, 
                top_for_analysis
            )
            
            logger.info(
                f"Step 3 (LLM Analysis): Analyzed {len(analyzed_candidates)}/{len(llm_reranked)} candidates"
            )

            # 6. matchScore 기준 최종 정렬
            analyzed_candidates.sort(key=lambda x: x.matchScore, reverse=True)
            
            # 7. 응답 생성
            candidates = analyzed_candidates[:10]
            
            # alternativeCandidates: 11위~ 모두 점수만!
            alternative_candidates = []
            for candidate in llm_reranked[analysis_top_k:]:
                alternative_candidates.append(AlternativeCandidate(
                    userId=candidate['userId'],
                    matchScore=None,
                    matchReason=None,
                    keywords=None,
                    vector_score=candidate.get('score', 0.0),
                    llm_rerank_score=candidate.get('llm_rerank_score', 0.0)
                ))
            
            logger.info(
                f"Created {len(alternative_candidates)} alternative candidates (score-only)"
            )

            elapsed = time.time() - start_time
            total_results = len(candidates) + len(alternative_candidates)
            
            response = SearchResponse(
                status="success",
                candidates=candidates,
                alternativeCandidates=alternative_candidates,
                searchTime=f"{elapsed:.2f}s",
                totalResults=total_results,
                queryRewrite=query_rewrite_info
            )
            
            logger.info(
                f"Search completed successfully in {elapsed:.2f}s with {total_results} total results "
                f"({len(candidates)} main + {len(alternative_candidates)} alternative)"
            )
            
            return Ok(response)
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            return Err(SystemError(error=e, context={"query": query[:50]}))

    def _empty_response(self, elapsed: float, query_rewrite_info) -> SearchResponse:
        """빈 응답 생성"""
        return SearchResponse(
            status="success", 
            candidates=[], 
            alternativeCandidates=[],
            searchTime=f"{elapsed:.2f}s", 
            totalResults=0,
            queryRewrite=query_rewrite_info
        )

    async def _analyze_candidates(
        self, 
        query: str, 
        results: List[dict]
    ) -> List[CandidateResult]:
        """병렬로 최종 후보자 목록에 대해 LLM 분석을 수행합니다."""
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
    ) -> Optional[CandidateResult]:
        """단일 후보자를 분석합니다 (타임아웃 포함)."""
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
                        f"Rate limit hit for candidate '{user_id}' (index {index}), skipping."
                    )
                    return None
                case Err():
                    logger.error(
                        f"Analysis failed for candidate '{user_id}' (index {index}), skipping."
                    )
                    return None
        
        except asyncio.TimeoutError:
            logger.warning(f"Candidate {index} analysis timeout after {timeout}s (userId: {user_id})")
            return None
        
        except Exception as e:
            logger.error(f"Candidate {index} analysis unexpected error: {type(e).__name__}: {str(e)}")
            return None