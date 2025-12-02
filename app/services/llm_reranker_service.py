"""
LLM Reranker Service using GPT-4o.
GPT-4o를 사용한 의미 기반 재순위 매기기 서비스.
"""
import json
import asyncio
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from app.core.config import settings
from app.core.logging import get_logger
from app.core.result import Result, Ok, Err, RateLimitError, SystemError

logger = get_logger(__name__)


class LLMRerankerService:
    """
    GPT-4o를 사용하여 후보자 목록을 의미 기반으로 재순위 매기는 서비스
    
    역할:
    - 벡터 검색의 100개 결과를 Threshold 기반으로 필터링
    - BGE Reranker를 완전히 대체
    - 의미 기반 관련성 점수 반환 (0.0-1.0)
    - 도메인 용어 및 관련 경험 정확히 평가
    
    특징:
    - 배치 처리 (30개씩)로 Timeout 방지
    - Semaphore로 배치 동시성 제어
    - Threshold 기반 유동적 필터링
    - 짧은 입력 (각 후보 200자 요약)
    """
    
    BATCH_SIZE = 30
    
    RERANK_PROMPT_TEMPLATE = """
You are a recruitment expert evaluating candidate portfolios for job matching.

# Task
Evaluate {num_candidates} candidates and rate their relevance to the search query.
Your scores will determine which candidates advance to detailed analysis.

# Search Query
"{query}"

# Evaluation Criteria
Rate each candidate's relevance on a scale of 0.0 to 1.0:

- **0.9-1.0**: Perfect match
  - All key requirements met
  - Strong, directly relevant experience
  - Clear expertise in the domain

- **0.7-0.8**: Strong match
  - Most requirements met
  - Solid relevant experience
  - Good fit for the role

- **0.5-0.6**: Moderate match
  - Some requirements met
  - Partial relevant experience
  - Potential fit with development

- **0.3-0.4**: Weak match
  - Few requirements met
  - Limited relevant experience
  - Tangential connection only

- **0.0-0.2**: No match
  - Requirements not met
  - Irrelevant experience
  - No clear connection

# Critical Evaluation Guidelines

**1. Semantic Understanding (Most Important)**
- Understand MEANING, not just keywords
- "위성 데이터 분석" IS highly relevant to "항공우주"
- "blockchain development" IS highly relevant to "블록체인"
- "machine learning project" IS highly relevant to "AI 엔지니어"

**2. Domain Intelligence**
- 항공우주 includes: aerospace, satellites, aviation, aircraft, space industry, 위성, 우주
- 블록체인 includes: Ethereum, Solidity, Web3, smart contracts, DeFi, NFT
- AI includes: machine learning, deep learning, TensorFlow, PyTorch, data science

**3. Related Experience Counts**
- Adjacent skills and technologies are valuable
- Don't require exact keyword matches
- Consider the depth and breadth of experience

**4. Language Flexibility**
- English and Korean terms are equivalent
- "aerospace" = "항공우주"
- "satellite" = "인공위성"

**5. Experience Context**
- Projects, internships, coursework all count
- Duration matters but isn't everything
- Passion and initiative are important

# Candidates
{candidates_json}

# Output Format
Respond ONLY with valid JSON (no markdown, no explanations, no extra text):
{{
  "scores": [
    {{"userId": "user-id-1", "score": 0.85}},
    {{"userId": "user-id-2", "score": 0.72}},
    {{"userId": "user-id-3", "score": 0.45}},
    ...
  ]
}}

CRITICAL CONSTRAINTS:
- Output MUST be valid JSON only
- Do NOT include ```json``` markers
- Do NOT include any text outside the JSON object
- Include scores for ALL {num_candidates} candidates
- Scores should reflect true relevance based on semantic understanding
- Be generous with related experience (don't over-filter)
"""
    
    def __init__(self):
        self._llm_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._settings = settings
        
        self._batch_semaphore = asyncio.Semaphore(
            getattr(settings, 'LLM_RERANKER_BATCH_CONCURRENCY', 2)
        )
        
        logger.info(
            f"LLMRerankerService initialized with GPT-4o "
            f"(batch size: {self.BATCH_SIZE}, "
            f"batch concurrency: {getattr(settings, 'LLM_RERANKER_BATCH_CONCURRENCY', 2)})"
        )
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict],
        threshold: Optional[float] = None 
    ) -> Result[List[Dict]]:
        """
        후보자 목록을 배치 단위로 LLM 재순위 매기기 (Threshold 기반)
        
        Args:
            query: 검색 쿼리
            candidates: 후보자 목록
            threshold: 최소 점수 (None이면 설정값 사용)
        
        Returns:
            Result[List[Dict]]: Threshold 통과한 모든 후보
        """
        if not candidates:
            logger.warning("No candidates to rerank")
            return Ok([])
        
        threshold = threshold if threshold is not None else self._settings.LLM_RERANKER_THRESHOLD
        min_candidates = self._settings.LLM_RERANKER_MIN_CANDIDATES
        
        try:
            num_candidates = len(candidates)
            logger.info(
                f"Starting LLM reranking for {num_candidates} candidates "
                f"(batch size: {self.BATCH_SIZE}, threshold: {threshold})"
            )
            
            batches = []
            for i in range(0, num_candidates, self.BATCH_SIZE):
                batch = candidates[i:i+self.BATCH_SIZE]
                batches.append(batch)
            
            logger.info(f"Split into {len(batches)} batches")
            
            tasks = [
                self._rerank_batch(query, batch, batch_idx)
                for batch_idx, batch in enumerate(batches)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_scored_candidates = []
            failed_batches = 0
            
            for batch_idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Batch {batch_idx} failed: {type(result).__name__}: {str(result)}"
                    )
                    failed_batches += 1
                    for c in batches[batch_idx]:
                        c['llm_rerank_score'] = c.get('score', 0.0)
                    all_scored_candidates.extend(batches[batch_idx])
                else:
                    all_scored_candidates.extend(result)
            
            if failed_batches > 0:
                logger.warning(f"{failed_batches}/{len(batches)} batches failed")
            
            all_scored_candidates.sort(
                key=lambda x: x.get('llm_rerank_score', 0.0), 
                reverse=True
            )
            
            passed_candidates = [
                c for c in all_scored_candidates 
                if c.get('llm_rerank_score', 0.0) >= threshold
            ]
            
            if len(passed_candidates) < min_candidates:
                logger.warning(
                    f"Only {len(passed_candidates)} passed threshold {threshold}, "
                    f"using top {min_candidates}"
                )
                passed_candidates = all_scored_candidates[:min_candidates]
            
            logger.info(
                f"LLM reranking completed: {num_candidates} → {len(passed_candidates)} candidates, "
                f"score range: {passed_candidates[0].get('llm_rerank_score', 0):.3f} - "
                f"{passed_candidates[-1].get('llm_rerank_score', 0):.3f}, "
                f"batches: {len(batches) - failed_batches}/{len(batches)} successful"
            )
            
            return Ok(passed_candidates)
            
        except Exception as e:
            logger.error(f"LLM Reranker failed: {str(e)}", exc_info=True)
            return Ok(self._fallback_scoring(candidates, min_candidates))
    
    async def _rerank_batch(
        self, 
        query: str, 
        batch: List[Dict], 
        batch_idx: int
    ) -> List[Dict]:
        """단일 배치 처리 (최대 30개)"""
        
        async with self._batch_semaphore:
            logger.debug(f"Processing batch {batch_idx}: {len(batch)} candidates")
            
            try:
                brief_candidates = [
                    {
                        "userId": c.get('userId', 'unknown'),
                        "brief": self._extract_brief(c),
                        "vector_score": c.get('score', 0.0)
                    }
                    for c in batch
                ]
                
                prompt = self.RERANK_PROMPT_TEMPLATE.format(
                    num_candidates=len(batch),
                    query=query,
                    candidates_json=json.dumps(brief_candidates, ensure_ascii=False, indent=2)
                )
                
                response = await asyncio.wait_for(
                    self._llm_client.chat.completions.create(
                        model=self._settings.LLM_RERANKER_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert recruitment evaluator. Respond only with valid JSON."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=self._settings.LLM_RERANKER_TEMPERATURE,
                        max_tokens=self._settings.LLM_RERANKER_MAX_TOKENS
                    ),
                    timeout=45.0
                )
                
                response_text = response.choices[0].message.content.strip()
                score_map = self._parse_scores(response_text)
                
                for candidate in batch:
                    user_id = candidate.get('userId', 'unknown')
                    candidate['llm_rerank_score'] = score_map.get(user_id, 0.0)
                
                logger.debug(
                    f"Batch {batch_idx} completed: "
                    f"scores [{min(score_map.values()):.3f} - {max(score_map.values()):.3f}]"
                )
                
                return batch
                
            except asyncio.TimeoutError:
                logger.error(f"Batch {batch_idx} timeout after 45s")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Batch {batch_idx} JSON parse error: {e}")
                raise
            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {type(e).__name__}: {str(e)}")
                raise
    
    def _extract_brief(self, candidate: Dict) -> str:
        """후보자 정보에서 간단한 요약 추출"""
        text = candidate.get('embeddings', {}).get('searchableText', '')
        
        if not text:
            return "No portfolio information available"
        
        max_chars = self._settings.LLM_RERANKER_BRIEF_MAX_CHARS
        brief = text[:max_chars]
        brief = ' '.join(brief.split())
        brief = brief.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
        
        return brief
    
    def _parse_scores(self, response_text: str) -> Dict[str, float]:
        """GPT-4 응답에서 점수 파싱"""
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        response_data = json.loads(response_text)
        
        scores = response_data.get('scores', [])
        score_map = {
            item['userId']: float(item['score'])
            for item in scores
            if 'userId' in item and 'score' in item
        }
        
        logger.debug(f"Parsed {len(score_map)} scores from LLM response")
        return score_map
    
    def _fallback_scoring(self, candidates: List[Dict], top_k: int) -> List[Dict]:
        """LLM 호출 실패 시 폴백 로직"""
        logger.warning("Using fallback scoring based on vector_score")
        
        candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        for c in candidates:
            c['llm_rerank_score'] = c.get('score', 0.0)
        
        return candidates[:top_k]