from typing import Dict
from openai import AsyncOpenAI
import time

class QueryRewriteService:
    """
    GPT-4 단일 전략 쿼리 재작성 서비스
    """
    
    def __init__(self, llm_client: AsyncOpenAI, settings):
        self._llm_client = llm_client
        self._settings = settings
        self._cache = {}
    
    async def rewrite_query(
        self, 
        original_query: str, 
        intent: Dict,
        enable: bool = True
    ):
        """
        GPT-4로 자연스러운 문장 생성
        
        Args:
            original_query: 원본 쿼리
            intent: analyze_search_intent 결과
            enable: 재작성 활성화 여부
        
        Returns:
            Result[RewrittenQueryInfo]: 재작성된 쿼리 또는 에러
        """
        from app.schemas.response import RewrittenQueryInfo
        from app.core.result import Ok
        from app.core.logging import get_logger
        
        logger = get_logger(__name__)
        
        try:
            if not enable or len(original_query.strip()) < self._settings.QUERY_REWRITE_MIN_LENGTH:
                return Ok(RewrittenQueryInfo(
                    original=original_query,
                    rewritten=original_query,
                    strategy="skip",
                    confidence=1.0,
                    cached=False
                ))
            
            if original_query in self._cache:
                cached_result = self._cache[original_query]
                cached_result.cached = True
                logger.info(f"Cache hit for query: {original_query}")
                return Ok(cached_result)
            
            rewritten = await self._generate_natural_query(original_query, intent)
            
            result = RewrittenQueryInfo(
                original=original_query,
                rewritten=rewritten,
                strategy="natural_generation",
                confidence=0.85,
                cached=False
            )
            
            if len(self._cache) < self._settings.QUERY_REWRITE_CACHE_SIZE:
                self._cache[original_query] = result
            
            logger.info(f"Query rewritten: '{original_query}' → '{rewritten}'")
            return Ok(result)
            
        except Exception as e:
            logger.error(f"Query rewrite failed: {str(e)}")
            return Ok(RewrittenQueryInfo(
                original=original_query,
                rewritten=original_query,
                strategy="failed",
                confidence=1.0,
                cached=False
            ))
    
    async def _generate_natural_query(
        self, 
        original: str, 
        intent: Dict
    ) -> str:
        """
        GPT-4로 자연스러운 문장 생성
        """
        from app.core.logging import get_logger
        logger = get_logger(__name__)
        
        prompt = f"""You are an expert at optimizing recruitment search queries for portfolio-based talent search.

Your task: Rewrite the query to be more search-friendly while preserving ALL keywords and original meaning.

--- REASONING PROCESS (Follow these steps internally) ---

Step 1: Identify ALL Keywords
- Extract every meaningful word: technologies, skills, job titles, experience levels, adjectives
- Example: "React나 TypeScript 경험이 있는 프론트엔드 개발자"
  → Keywords: React, TypeScript, 경험, 프론트엔드, 개발자

Step 2: Understand Core Meaning
- What is being searched for? (person, skill, experience)
- Example: "경험이 있는 개발자" = searching for a developer WITH experience

Step 3: Expand Naturally
- Keep ALL keywords
- Replace commas with natural conjunctions (및, 와, 그리고, 또는)
- Add minimal related keywords if helpful
- Keep it natural and readable

Step 4: Validate
✅ All keywords preserved?
✅ Original meaning unchanged?
✅ Natural conjunctions instead of commas?
✅ 50-100 characters?
✅ No over-expansion?

--- CORE RULES ---

Rule 1: 100% Keyword Preservation (CRITICAL)
- Preserve EVERY keyword from original query
- Technologies: React, TypeScript, Figma, Python...
- Job roles: 개발자, 디자이너, 마케터, 기획자...
- Adjectives: 능숙한, 경험 있는, 3년...
- ALL must be kept or replaced with synonyms only

Examples:
✅ "개발자" → "개발자" or "개발 경험자" (synonym OK)
✅ "경험 있는" → "경험을 갖춘" (natural form OK)
❌ "개발자" → "개발" (meaning changed!)
❌ "디자이너" → "디자인" (person → activity, wrong!)

Rule 2: 100% Meaning Preservation (CRITICAL)
- Do NOT change what is being searched for
- "찾는 대상" must stay the same

Examples:
✅ "React 개발자" → "React를 활용하는 개발자" (still searching for developer)
❌ "React 개발자" → "React 개발 및 구축" (changed to activities, wrong!)
✅ "능숙한 디자이너" → "능숙하게 다루는 디자이너" (still searching for designer)
❌ "능숙한 디자이너" → "디자인 및 프로토타이핑" (lost designer, wrong!)

Rule 3: Natural Expansion Only
- Expansion = adding related helpful keywords
- Transformation = changing structure/meaning (FORBIDDEN)

✅ Allowed expansion:
- "React" → "React를 활용한" (adding natural phrase)
- "경험" → "경험을 갖춘" (natural form)
- "프론트엔드" → "프론트엔드 및 웹" (related keyword)

❌ Forbidden transformation:
- "개발자" → "개발" (noun → verb)
- "디자이너" → "디자인" (person → activity)
- "블록체인" → "Ethereum, Solidity..." (over-expansion)

Rule 4: Natural Conjunctions
- Replace commas with natural Korean conjunctions
- Use: "및", "와", "그리고", "또는", "로"

Examples:
✅ "A 경험, B 능력" → "A 및 B 경험"
✅ "React, TypeScript" → "React 또는 TypeScript"
❌ "A, B, C" (comma-separated list)

--- EXAMPLES BY INDUSTRY ---

IT/Development:
Input: "React나 TypeScript 경험이 있는 프론트엔드 개발자"
Output: "React 또는 TypeScript를 활용한 프론트엔드 개발 경험자"
✓ All keywords: React ✓, TypeScript ✓, 경험 ✓, 프론트엔드 ✓, 개발자→경험자 ✓
✓ Meaning: Still searching for "experienced developer" ✓

Design:
Input: "Figma 능숙한 UI/UX 디자이너"
Output: "Figma를 능숙하게 다루는 UI/UX 디자이너"
✓ All keywords: Figma ✓, 능숙 ✓, UI/UX ✓, 디자이너 ✓
✓ Meaning: Still searching for "proficient designer" ✓

Marketing:
Input: "구글 애널리틱스로 데이터 분석 가능한 마케터"
Output: "Google Analytics를 활용한 데이터 분석 및 성과 측정이 가능한 마케터"
✓ All keywords: GA ✓, 데이터 분석 ✓, 마케터 ✓
✓ Meaning: Still searching for "marketer who can analyze" ✓

Planning:
Input: "서비스 기획 3년 이상"
Output: "서비스 기획 3년 이상 경험자"
✓ All keywords: 서비스 기획 ✓, 3년 이상 ✓
✓ Meaning: Searching for "experienced person" (added for clarity) ✓

Sales:
Input: "B2B 영업, Salesforce"
Output: "B2B 영업 및 Salesforce를 활용한 고객 관리 경험자"
✓ All keywords: B2B ✓, 영업 ✓, Salesforce ✓
✓ Meaning: Searching for "sales person" ✓

Content:
Input: "영상 편집 가능한 크리에이터"
Output: "영상 편집이 가능한 콘텐츠 크리에이터"
✓ All keywords: 영상 편집 ✓, 가능 ✓, 크리에이터 ✓
✓ Meaning: Still searching for "creator" ✓

--- BAD EXAMPLES (Common Mistakes) ---

❌ Mistake 1: Lost Keywords
Input: "React 경험이 있는 개발자"
Bad: "React를 활용한 프론트엔드 개발" (lost "개발자"!)
✅ Correct: "React를 활용한 프론트엔드 개발 경험자"

❌ Mistake 2: Changed Meaning
Input: "Figma 능숙한 디자이너"
Bad: "Figma를 활용한 UI/UX 디자인" (searching for design, not designer!)
✅ Correct: "Figma를 능숙하게 다루는 UI/UX 디자이너"

❌ Mistake 3: Comma Lists
Input: "React와 TypeScript"
Bad: "React 경험, TypeScript 능숙도, 개발 능력" (comma-separated!)
✅ Correct: "React 및 TypeScript를 활용한 개발 경험"

❌ Mistake 4: Over-Expansion
Input: "블록체인"
Bad: "Ethereum, Solidity, Hyperledger, Web3, DeFi" (too specific!)
✅ Correct: "블록체인 관련 기술을 활용한 개발"

❌ Mistake 5: Lost OR Condition
Input: "Python 또는 Java"
Bad: "Python, Java" (lost "또는"!)
✅ Correct: "Python 또는 Java를 활용한 개발"

--- CRITICAL CONSTRAINTS ---

MUST:
✅ Preserve EVERY keyword from original
✅ Keep original meaning 100%
✅ Use natural conjunctions (및, 와, 또는)
✅ Keep 50-100 characters
✅ Be natural and readable

MUST NOT:
❌ Lose any keywords
❌ Change meaning (person → activity)
❌ Use comma-separated lists
❌ Over-expand domain terms
❌ Add unnecessary verbosity

--- TASK ---

Original Query: "{original}"

Output ONLY the rewritten query in Korean. No explanations, no JSON, just the natural Korean query text.
"""
        
        response = await self._llm_client.chat.completions.create(
            model=self._settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at optimizing search queries for portfolio-based talent search. Create natural, search-optimized Korean sentences."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=self._settings.QUERY_REWRITE_TEMPERATURE,
            max_tokens=self._settings.QUERY_REWRITE_MAX_TOKENS
        )
        
        rewritten = response.choices[0].message.content.strip()
        
        if len(rewritten) > 150 or len(rewritten) < 10:
            logger.warning(f"Rewritten query length out of range: {len(rewritten)}")
            return original
        
        return rewritten