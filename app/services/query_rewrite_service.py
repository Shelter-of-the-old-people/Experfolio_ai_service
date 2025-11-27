"""
Query Rewrite Service using GPT-4o.
GPT-4o를 사용한 쿼리 재작성 서비스.
"""
from typing import Dict, Optional
from openai import AsyncOpenAI
from app.core.config import settings
from app.core.logging import get_logger
from app.core.result import Result, Ok, Err, RateLimitError, InvalidDataError
from app.schemas.response import RewrittenQueryInfo

logger = get_logger(__name__)


class QueryRewriteService:
    """
    검색 쿼리를 재작성하여 검색 품질을 향상시키는 서비스
    GPT-4o를 사용하여 자연어 쿼리를 검색 최적화된 형태로 변환합니다.
    """
    
    PROMPT_V2_1_DOMAIN_EXPANSION = """
You are an expert at optimizing Korean search queries for portfolio-based talent search.

Your task: Rewrite the query to be more search-friendly while preserving ALL keywords and original meaning.

--- REASONING PROCESS (Follow these steps internally) ---

Step 1: Identify ALL Keywords
- Extract every meaningful word: technologies, skills, job titles, experience levels, adjectives
- Example: "React나 TypeScript 경험이 있는 프론트엔드 개발자"
  → Keywords: React, TypeScript, 경험, 프론트엔드, 개발자

Step 2: Understand Core Meaning
- What is being searched for? (person, skill, experience)
- Example: "경험이 있는 개발자" = searching for a developer WITH experience

Step 3: Check for Domain Expansion
- Is this a specialized domain? (항공우주, 블록체인, AI, 바이오, FinTech...)
- If YES: Apply domain expansion pattern
- If NO: Apply basic expansion only

Step 4: Expand Naturally
- Keep ALL keywords
- Replace commas with natural conjunctions (및, 와, 그리고, 또는)
- If domain: add 3-5 related keywords with pattern
- Keep it natural and readable

Step 5: Validate
✅ All keywords preserved?
✅ Original meaning unchanged?
✅ Natural conjunctions instead of commas?
✅ If domain: correct pattern used?
✅ 50-150 characters?

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

Rule 3: Natural Expansion with Domain Intelligence (UPDATED!)

A. 기본 확장 (모든 쿼리):
- Add natural Korean phrases
- "React" → "React를 활용한"
- "경험" → "경험을 갖춘"

B. 도메인 확장 (전문 분야만):
Pattern: [확장키워드1], [확장키워드2], [확장키워드3] 등 [원본도메인] 관련 [기술/분야/경험] [직무]

확장 대상 (Expand these domains):
✅ 전문 산업: 항공우주, 바이오, 양자컴퓨팅, 반도체, 로봇공학
✅ 신기술: 블록체인, Web3, AI, 머신러닝, FinTech, 메타버스
✅ 특수 플랫폼: 클라우드 (AWS, GCP 등)

확장 제외 (Don't expand):
❌ 일반 기술: React, Python, Java (already specific)
❌ 직무: 개발자, 디자이너 (clear enough)
❌ 경험: 3년 이상 (numerical)

확장 규칙:
1. 확장 키워드 3-5개 선택 (업계 표준 용어)
2. 쉼표로 구분
3. "등" 추가 (MANDATORY)
4. 원본 도메인 다시 명시 (MANDATORY)
5. "관련 기술/분야/경험" 연결
6. 원본의 직무/대상 유지

좋은 예:
✅ "Ethereum, Solidity, Web3, 스마트 컨트랙트 등 블록체인 관련 기술 개발자"
✅ "머신러닝, 딥러닝, TensorFlow, PyTorch 등 AI 관련 기술 엔지니어"
✅ "인공위성, 우주 산업, 항공기 시스템 등 항공우주 관련 기술 경험자"

나쁜 예:
❌ "블록체인, Ethereum, Solidity 개발자" (원본 도메인 위치 잘못)
❌ "Ethereum, Solidity, Web3" (등, 관련 표현 없음)
❌ "블록체인 개발자" (도메인 확장 안 함)

Rule 4: Natural Conjunctions
- Replace commas with natural Korean conjunctions
- Use: "및", "와", "그리고", "또는", "로"

Examples:
✅ "A 경험, B 능력" → "A 및 B 경험"
✅ "React, TypeScript" → "React 또는 TypeScript"
❌ "A, B, C" (comma-separated list for non-domain terms)

--- DOMAIN EXPANSION EXAMPLES ---

Pattern: [확장키워드1], [확장키워드2], [확장키워드3] 등 [원본도메인] 관련 [기술/분야/경험] [직무]

Example 1: 항공우주
Input: "항공우주 관련 경험이 있는 인재"
Output: "인공위성, 우주 산업, 항공기 시스템 등 항공우주 관련 기술 경험자"
Pattern: [인공위성, 우주 산업, 항공기 시스템] + 등 + [항공우주] + 관련 기술 + 경험자
Why: 항공우주는 전문 도메인 → 관련 키워드 확장 + 원본 재명시

Example 2: 블록체인
Input: "블록체인 개발자"
Output: "Ethereum, Solidity, Web3, 스마트 컨트랙트 등 블록체인 관련 기술 개발자"
Pattern: [Ethereum, Solidity, Web3, 스마트 컨트랙트] + 등 + [블록체인] + 관련 기술 + 개발자
Why: 블록체인 → 플랫폼 및 기술로 확장 + 원본 재명시

Example 3: AI/머신러닝
Input: "AI 엔지니어 구합니다"
Output: "머신러닝, 딥러닝, TensorFlow, PyTorch 등 AI 관련 기술 엔지니어"
Pattern: [머신러닝, 딥러닝, TensorFlow, PyTorch] + 등 + [AI] + 관련 기술 + 엔지니어
Why: AI → 관련 분야와 프레임워크로 확장 + 원본 재명시

Example 4: FinTech
Input: "FinTech 스타트업 경험"
Output: "핀테크, 결제 시스템, 디지털 금융, 금융 API 등 FinTech 관련 분야 경험자"
Pattern: [핀테크, 결제 시스템, 디지털 금융, 금융 API] + 등 + [FinTech] + 관련 분야 + 경험자
Why: FinTech → 한글 표기 및 관련 분야로 확장 + 원본 재명시

Example 5: 바이오/헬스케어
Input: "바이오 연구 개발자"
Output: "생명공학, 유전자, 신약 개발, 의료 기기 등 바이오 관련 연구 개발자"
Pattern: [생명공학, 유전자, 신약 개발, 의료 기기] + 등 + [바이오] + 관련 연구 + 개발자
Why: 바이오 → 구체적 분야로 확장 + 원본 재명시

Example 6: 클라우드
Input: "클라우드 인프라 담당"
Output: "AWS, GCP, Azure, Kubernetes, Docker 등 클라우드 관련 인프라 담당자"
Pattern: [AWS, GCP, Azure, Kubernetes, Docker] + 등 + [클라우드] + 관련 인프라 + 담당자
Why: 클라우드 → 주요 플랫폼 및 도구로 확장 + 원본 재명시

Example 7: 일반 기술 (No domain expansion)
Input: "React와 TypeScript 개발자"
Output: "React 및 TypeScript를 활용한 개발자"
Pattern: Basic expansion only (natural conjunctions)
Why: React, TypeScript는 이미 구체적 → 도메인 확장 불필요, 기본 확장만

--- BAD EXAMPLES (Common Mistakes) ---

❌ Mistake 1: Lost Keywords
Input: "React 경험이 있는 개발자"
Bad: "React를 활용한 프론트엔드 개발" (lost "개발자"!)
✅ Correct: "React를 활용한 프론트엔드 개발 경험자"

❌ Mistake 2: Changed Meaning
Input: "Figma 능숙한 디자이너"
Bad: "Figma를 활용한 UI/UX 디자인" (searching for design, not designer!)
✅ Correct: "Figma를 능숙하게 다루는 UI/UX 디자이너"

❌ Mistake 3: Domain Pattern Wrong
Input: "항공우주 관련 경험"
Bad: "항공우주, 인공위성, 우주 산업 관련 경험" (no "등", no 재명시!)
✅ Correct: "인공위성, 우주 산업, 항공기 시스템 등 항공우주 관련 기술 경험자"

❌ Mistake 4: Over-Expansion
Input: "블록체인"
Bad: "Ethereum, Solidity, Hyperledger, Web3, DeFi, NFT, DAO 등..." (7개+, too many!)
✅ Correct: "Ethereum, Solidity, Web3, 스마트 컨트랙트 등 블록체인 관련 기술" (4개, good!)

❌ Mistake 5: Lost "등" or Original Domain
Input: "AI 엔지니어"
Bad: "머신러닝, 딥러닝, TensorFlow AI 엔지니어" (no "등", wrong position!)
✅ Correct: "머신러닝, 딥러닝, TensorFlow, PyTorch 등 AI 관련 기술 엔지니어"

--- CRITICAL CONSTRAINTS ---

MUST:
✅ Preserve EVERY keyword from original
✅ Keep original meaning 100%
✅ Use natural conjunctions (및, 와, 또는)
✅ For domains: Use pattern "[확장들] 등 [원본] 관련 [표현] [직무]"
✅ "등" is MANDATORY for domain expansion
✅ Original domain MUST be mentioned again after "등"
✅ Keep 50-150 characters
✅ Be natural and readable

MUST NOT:
❌ Lose any keywords
❌ Change meaning (person → activity)
❌ Use comma-separated lists (except domain expansion)
❌ Over-expand (7+ keywords)
❌ Skip "등" in domain expansion
❌ Skip original domain re-mention
❌ Add unnecessary verbosity

--- TASK ---

Original Query: "{original}"

Output ONLY the rewritten query in Korean. No explanations, no JSON, just the natural Korean query text.
"""
    
    def __init__(self):
        """
        QueryRewriteService 초기화
        """
        self._llm_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._settings = settings
        
        logger.info("QueryRewriteService initialized with GPT-4o")
    
    async def rewrite_query(
        self,
        original: str,
        intent: Dict,
        enable: bool = True
    ) -> Result:
        """
        검색 쿼리를 재작성합니다.
        
        Args:
            original: 원본 쿼리
            intent: 검색 의도 분석 결과 (사용 안 함, 호환성 유지용)
            enable: 재작성 활성화 여부
        
        Returns:
            Result[RewrittenQueryInfo]: 재작성된 쿼리 정보
        """
        if not enable:
            logger.debug("Query rewrite disabled")
            return Ok(RewrittenQueryInfo(
                original=original,
                rewritten=original,
                strategy="skip",
                confidence=1.0,
                cached=False
            ))
        
        try:
            logger.info(f"Rewriting query: '{original[:50]}...'")
            
            rewritten = await self._natural_generation(original)
            
            if rewritten == original:
                logger.warning(f"Rewrite returned original query, using skip strategy")
                return Ok(RewrittenQueryInfo(
                    original=original,
                    rewritten=original,
                    strategy="skip",
                    confidence=0.5,
                    cached=False
                ))
            
            logger.info(
                f"Query rewrite successful: '{original[:30]}...' → '{rewritten[:30]}...'"
            )
            
            return Ok(RewrittenQueryInfo(
                original=original,
                rewritten=rewritten,
                strategy="natural_generation",
                confidence=0.85,
                cached=False
            ))
            
        except Exception as e:
            logger.error(f"Query rewrite failed: {str(e)}", exc_info=True)
            return Ok(RewrittenQueryInfo(
                original=original,
                rewritten=original,
                strategy="failed",
                confidence=0.0,
                cached=False
            ))
    
    async def _natural_generation(self, original: str) -> str:
        """
        GPT-4o를 사용한 자연어 생성 방식 재작성 (v2.1 - Domain Expansion)
        """
        prompt = self.PROMPT_V2_1_DOMAIN_EXPANSION.format(original=original)
        
        try:
            response = await self._llm_client.chat.completions.create(
                model=self._settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at optimizing search queries for portfolio-based talent search. Create natural, search-optimized Korean sentences with domain intelligence."
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
            
            if len(rewritten) > 200 or len(rewritten) < 10:
                logger.warning(
                    f"Rewritten query length out of range: {len(rewritten)}, "
                    f"using original"
                )
                return original
            
            return rewritten
            
        except Exception as e:
            logger.error(f"GPT-4o natural generation failed: {str(e)}")
            return original