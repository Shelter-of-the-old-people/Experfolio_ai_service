"""
Analysis Service using OpenAI GPT-4.
OpenAI GPT-4를 사용한 분석 서비스.
"""
from typing import Dict
import json
from openai import OpenAI, OpenAIError
from openai import RateLimitError as OpenAIRateLimitError
from openai import AuthenticationError as OpenAIAuthenticationError
from app.core.config import settings
from app.core.logging import get_logger
from app.core.result import (
    Result, Ok, Err,
    RateLimitError, AuthenticationError, NetworkError, InvalidDataError
)

logger = get_logger(__name__)


class AnalysisService:
    """
    GPT-4를 사용한 검색 의도 분석 및 후보자 매칭 분석 서비스
    """
    
    def __init__(
        self, 
        api_key: str = None,
        model_name: str = None,
        temperature: float = None
    ):
        """
        AnalysisService 초기화
        
        Args:
            api_key: OpenAI API 키 (기본값: settings에서 로드)
            model_name: 사용할 모델 이름 (기본값: settings에서 로드)
            temperature: 생성 온도 (기본값: settings에서 로드)
        """
        self._api_key = api_key or settings.OPENAI_API_KEY
        self._model_name = model_name or settings.OPENAI_MODEL
        self._temperature = temperature or settings.OPENAI_TEMPERATURE
        
        self._llm_client = OpenAI(api_key=self._api_key)
        
        logger.info(f"AnalysisService initialized with model: {self._model_name}")
    
    def analyze_search_intent(self, query: str) -> Result:
        """
        검색 쿼리의 의도를 분석합니다.
        
        Args:
            query: 검색 쿼리
        
        Returns:
            Result:
                - Ok(Dict): {"complexity": ..., "focus": [...], "keywords": [...]}
                - Err: 에러 정보
        """
        try:
            logger.info(f"Analyzing search intent for query: {query[:50]}...")
            
            prompt = self._create_intent_prompt(query)
            
            response = self._llm_client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 채용 검색 쿼리를 분석하는 전문가입니다. 항상 JSON 형식으로만 응답하세요."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self._temperature,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            result = self._parse_json_response(result_text)
            
            logger.info(f"Intent analysis complete: {result.get('complexity', 'unknown')}")
            
            return Ok(result)
            
        except OpenAIRateLimitError as e:
            logger.warning(f"Intent analysis hit rate limit: {str(e)}")
            return Err(RateLimitError(
                error=e,
                context={"query": query[:50], "model": self._model_name}
            ))
            
        except OpenAIAuthenticationError as e:
            logger.error(f"Intent analysis authentication failed: {str(e)}")
            return Err(AuthenticationError(
                error=e,
                context={"api_key_prefix": self._api_key[:10] + "..."}
            ))
            
        except ValueError as e:
            logger.error(f"Intent analysis JSON parsing failed: {str(e)}")
            return Err(InvalidDataError(
                error=e,
                context={"query": query[:50]}
            ))
            
        except OpenAIError as e:
            logger.error(f"Intent analysis OpenAI error: {str(e)}")
            return Err(NetworkError(
                error=e,
                context={"query": query[:50]}
            ))
            
        except Exception as e:
            logger.error(f"Intent analysis unexpected error: {str(e)}")
            return Err(NetworkError(
                error=e,
                context={"query": query[:50]}
            ))
    
    def analyze_candidate_match(
        self, 
        query: str, 
        portfolio_text: str
    ) -> Result:
        """
        후보자와 검색 쿼리의 매칭도를 분석합니다.
        
        Args:
            query: 검색 쿼리
            portfolio_text: 포트폴리오 텍스트
        
        Returns:
            Result:
                - Ok(Dict): {"matchScore": ..., "matchReason": ..., "keywords": [...]}
                - Err: 에러 정보
        """
        try:
            logger.debug(f"Analyzing candidate match for query: {query[:50]}...")
            
            prompt = self._create_match_prompt(query, portfolio_text)
            
            response = self._llm_client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a highly experienced senior tech recruiter. Your task is to provide a critical and evidence-based analysis and output it in a structured JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self._temperature,
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content.strip()
            result = self._parse_json_response(result_text)
            
            match_score = result.get('matchScore', -1)
            if not (0.0 <= match_score <= 1.0):
                return Err(InvalidDataError(
                    error=ValueError(f"Invalid matchScore: {match_score}"),
                    context={"query": query[:50], "matchScore": match_score}
                ))
            
            logger.debug(f"Match analysis complete: score={match_score}")
            
            return Ok(result)
            
        except OpenAIRateLimitError as e:
            logger.warning(f"Match analysis hit rate limit: {str(e)}")
            return Err(RateLimitError(
                error=e,
                context={"query": query[:50], "model": self._model_name}
            ))
            
        except OpenAIAuthenticationError as e:
            logger.error(f"Match analysis authentication failed: {str(e)}")
            return Err(AuthenticationError(
                error=e,
                context={"api_key_prefix": self._api_key[:10] + "..."}
            ))
            
        except ValueError as e:
            logger.error(f"Match analysis validation failed: {str(e)}")
            return Err(InvalidDataError(
                error=e,
                context={"query": query[:50]}
            ))
            
        except OpenAIError as e:
            logger.error(f"Match analysis OpenAI error: {str(e)}")
            return Err(NetworkError(
                error=e,
                context={"query": query[:50]}
            ))
            
        except Exception as e:
            logger.error(f"Match analysis unexpected error: {str(e)}")
            return Err(NetworkError(
                error=e,
                context={"query": query[:50]}
            ))
    
    def _create_intent_prompt(self, query: str) -> str:
        """검색 의도 분석용 프롬프트를 생성합니다."""
        return f"""
Analyze the following recruitment search query and respond in JSON format.

Query: "{query}"

Respond with a JSON object in the following format:
{{
  "complexity": "simple" or "complex",
  "focus": ["Skills", "Experience", "Education", "Projects", etc.],
  "keywords": ["extracted", "main", "keywords"]
}}

- complexity: Determine if the query is simple (single condition) or complex (multiple conditions).
- focus: Identify the main areas the query focuses on (up to 3).
- keywords: Extract the most important keywords for the search (up to 5).

You must only output a valid JSON. Do not include any other text.
"""
    
    def _create_match_prompt(self, query: str, portfolio_text: str) -> str:
        """후보자 매칭 분석용 프롬프트를 생성합니다."""
        if len(portfolio_text) > 4000:
            portfolio_text = portfolio_text[:4000] + "..."
        
        return f"""
Follow these steps in your reasoning process before generating the final JSON:
1.  **Deconstruct the Query:** Break down the search query into essential requirements (e.g., specific technologies, years of experience, soft skills like 'problem-solving').
2.  **Scan for Evidence:** Meticulously scan the candidate's portfolio for explicit keywords and implicit evidence related to each requirement.
3.  **Evaluate Evidence against Scoring Rubric:** For each piece of evidence, evaluate its strength and relevance using the detailed rubric below.
4.  **Synthesize Reason:** Formulate a concise `matchReason` in **Korean**. This reason MUST be directly supported by the evidence found. If there is no evidence, state that clearly.
5.  **Extract Keywords:** Identify and extract up to 5 of the most relevant technical skills or project names from the portfolio as keywords, in **Korean**.
6.  **Finalize JSON:** Construct the final JSON object based on your analysis.

**--- SCORING RUBRIC ---**
- **1.0 (Perfect Match):** All essential requirements of the query are explicitly and strongly met in the portfolio.
- **0.7-0.9 (Strong Match):** Most essential requirements are met. Some minor requirements might be inferred rather than explicit.
- **0.4-0.6 (Partial Match):** Some key requirements are met, but there are significant gaps. The candidate is promising but not a direct fit.
- **0.1-0.3 (Weak Match):** Only tangential or peripheral connections to the query. The candidate has some related skills but misses the core requirements.
- **0.0 (No Match):** No meaningful evidence found that relates to the core requirements of the query.

**--- INPUT DATA ---**
**Search Query:**
"{query}"

**Candidate Portfolio:**
{portfolio_text}

**--- CONSTRAINTS & OUTPUT FORMAT ---**
- Your FINAL output MUST be a single, valid JSON object and nothing else.
- The `matchReason` and `keywords` MUST be in Korean.
- Do NOT hallucinate or invent skills. If a skill is not in the portfolio, it does not exist.
- Be objective and critical. Overly optimistic scores are not helpful.

**JSON OUTPUT STRUCTURE:**
{{
  "matchScore": <A float between 0.0 and 1.0 based on the rubric>,
  "matchReason": "<Your concise, evidence-based reasoning in Korean (2-3 sentences)>",
  "keywords": ["<Up to 5 extracted keywords in Korean>"]
}}

Now, perform the analysis and provide ONLY the final JSON output.
"""
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """
        LLM 응답을 JSON으로 파싱합니다.
        
        Args:
            response_text: LLM 응답 텍스트
        
        Returns:
            Dict: 파싱된 JSON
        
        Raises:
            ValueError: JSON 파싱 실패 시
        """
        try:
            # Markdown 코드 블록 제거 (` ```json ` 또는 ` ``` `)
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            result = json.loads(response_text.strip())
            
            return result
            
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text that failed parsing: {response_text[:500]}")
            raise ValueError(f"Failed to parse JSON from LLM response: {e}")