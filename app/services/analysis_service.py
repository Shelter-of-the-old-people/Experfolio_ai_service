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
                - Ok(Dict): {"focus": [...], "keywords": [...]}
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
                        "content": "You are an expert query analyst for a talent search engine. Your task is to deconstruct a user's search query into its core components for filtering and query augmentation. You must always respond only in a valid JSON format."
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
            
            logger.info(f"Intent analysis complete: {result.get('focus', 'N/A')}")
            
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
Analyze the following recruitment search query and respond in JSON format based on the rules and examples below.

--- RULES ---

1.  **JSON Format:**
    Respond with a JSON object in the following format:
    {{
      "focus": ["<list of focus areas>"],
      "keywords": ["<list of keywords>"]
    }}

2.  **Focus Categories:**
    Identify the main areas the query focuses on (up to 3) from the following fixed list:
    ["TechnicalSkills", "Experience", "Background"]

    - **TechnicalSkills**: Hard skills, tools, frameworks, languages (e.g., React, Python, AWS, Docker).
    - **Experience**: Career level, domain knowledge, soft skills (e.g., 신입, 3년차, 핀테크, 커머스, 문제해결능력).
    - **Background**: Education, certifications, location (e.g., 학력, 자격증, 서울).

3.  **Keywords Priority:**
    Extract the most important keywords (up to 5) following this priority order:
    1. Programming languages and frameworks (e.g., React, Python, TypeScript)
    2. Technical platforms and tools (e.g., AWS, Docker, Git)
    3. Specific job titles or roles (e.g., Frontend Developer, Data Scientist)
    Focus on concrete, searchable terms.

--- EXAMPLES ---

Query 1: "React와 TypeScript 가능한 신입 프론트엔드 개발자"
Output 1:
{{
  "focus": ["TechnicalSkills", "Experience"],
  "keywords": ["React", "TypeScript", "프론트엔드 개발자", "신입"]
}}

Query 2: "서울에서 근무 가능한 3년차 이상 백엔드 엔지니어"
Output 2:
{{
  "focus": ["Background", "Experience", "TechnicalSkills"],
  "keywords": ["백엔드 엔지니어", "3년 이상", "서울"]
}}

Query 3: "핀테크 도메인 경험 있는 파이썬 개발자, AWS 자격증 우대"
Output 3:
{{
  "focus": ["Experience", "TechnicalSkills", "Background"],
  "keywords": ["Python", "핀테크", "AWS", "자격증"]
}}

--- TASK ---

Query: "{query}"

You must only output a valid JSON.
"""
    
    def _create_match_prompt(self, query: str, portfolio_text: str) -> str:
        """후보자 매칭 분석용 프롬프트를 생성합니다."""
        if len(portfolio_text) > 4000:
            portfolio_text = portfolio_text[:4000] + "..."
        
        return f"""
Follow these steps in your reasoning process before generating the final JSON:

1.  **Deconstruct Query:**
    First, analyze the Search Query to identify "Essential Requirements" (must-haves) and "Preferred Requirements" (nice-to-haves).
    - Essential: Core skills, non-negotiable experience (e.g., 'React', '3년차').
    - Preferred: '우대', '있으면 좋음' (e.g., 'AWS 자격증 우대').

2.  **Scan for Evidence:**
    Meticulously scan the Candidate Portfolio for explicit evidence related to BOTH essential and preferred requirements.

3.  **Evaluate Evidence against NEW Scoring Rubric:**
    Apply the following quantitative rubric based on your findings:
    - **0.8 - 1.0 (Strong Match):** ALL Essential Requirements are met AND one or more Preferred Requirements are met. (Base score 0.9)
    - **0.5 - 0.7 (Partial Match):** ALL Essential Requirements are met, but NO Preferred Requirements are met. (Base score 0.7)
    - **0.1 - 0.4 (Weak Match):** One or more Essential Requirements are NOT met, but there are some related skills.
    - **0.0 (No Match):** No meaningful evidence found for any essential requirements.

4.  **Synthesize Reason:**
    Formulate a concise `matchReason` in **Korean**. This reason MUST explicitly state which essential/preferred requirements were met or not met, justifying the score.

5.  **Extract Keywords:**
    Identify and extract up to 5 of the most relevant technical skills or project names from the portfolio as keywords, in **Korean**.

--- EXAMPLES (Based on NEW Scoring Rubric) ---

**Example 1 (Partial Match - 0.7 Score)**
* Search Query: "React 3년차 개발자, AWS 자격증 우대"
* Portfolio Summary: "...React를 메인 스킬로 3년간 4개의 프로젝트를 리딩함. (AWS 관련 언급 없음)..."
* Ideal Output:
    {{
      "matchScore": 0.7,
      "matchReason": "React 3년 경력이라는 필수 요건은 명확히 충족합니다. 하지만 우대 사항인 AWS 자격증 관련 내용은 포트폴리오에서 확인되지 않았습니다.",
      "keywords": ["React", "3년 경력", "프로젝트 리딩"]
    }}

**Example 2 (Strong Match - 0.9 Score)**
* Search Query: "React 3년차 개발자, AWS 자격증 우대"
* Portfolio Summary: "...React로 3년간 4개의 프로젝트를 리딩함... AWS SAA 자격증 보유 (2023년 취득)..."
* Ideal Output:
    {{
      "matchScore": 0.9,
      "matchReason": "React 3년 경력의 필수 요건을 완벽히 충족하며, 우대 사항인 AWS 자격증(SAA) 또한 보유하고 있어 매우 적합합니다.",
      "keywords": ["React", "3년 경력", "AWS SAA"]
    }}

--- TASK ---

**Search Query:**
"{query}"

**Candidate Portfolio:**
{portfolio_text}

**--- CONSTRAINTS & OUTPUT FORMAT ---**
- Your FINAL output MUST be a single, valid JSON object and nothing else.
- The `matchReason` and `keywords` MUST be in Korean.
- Do NOT hallucinate. Base your analysis ONLY on the evidence in the portfolio.
- Be objective and strictly follow the NEW Scoring Rubric.

**JSON OUTPUT STRUCTURE:**
{{
  "matchScore": <A float between 0.0 and 1.0 based on the rubric>,
  "matchReason": "<Your concise, evidence-based reasoning in Korean>",
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