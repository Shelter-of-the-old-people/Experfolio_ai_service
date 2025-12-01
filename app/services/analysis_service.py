"""
Analysis Service using OpenAI GPT-4.
OpenAI GPT-4를 사용한 분석 서비스.
"""
from typing import Dict
import json
from openai import OpenAI, AsyncOpenAI, OpenAIError
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
        self._model_name = model_name or settings.LLM_ANALYSIS_MODEL
        self._temperature = temperature or settings.OPENAI_TEMPERATURE

        self._llm_client = OpenAI(api_key=self._api_key)
        self._async_llm_client = AsyncOpenAI(api_key=self._api_key)

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
                        "content": "You are an expert query analyst for a portfolio-based talent search engine supporting ALL industries and job types (IT, Design, Marketing, Planning, Sales, etc.). Your task is to deconstruct a user's search query into its core components for filtering and query augmentation. You must always respond only in a valid JSON format."
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
                        "content": "You are a highly experienced senior recruiter specializing in portfolio-based hiring across ALL industries (IT, Design, Marketing, Planning, Sales, Content Creation, etc.). Your task is to provide critical, evidence-based analysis comparing a search query to a candidate's portfolio, and output the result in a structured JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self._temperature,
                max_tokens=1000
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

    async def analyze_candidate_match_async(
        self,
        query: str,
        portfolio_text: str
    ) -> Result:
        """
        후보자와 검색 쿼리의 매칭도를 비동기로 분석합니다.

        Args:
            query: 검색 쿼리
            portfolio_text: 포트폴리오 텍스트

        Returns:
            Result:
                - Ok(Dict): {"matchScore": ..., "matchReason": ..., "keywords": [...]}
                - Err: 에러 정보
        """
        try:
            logger.debug(f"Analyzing candidate match (async) for query: {query[:50]}...")

            prompt = self._create_match_prompt(query, portfolio_text)

            response = await self._async_llm_client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a highly experienced senior recruiter specializing in portfolio-based hiring across ALL industries (IT, Design, Marketing, Planning, Sales, Content Creation, etc.). Your task is to provide critical, evidence-based analysis comparing a search query to a candidate's portfolio, and output the result in a structured JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self._temperature,
                max_tokens=1000
            )

            result_text = response.choices[0].message.content.strip()
            result = self._parse_json_response(result_text)

            match_score = result.get('matchScore', -1)
            if not (0.0 <= match_score <= 1.0):
                return Err(InvalidDataError(
                    error=ValueError(f"Invalid matchScore: {match_score}"),
                    context={"query": query[:50], "matchScore": match_score}
                ))

            logger.debug(f"Match analysis (async) complete: score={match_score}")

            return Ok(result)

        except OpenAIRateLimitError as e:
            logger.warning(f"Match analysis (async) hit rate limit: {str(e)}")
            return Err(RateLimitError(
                error=e,
                context={"query": query[:50], "model": self._model_name}
            ))

        except OpenAIAuthenticationError as e:
            logger.error(f"Match analysis (async) authentication failed: {str(e)}")
            return Err(AuthenticationError(
                error=e,
                context={"api_key_prefix": self._api_key[:10] + "..."}
            ))

        except ValueError as e:
            logger.error(f"Match analysis (async) validation failed: {str(e)}")
            return Err(InvalidDataError(
                error=e,
                context={"query": query[:50]}
            ))

        except OpenAIError as e:
            logger.error(f"Match analysis (async) OpenAI error: {str(e)}")
            return Err(NetworkError(
                error=e,
                context={"query": query[:50]}
            ))

        except Exception as e:
            logger.error(f"Match analysis (async) unexpected error: {str(e)}")
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

    - **TechnicalSkills**: Hard skills, tools, software, platforms across ALL industries.
      * IT/Development: React, Python, AWS, Docker, Git, TypeScript, Node.js
      * Design: Figma, Sketch, Adobe XD, Photoshop, Illustrator, After Effects
      * Marketing: Google Analytics, Meta Ads Manager, SEO, GTM, Tableau, Mixpanel
      * Planning: Notion, Jira, Confluence, Figma(prototyping), Excel, PowerPoint
      * Sales: Salesforce, HubSpot, CRM tools, Excel, Presentation skills
      * Content: Premiere Pro, Final Cut, Canva, WordPress, YouTube Studio
    
    - **Experience**: Career level, industry domain, work experience, soft skills.
      * Examples: 신입, 3년차, 시니어, 핀테크, 커머스, B2B, 스타트업, 리더십, 협업능력, 문제해결능력
    
    - **Background**: Education, certifications, location, languages.
      * Examples: 학력, 학사, 석사, 자격증, AWS 인증, 디자인 수상, 서울, 경기, 영어, 일본어

3.  **Keywords Priority:**
    Extract the most important keywords (up to 5) following this priority order:
    1. Core tools/skills relevant to the job (e.g., Figma, Python, Google Analytics)
    2. Specific job titles or roles (e.g., UI/UX 디자이너, 퍼포먼스 마케터, 백엔드 개발자)
    3. Domain or industry focus (e.g., 핀테크, 이커머스, B2B SaaS)
    4. Experience level or qualifications (e.g., 3년차, 신입, 자격증)
    Focus on concrete, searchable terms that represent the query's core intent.

--- EXAMPLES ---

Query 1: "React와 TypeScript 가능한 신입 프론트엔드 개발자"
Output 1:
{{
  "focus": ["TechnicalSkills", "Experience"],
  "keywords": ["React", "TypeScript", "프론트엔드 개발자", "신입"]
}}

Query 2: "Figma와 Sketch를 다루는 3년차 이상 UI/UX 디자이너"
Output 2:
{{
  "focus": ["TechnicalSkills", "Experience"],
  "keywords": ["Figma", "Sketch", "UI/UX 디자이너", "3년차"]
}}

Query 3: "구글 애널리틱스로 성과 분석 가능한 퍼포먼스 마케터"
Output 3:
{{
  "focus": ["TechnicalSkills"],
  "keywords": ["Google Analytics", "성과 분석", "퍼포먼스 마케터"]
}}

Query 4: "서비스 기획 경험 3년 이상, 사용자 리서치 역량 보유"
Output 4:
{{
  "focus": ["Experience"],
  "keywords": ["서비스 기획", "3년 이상", "사용자 리서치"]
}}

Query 5: "B2B 영업 경험자, Salesforce 활용 가능"
Output 5:
{{
  "focus": ["Experience", "TechnicalSkills"],
  "keywords": ["B2B 영업", "Salesforce", "영업 경험"]
}}

Query 6: "영상 편집 가능한 콘텐츠 크리에이터, Premiere Pro 사용"
Output 6:
{{
  "focus": ["TechnicalSkills"],
  "keywords": ["영상 편집", "Premiere Pro", "콘텐츠 크리에이터"]
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
    Analyze the Search Query to identify "Essential Requirements" (must-haves) and "Preferred Requirements" (nice-to-haves).

2.  **Scan for Evidence:**
    Meticulously scan the Candidate Portfolio for explicit evidence related to BOTH essential and preferred requirements. Look for specific projects, skills mentioned, or experiences described.

3.  **Evaluate Evidence against Scoring Rubric:**
    Apply the following quantitative rubric based on your findings:
    
    - **0.8 - 1.0 (Strong Match):** ALL Essential Requirements are clearly met with strong evidence AND one or more Preferred Requirements are met. (Base score 0.9)
    
    - **0.5 - 0.7 (Partial Match):** ALL Essential Requirements are met, but NO Preferred Requirements are met, OR evidence for essential requirements is present but weak/implicit. (Base score 0.7)
    
    - **0.1 - 0.4 (Weak Match):** One or more Essential Requirements are NOT met, but there are some related skills, potential, or partial fulfillment.
    
    - **0.0 (No Match):** No meaningful evidence found for any essential requirements.

4.  **Adapt Evaluation to Job Type:**
    Consider the nature of the role when evaluating the portfolio:
    
    **For Technical/IT Roles:**
    - Focus on: Code quality, technical depth, project complexity, framework/tool proficiency, system design understanding
    - Evidence: GitHub links, technical blog posts, specific technology mentions, problem-solving descriptions
    
    **For Design Roles:**
    - Focus on: Visual quality, design consistency, user-centered thinking, portfolio presentation quality, design tool proficiency
    - Evidence: Design portfolio links, case studies, design process descriptions, visual examples, user research mentions
    
    **For Marketing Roles:**
    - Focus on: Campaign results (ROAS, CTR, conversion rates), data analysis ability, marketing tool proficiency, strategic thinking
    - Evidence: Specific metrics/KPIs, campaign case studies, A/B testing experience, growth achievements
    
    **For Planning Roles:**
    - Focus on: Logic and structure in documents, user research depth, problem-solving approach, documentation quality
    - Evidence: Planning documents, wireframes, user research reports, PRD examples, logical analysis
    
    **For Sales Roles:**
    - Focus on: Sales achievement records, client relationship management, communication skills, negotiation ability
    - Evidence: Sales numbers, client testimonials, deal closing examples, CRM proficiency
    
    **For Content Creation Roles:**
    - Focus on: Content quality, creativity, engagement metrics, platform knowledge, production skills
    - Evidence: Published content links, view counts, engagement rates, creative concepts, production examples

5.  **Synthesize Reason (Analytical Focus):**
    Formulate a concise `matchReason` in **Korean** focusing on *analysis* rather than just evaluation.
    - **Describe Strengths:** Highlight 1-2 key experiences or skills from the portfolio that **directly relate** to the query's requirements. **Quote or reference specific portfolio content** (e.g., project names, specific phrases) as evidence. Explain *how* this evidence demonstrates relevant capabilities.
    - **Identify Gaps:** Clearly state which requirements from the query are **missing or weakly supported** in the portfolio.
    - **Provide Insight (Optional but encouraged):** Briefly mention potential or related strengths visible in the portfolio, even if not directly asked for in the query.

6.  **Extract Keywords:**
    Identify and extract up to 5 of the most relevant skills, tools, or project types mentioned *in the portfolio text*, in **Korean**. Prioritize skills directly related to the query requirements and the strengths you identified.

--- EXAMPLES (Based on Rubric & Analytical Reason Style) ---

**Example 1 (IT Role - Partial Match - 0.7 Score)**
* Search Query: "React 3년차 개발자, AWS 자격증 우대"
* Portfolio Summary: "...React를 메인 스킬로 3년간 4개의 프로젝트를 리딩함. (AWS 관련 언급 없음)..."
* Ideal Output:
    {{
      "matchScore": 0.7,
      "matchReason": "React 3년 경력은 포트폴리오의 'React 메인 스킬 리딩 경험'으로 확인됩니다. 이 경험은 React 기반 개발 역량을 보여주지만, 쿼리에서 우대한 AWS 관련 경험은 언급되지 않았습니다.",
      "keywords": ["React", "3년 경력", "프로젝트 리딩"]
    }}

**Example 2 (Design Role - Strong Match - 0.9 Score)**
* Search Query: "Figma 능숙한 UI/UX 디자이너, 사용자 리서치 경험"
* Portfolio Summary: "...Figma로 5개 프로젝트 디자인... 사용자 인터뷰 20회 진행하여 개선안 도출..."
* Ideal Output:
    {{
      "matchScore": 0.9,
      "matchReason": "Figma 사용 경험은 '5개 프로젝트 디자인' 이력으로, 사용자 리서치 역량은 '사용자 인터뷰 20회 진행' 문구로 확인됩니다. 두 핵심 요건을 모두 충족하며, 특히 리서치 기반 개선안 도출 경험은 사용자 중심 설계 능력을 보여줍니다.",
      "keywords": ["Figma", "UI/UX 디자인", "사용자 리서치", "인터뷰"]
    }}

**Example 3 (Marketing Role - Weak Match - 0.3 Score)**
* Search Query: "구글 애널리틱스로 데이터 분석 가능한 퍼포먼스 마케터"
* Portfolio Summary: "...SNS 콘텐츠 제작 경험... (데이터 분석 도구 언급 없음)..."
* Ideal Output:
    {{
      "matchScore": 0.3,
      "matchReason": "SNS 콘텐츠 제작 경험은 마케팅 관련성이 있으나, 핵심 요건인 구글 애널리틱스 사용이나 데이터 분석 역량에 대한 언급이 포트폴리오에 없습니다. 퍼포먼스 마케팅보다는 콘텐츠 마케팅에 가까운 경험으로 보입니다.",
      "keywords": ["SNS 마케팅", "콘텐츠 제작"]
    }}

--- TASK ---

**Search Query:**
"{query}"

**Candidate Portfolio:**
{portfolio_text}

**--- CONSTRAINTS & OUTPUT FORMAT ---**
- Your FINAL output MUST be a single, valid JSON object and nothing else.
- The `matchReason` (analytical explanation) and `keywords` (from portfolio) MUST be in Korean.
- Do NOT hallucinate. Base your analysis ONLY on the evidence found in the portfolio text provided.
- Strictly follow the Scoring Rubric and adapt your evaluation criteria to the job type.
- Use the analytical `matchReason` style, including evidence citation.

**JSON OUTPUT STRUCTURE:**
{{
  "matchScore": <A float between 0.0 and 1.0 based on the rubric>,
  "matchReason": "<Your concise, analytical reasoning in Korean, citing portfolio evidence>",
  "keywords": ["<Up to 5 extracted keywords from portfolio in Korean>"]
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