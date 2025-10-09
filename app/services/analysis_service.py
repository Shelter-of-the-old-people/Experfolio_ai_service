"""
Analysis Service using OpenAI GPT-4.
OpenAI GPT-4를 사용한 분석 서비스.
"""
from typing import Dict
import json
from openai import OpenAI
from app.core.config import settings
from app.core.logging import get_logger

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
        
        # OpenAI 클라이언트 초기화
        self._llm_client = OpenAI(api_key=self._api_key)
        
        logger.info(f"AnalysisService initialized with model: {self._model_name}")
    
    def analyze_search_intent(self, query: str) -> Dict:
        """
        검색 쿼리의 의도를 분석합니다.
        
        Args:
            query: 검색 쿼리
        
        Returns:
            Dict: 분석 결과
                {
                    "complexity": "simple" | "complex",
                    "focus": ["기술스택", "경력", "학력" 등],
                    "keywords": ["React", "3년" 등]
                }
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
            
            # 응답 파싱
            result_text = response.choices[0].message.content.strip()
            result = self._parse_json_response(result_text)
            
            logger.info(f"Intent analysis complete: {result.get('complexity', 'unknown')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {str(e)}")
            # 실패 시 기본값 반환
            return {
                "complexity": "simple",
                "focus": ["기술스택"],
                "keywords": []
            }
    
    def analyze_candidate_match(
        self, 
        query: str, 
        portfolio_text: str
    ) -> Dict:
        """
        후보자와 검색 쿼리의 매칭도를 분석합니다.
        
        Args:
            query: 검색 쿼리
            portfolio_text: 포트폴리오 텍스트
        
        Returns:
            Dict: 매칭 분석 결과
                {
                    "matchScore": 0.85,
                    "matchReason": "매칭 근거 설명",
                    "keywords": ["React", "TypeScript"]
                }
        """
        try:
            logger.debug(f"Analyzing candidate match for query: {query[:50]}...")
            
            prompt = self._create_match_prompt(query, portfolio_text)
            
            response = self._llm_client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 채용 매칭 전문가입니다. 항상 JSON 형식으로만 응답하세요."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self._temperature,
                max_tokens=800
            )
            
            # 응답 파싱
            result_text = response.choices[0].message.content.strip()
            result = self._parse_json_response(result_text)
            
            logger.debug(f"Match analysis complete: score={result.get('matchScore', 0)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Match analysis failed: {str(e)}")
            # 실패 시 기본값 반환
            return {
                "matchScore": 0.5,
                "matchReason": "분석 실패",
                "keywords": []
            }
    
    def _create_intent_prompt(self, query: str) -> str:
        """
        검색 의도 분석용 프롬프트를 생성합니다.
        
        Args:
            query: 검색 쿼리
        
        Returns:
            str: 프롬프트
        """
        return f"""
다음 채용 검색 쿼리를 분석하세요:

쿼리: "{query}"

다음 형식의 JSON으로 응답하세요:
{{
  "complexity": "simple" 또는 "complex",
  "focus": ["기술스택", "경력", "학력", "프로젝트" 등],
  "keywords": ["추출된", "주요", "키워드"]
}}

- complexity: 쿼리가 단순한지(단일 조건) 복잡한지(다중 조건) 판단
- focus: 쿼리가 집중하는 영역 (최대 3개)
- keywords: 검색에 중요한 키워드 추출 (최대 5개)

반드시 유효한 JSON만 출력하세요. 다른 텍스트는 포함하지 마세요.
"""
    
    def _create_match_prompt(self, query: str, portfolio_text: str) -> str:
        """
        후보자 매칭 분석용 프롬프트를 생성합니다.
        
        Args:
            query: 검색 쿼리
            portfolio_text: 포트폴리오 텍스트
        
        Returns:
            str: 프롬프트
        """
        # 포트폴리오 텍스트가 너무 길면 잘라내기
        if len(portfolio_text) > 3000:
            portfolio_text = portfolio_text[:3000] + "..."
        
        return f"""
채용 검색 쿼리와 후보자 포트폴리오의 매칭도를 분석하세요.

검색 쿼리: "{query}"

후보자 포트폴리오:
{portfolio_text}

다음 형식의 JSON으로 응답하세요:
{{
  "matchScore": 0.0에서 1.0 사이의 숫자,
  "matchReason": "매칭이 적합한 이유를 2-3문장으로 설명",
  "keywords": ["포트폴리오에서", "발견된", "주요", "키워드"]
}}

평가 기준:
1. 쿼리의 요구사항이 포트폴리오에 얼마나 반영되어 있는가
2. 관련 기술스택, 경험, 프로젝트의 구체성
3. 전반적인 적합도

반드시 유효한 JSON만 출력하세요. 다른 텍스트는 포함하지 마세요.
matchScore는 0.0~1.0 사이의 소수점 숫자여야 합니다.
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
            # markdown 코드 블록 제거
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            # JSON 파싱
            result = json.loads(response_text.strip())
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            logger.error(f"Response text: {response_text}")
            raise ValueError(f"Failed to parse JSON response: {str(e)}")
