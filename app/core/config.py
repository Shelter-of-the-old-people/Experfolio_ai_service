"""
Configuration management using Pydantic Settings.
환경 변수를 로드하고 검증하는 설정 관리 모듈.
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    애플리케이션 설정 클래스
    .env 파일에서 자동으로 환경 변수를 로드합니다.
    """
    
    # MongoDB 설정
    MONGODB_URI: str = Field(..., description="MongoDB 연결 URI")
    MONGODB_DATABASE: str = Field(default="experfolio", description="데이터베이스 이름")
    
    # OpenAI 설정
    OPENAI_API_KEY: str = Field(..., description="OpenAI API 키")
    OPENAI_MODEL: str = Field(default="gpt-4o", description="사용할 OpenAI 모델") # 모델 변경
    OPENAI_TEMPERATURE: float = Field(default=0.7, description="생성 온도")
    
    # KURE 모델 설정
    KURE_MODEL_NAME: str = Field(
        default="nlpai-lab/KURE-v1", 
        description="KURE 임베딩 모델 이름"
    )
    
    # Reranker 모델 설정
    RERANKER_MODEL_NAME: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Reranker 모델 이름"
    )
    
    # Docker Volume 설정
    DOCKER_VOLUME_PATH: str = Field(
        default="/data/portfolios",
        description="Docker Volume 마운트 경로"
    )
    
    # 배치 스케줄 설정
    BATCH_SCHEDULE_TIME: str = Field(
        default="02:00",
        description="일일 배치 실행 시간 (HH:MM)"
    )
    
    # 로깅 설정
    LOG_LEVEL: str = Field(default="INFO", description="로그 레벨")
    LOG_FILE: str = Field(default="logs/app.log", description="로그 파일 경로")
    
    # API 설정
    API_HOST: str = Field(default="0.0.0.0", description="API 호스트")
    API_PORT: int = Field(default=8001, description="API 포트")
    API_TITLE: str = Field(default="Experfolio AI Service", description="API 제목")
    API_VERSION: str = Field(default="1.0.0", description="API 버전")
    
    # 벡터 검색 설정
    VECTOR_SEARCH_LIMIT: int = Field(default=50, description="벡터 검색 초기 결과 수")
    RERANK_TOP_K: int = Field(default=10, description="재순위 후 최종 결과 수")

    # 검색 필터링 설정
    VECTOR_SEARCH_SCORE_THRESHOLD: float = Field(
        default=0.5, 
        description="1단계 필터: 벡터 검색 결과의 최소 유사도 점수"
    )
    RERANKER_SCORE_THRESHOLD: float = Field(
        default=0.1, 
        description="2단계 필터: 재순위 모델의 최소 관련도 점수"
    )
    LLM_MATCH_SCORE_THRESHOLD: float = Field(
        default=0.0, 
        description="3단계 필터: LLM 평가 점수(matchScore)의 기준점"
    )
    
    # 병렬 처리 설정
    CANDIDATE_ANALYSIS_TIMEOUT: float = Field(
        default=10.0,
        description="개별 후보자 분석 타임아웃 (초)"
    )
    MAX_CONCURRENT_ANALYSIS: int = Field(
        default=10,
        description="최대 동시 분석 수 (미래 확장용)"
    )
    
    # GPU 설정
    USE_GPU: bool = Field(
        default=True,
        description="GPU 사용 여부 (가능한 경우 자동 사용)"
    )
    FORCE_CPU: bool = Field(
        default=False,
        description="강제로 CPU 사용 (디버깅용)"
    )
    
    class Config:
        """Pydantic 설정"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @validator("MONGODB_URI")
    def validate_mongodb_uri(cls, v):
        """MONGODB_URI 필수 값 및 형식 검증"""
        if not v or "your-mongodb-uri-here" in v:
            raise ValueError("MONGODB_URI must be set in .env file.")
        if not v.startswith("mongodb"):
            raise ValueError("MONGODB_URI must be a valid MongoDB connection string.")
        return v

    @validator("OPENAI_API_KEY")
    def validate_openai_api_key(cls, v):
        """OPENAI_API_KEY 필수 값 및 형식 검증"""
        if not v or "your-openai-api-key-here" in v:
            raise ValueError("OPENAI_API_KEY must be set in .env file.")
        if not v.startswith("sk-"):
            raise ValueError("OPENAI_API_KEY must start with 'sk-'.")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """로그 레벨 검증"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @validator("BATCH_SCHEDULE_TIME")
    def validate_schedule_time(cls, v):
        """스케줄 시간 형식 검증 (HH:MM)"""
        try:
            hour, minute = v.split(":")
            hour_int = int(hour)
            minute_int = int(minute)
            if not (0 <= hour_int < 24 and 0 <= minute_int < 60):
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError("BATCH_SCHEDULE_TIME must be in HH:MM format (e.g., 02:00)")
        return v
    
    @validator("LOG_FILE")
    def create_log_directory(cls, v):
        """로그 디렉토리 자동 생성"""
        log_path = Path(v)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return v


# 싱글톤 설정 인스턴스
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    설정 인스턴스를 반환합니다 (싱글톤).
    
    Returns:
        Settings: 설정 객체
    
    Raises:
        ValueError: 설정 로드 실패 시
    """
    global _settings
    
    if _settings is None:
        try:
            _settings = Settings()
        except Exception as e:
            raise ValueError(f"Failed to load or validate settings: {str(e)}")
    
    return _settings


# 편의를 위한 설정 인스턴스 export
settings = get_settings()