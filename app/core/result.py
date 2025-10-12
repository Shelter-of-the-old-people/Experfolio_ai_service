"""
Result pattern for explicit success/failure handling.
명시적 성공/실패 처리를 위한 Result 패턴.
"""
from typing import TypeVar, Generic, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

T = TypeVar('T')


@dataclass(frozen=True)
class Ok(Generic[T]):
    """
    성공 결과를 나타냅니다.
    
    Attributes:
        value: 성공 시 반환할 실제 값
    """
    value: T


class ErrorType(ABC):
    """
    에러 타입 추상 클래스.
    각 구체적 에러는 이를 상속하여 자신의 행동을 정의합니다.
    """
    
    def __init__(self, error: Exception, context: dict = None):
        """
        Args:
            error: 발생한 예외
            context: 추가 컨텍스트 정보 (선택)
        """
        self.error = error
        self.context = context or {}
    
    @abstractmethod
    def is_retryable(self) -> bool:
        """재시도 가능 여부를 반환합니다."""
        pass
    
    @abstractmethod
    def retry_delay(self) -> float:
        """재시도 대기 시간(초)을 반환합니다."""
        pass
    
    @property
    def error_message(self) -> str:
        """에러 메시지를 반환합니다."""
        return str(self.error)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.error_message}"


class NetworkError(ErrorType):
    """
    네트워크 일시 오류 (MongoDB, API 호출 등).
    재시도 가능하며, 1초 후 재시도를 권장합니다.
    """
    
    def is_retryable(self) -> bool:
        return True
    
    def retry_delay(self) -> float:
        return 1.0


class RateLimitError(ErrorType):
    """
    API Rate Limit 에러.
    재시도 가능하며, 60초 후 재시도를 권장합니다.
    """
    
    def is_retryable(self) -> bool:
        return True
    
    def retry_delay(self) -> float:
        return 60.0


class TimeoutError(ErrorType):
    """
    타임아웃 에러.
    재시도 가능하며, 5초 후 재시도를 권장합니다.
    """
    
    def is_retryable(self) -> bool:
        return True
    
    def retry_delay(self) -> float:
        return 5.0


class InvalidDataError(ErrorType):
    """
    잘못된 데이터 에러 (빈 텍스트, 범위 초과 등).
    재시도 불가능합니다.
    """
    
    def is_retryable(self) -> bool:
        return False
    
    def retry_delay(self) -> float:
        return 0.0


class AuthenticationError(ErrorType):
    """
    인증 실패 에러 (API 키 문제 등).
    재시도 불가능합니다.
    """
    
    def is_retryable(self) -> bool:
        return False
    
    def retry_delay(self) -> float:
        return 0.0


class ConfigurationError(ErrorType):
    """
    설정 오류 에러.
    재시도 불가능합니다.
    """
    
    def is_retryable(self) -> bool:
        return False
    
    def retry_delay(self) -> float:
        return 0.0


class SystemError(ErrorType):
    """
    시스템 리소스 문제 (메모리 부족 등).
    재시도 불가능합니다.
    """
    
    def is_retryable(self) -> bool:
        return False
    
    def retry_delay(self) -> float:
        return 0.0


@dataclass(frozen=True)
class Err:
    """
    실패 결과를 나타냅니다.
    
    Attributes:
        error_type: 에러 타입 (ErrorType의 구체 클래스)
    """
    error_type: ErrorType
    
    @property
    def is_retryable(self) -> bool:
        """재시도 가능 여부를 반환합니다."""
        return self.error_type.is_retryable()
    
    @property
    def retry_delay(self) -> float:
        """재시도 대기 시간(초)을 반환합니다."""
        return self.error_type.retry_delay()
    
    @property
    def error_message(self) -> str:
        """에러 메시지를 반환합니다."""
        return self.error_type.error_message
    
    @property
    def context(self) -> dict:
        """컨텍스트 정보를 반환합니다."""
        return self.error_type.context


Result = Union[Ok[T], Err]
