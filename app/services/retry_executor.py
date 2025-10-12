"""
A generic executor that handles retry logic for tasks.
작업에 대한 재시도 로직을 처리하는 범용 실행기.
"""
import asyncio
from typing import Callable, Awaitable
from app.core.result import Result, Ok, Err
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

class RetryExecutor:
    """
    재시도 로직을 캡슐화하여 작업을 실행하는 클래스.
    '어떻게 재시도할 것인가'에 대한 책임을 가진다.
    """
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0):
        """
        RetryExecutor 초기화

        Args:
            max_retries: 최대 재시도 횟수
            initial_delay: 초기 대기 시간 (초)
        """
        self._max_retries = max_retries
        self._initial_delay = initial_delay

    async def run(self, task: Callable[..., Awaitable[Result]], **kwargs) -> Result:
        """
        주어진 작업을 재시도 로직을 적용하여 실행합니다.

        Args:
            task: 실행할 비동기 함수 (Result 객체를 반환해야 함)
            **kwargs: task 함수에 전달할 인자들

        Returns:
            Result: 작업의 최종 결과 (성공 또는 모든 재시도 후의 실패)
        """
        last_error_result = None
        
        for attempt in range(self._max_retries):
            try:
                result = await task(**kwargs)

                match result:
                    case Ok():
                        # 성공 시, 즉시 결과를 반환하고 종료
                        if attempt > 0:
                            logger.info(f"Task succeeded after {attempt + 1} attempts.")
                        return result
                    
                    case Err() if result.is_retryable and attempt < self._max_retries - 1:
                        # 재시도 가능한 에러이고, 아직 재시도 기회가 남았을 경우
                        last_error_result = result
                        # 지수 백오프: 1초, 2초, 4초...
                        wait_time = self._initial_delay * (2 ** attempt)
                        
                        logger.warning(
                            f"Task failed (attempt {attempt + 1}/{self._max_retries}). "
                            f"Reason: {result.error_message}. Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                        continue # 다음 재시도 실행
                    
                    case Err():
                        # 영구 실패 또는 모든 재시도 기회를 소진한 경우
                        if result.is_retryable:
                            logger.error(f"Task failed after {self._max_retries} attempts. Reason: {result.error_message}")
                        return result

            except Exception as e:
                logger.error(f"An unexpected exception occurred during task execution: {e}", exc_info=True)
                from app.core.result import SystemError
                return Err(SystemError(error=e))
        
        # 모든 재시도 실패 시 마지막으로 기록된 에러를 반환
        return last_error_result