"""
Batch Service for orchestrating daily portfolio processing.
일일 포트폴리오 배치 처리를 총괄하는 서비스.
"""
import time
from typing import List, Dict
from app.repositories.portfolio_repository import PortfolioRepository
from app.schemas.batch import BatchResult
from app.core.logging import get_logger
from app.core.result import Ok, Err
from app.services.portfolio_processor import PortfolioProcessor
from app.services.retry_executor import RetryExecutor

logger = get_logger(__name__)

class BatchService:
    """
    배치 처리 흐름을 총괄(오케스트레이션)하는 클래스.
    '어떤 포트폴리오를 처리할 것인가'와 '누구에게 처리를 맡길 것인가'에 대한 책임을 가진다.
    """
    def __init__(
        self,
        portfolio_repo: PortfolioRepository,
        processor: PortfolioProcessor,
        executor: RetryExecutor
    ):
        """
        BatchService 초기화

        Args:
            portfolio_repo: 포트폴리오 저장소
            processor: 단일 포트폴리오 처리기
            executor: 재시도 로직을 갖춘 실행기
        """
        self._portfolio_repo = portfolio_repo
        self._processor = processor
        self._executor = executor
        logger.info("BatchService initialized with Processor and Executor.")

    async def process_daily_batch(self) -> BatchResult:
        """
        일일 배치 처리를 실행합니다.
        """
        start_time = time.time()
        logger.info("Daily batch processing started.")

        try:
            portfolios = await self._portfolio_repo.find_portfolios_to_process()
            total = len(portfolios)

            if total == 0:
                logger.info("No portfolios to process today.")
                return BatchResult(total=0, success=0, failed=0, failedIds=[], processingTime="0.0s")

            logger.info(f"Found {total} portfolios to process.")
            
            success_count, failed_count = 0, 0
            failed_ids = []

            for i, portfolio in enumerate(portfolios):
                portfolio_id = str(portfolio.get('_id', 'unknown'))
                logger.info(f"Submitting portfolio {i+1}/{total} (ID: {portfolio_id}) to executor...")

                # 실행할 '작업'과 '인자'를 정의
                task = self._processor.process
                task_kwargs = {"portfolio": portfolio}

                # '실행기'에게 작업 실행을 위임. 재시도 로직은 실행기가 모두 처리.
                result = await self._executor.run(task, **task_kwargs)
                
                # 최종 결과 집계
                match result:
                    case Ok(processed_id):
                        success_count += 1
                        logger.info(f"✓ Final Succeeded for portfolio ID: {processed_id}")
                    case Err():
                        failed_count += 1
                        failed_ids.append(portfolio_id)
                        logger.error(f"✗ Final Failed for portfolio ID: {portfolio_id}. Reason: {result.error_message}")
            
            elapsed = time.time() - start_time
            result_summary = BatchResult(
                total=total,
                success=success_count,
                failed=failed_count,
                failedIds=failed_ids,
                processingTime=self._format_time(elapsed)
            )
            
            logger.info("Batch processing finished.")
            logger.info(f"Summary - Total: {total}, Success: {success_count}, Failed: {failed_count}")
            if failed_ids:
                logger.error(f"Permanently failed IDs: {failed_ids}")
            
            return result_summary
            
        except Exception as e:
            logger.error(f"Batch processing failed entirely due to an unexpected error: {e}", exc_info=True)
            elapsed = time.time() - start_time
            return BatchResult(total=0, success=0, failed=0, failedIds=[], processingTime=self._format_time(elapsed))

    def _format_time(self, seconds: float) -> str:
        """초를 읽기 쉬운 형식으로 변환합니다."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"