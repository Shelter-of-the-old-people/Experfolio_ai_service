"""
Batch Scheduler for daily portfolio processing.
일일 포트폴리오 처리 배치 스케줄러.
"""
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from app.services.batch_service import BatchService
from app.schemas.batch import BatchResult
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BatchScheduler:
    """
    일일 배치 작업을 스케줄링하는 클래스
    APScheduler를 사용하여 매일 지정된 시간에 배치 작업을 실행합니다.
    """
    
    def __init__(self, batch_service: BatchService):
        """
        BatchScheduler 초기화
        
        Args:
            batch_service: 배치 서비스 인스턴스
        """
        self._batch_service = batch_service
        self._scheduler = AsyncIOScheduler()
        self._schedule_time = settings.BATCH_SCHEDULE_TIME  # "02:00" 형식
        self._last_result: BatchResult = None
        
        # 스케줄 시간 파싱
        self._hour, self._minute = self._parse_schedule_time(self._schedule_time)
        
        logger.info(f"BatchScheduler initialized (schedule: {self._schedule_time})")
    
    def start(self) -> None:
        """
        스케줄러를 시작합니다.
        매일 지정된 시간에 배치 작업이 실행됩니다.
        """
        try:
            logger.info("Starting batch scheduler...")
            
            # Cron 트리거 생성 (매일 지정된 시간)
            trigger = CronTrigger(
                hour=self._hour,
                minute=self._minute,
                timezone="Asia/Seoul"
            )
            
            # 작업 등록
            self._scheduler.add_job(
                self._execute_batch,
                trigger=trigger,
                id="daily_portfolio_batch",
                name="Daily Portfolio Embedding Batch",
                replace_existing=True
            )
            
            # 스케줄러 시작
            self._scheduler.start()
            
            logger.info(f"Batch scheduler started successfully")
            logger.info(f"Next run: {self._get_next_run_time()}")
            
        except Exception as e:
            logger.error(f"Failed to start batch scheduler: {str(e)}")
            raise
    
    def stop(self) -> None:
        """
        스케줄러를 중지합니다.
        """
        try:
            logger.info("Stopping batch scheduler...")
            
            if self._scheduler.running:
                self._scheduler.shutdown(wait=True)
                logger.info("Batch scheduler stopped")
            else:
                logger.warning("Batch scheduler was not running")
                
        except Exception as e:
            logger.error(f"Error stopping batch scheduler: {str(e)}")
    
    async def _execute_batch(self) -> None:
        """
        배치 작업을 실행합니다 (스케줄러 콜백).
        """
        logger.info("=" * 70)
        logger.info(f"Scheduled batch job started at {datetime.now()}")
        logger.info("=" * 70)
        
        try:
            # 배치 처리 실행
            result = await self._batch_service.process_daily_batch()
            
            # 결과 저장
            self._last_result = result
            
            # 결과 로깅
            self._log_batch_result(result)
            
            logger.info("=" * 70)
            logger.info(f"Scheduled batch job completed at {datetime.now()}")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"Scheduled batch job failed: {str(e)}")
            logger.error("=" * 70)
    
    def _log_batch_result(self, result: BatchResult) -> None:
        """
        배치 결과를 로깅합니다.
        
        Args:
            result: 배치 처리 결과
        """
        logger.info("Batch Result Summary:")
        logger.info(f"  Total: {result.total}")
        logger.info(f"  Success: {result.success} ({result.success_rate*100:.1f}%)")
        logger.info(f"  Failed: {result.failed}")
        logger.info(f"  Processing Time: {result.processingTime}")
        
        if result.failedIds:
            logger.warning(f"  Failed IDs: {', '.join(result.failedIds[:5])}")
            if len(result.failedIds) > 5:
                logger.warning(f"  ... and {len(result.failedIds) - 5} more")
    
    def _parse_schedule_time(self, time_str: str) -> tuple:
        """
        스케줄 시간 문자열을 파싱합니다.
        
        Args:
            time_str: "HH:MM" 형식의 시간 문자열
        
        Returns:
            tuple: (hour, minute)
        
        Raises:
            ValueError: 잘못된 형식
        """
        try:
            hour, minute = time_str.split(":")
            return int(hour), int(minute)
        except Exception as e:
            logger.error(f"Invalid schedule time format: {time_str}")
            raise ValueError(f"Schedule time must be in HH:MM format: {str(e)}")
    
    def _get_next_run_time(self) -> str:
        """
        다음 실행 예정 시간을 반환합니다.
        
        Returns:
            str: 다음 실행 시간
        """
        job = self._scheduler.get_job("daily_portfolio_batch")
        if job and job.next_run_time:
            return job.next_run_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        return "Not scheduled"
    
    @property
    def is_running(self) -> bool:
        """
        스케줄러 실행 상태를 반환합니다.
        
        Returns:
            bool: 실행 중이면 True
        """
        return self._scheduler.running if self._scheduler else False
    
    @property
    def last_result(self) -> BatchResult:
        """
        마지막 배치 실행 결과를 반환합니다.
        
        Returns:
            BatchResult: 마지막 배치 결과 (없으면 None)
        """
        return self._last_result
    
    @property
    def next_run_time(self) -> str:
        """
        다음 실행 예정 시간을 반환합니다.
        
        Returns:
            str: 다음 실행 시간
        """
        return self._get_next_run_time()


# 전역 스케줄러 인스턴스
_batch_scheduler: BatchScheduler = None


def get_batch_scheduler() -> BatchScheduler:
    """
    배치 스케줄러 인스턴스를 반환합니다 (싱글톤).
    
    Returns:
        BatchScheduler: 배치 스케줄러 인스턴스
    """
    global _batch_scheduler
    
    if _batch_scheduler is None:
        raise RuntimeError(
            "Batch scheduler not initialized. "
            "Call initialize_batch_scheduler() first."
        )
    
    return _batch_scheduler


def initialize_batch_scheduler(batch_service: BatchService) -> BatchScheduler:
    """
    배치 스케줄러를 초기화합니다.
    
    Args:
        batch_service: 배치 서비스 인스턴스
    
    Returns:
        BatchScheduler: 초기화된 배치 스케줄러
    """
    global _batch_scheduler
    
    if _batch_scheduler is None:
        _batch_scheduler = BatchScheduler(batch_service)
        logger.info("Batch scheduler initialized")
    
    return _batch_scheduler
