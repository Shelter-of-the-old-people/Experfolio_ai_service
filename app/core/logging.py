"""
Logging configuration using Loguru.
Loguru를 사용한 로깅 설정 모듈.
"""
import sys
from pathlib import Path
from loguru import logger
from app.core.config import settings


def setup_logging():
    """
    로거를 설정합니다.
    - 콘솔 출력
    - 파일 출력 (rotation)
    - 로그 레벨 설정
    """
    # 기본 핸들러 제거
    logger.remove()
    
    # 로그 파일 디렉토리 생성
    log_file_path = Path(settings.LOG_FILE)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 콘솔 핸들러 추가 (컬러풀한 출력)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True,
    )
    
    # 파일 핸들러 추가 (로테이션)
    logger.add(
        settings.LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
        rotation="100 MB",  # 100MB마다 새 파일
        retention="30 days",  # 30일간 보관
        compression="zip",  # 압축 저장
        encoding="utf-8",
    )
    
    # 에러 로그 별도 파일
    error_log_path = log_file_path.parent / "error.log"
    logger.add(
        str(error_log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="50 MB",
        retention="60 days",
        compression="zip",
        encoding="utf-8",
    )
    
    logger.info("Logging configured successfully")
    logger.info(f"Log level: {settings.LOG_LEVEL}")
    logger.info(f"Log file: {settings.LOG_FILE}")


def get_logger(name: str = None):
    """
    로거 인스턴스를 반환합니다.
    
    Args:
        name: 로거 이름 (모듈명 등)
    
    Returns:
        logger: Loguru logger 인스턴스
    """
    if name:
        return logger.bind(name=name)
    return logger


# 모듈 import 시 자동 설정
setup_logging()
