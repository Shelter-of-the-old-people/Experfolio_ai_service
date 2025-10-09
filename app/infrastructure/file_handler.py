"""
File Handler for managing portfolio files.
포트폴리오 파일 관리를 위한 핸들러.
"""
from pathlib import Path
from typing import Optional
import shutil
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class FileHandler:
    """
    파일 시스템 작업을 담당하는 클래스
    Docker Volume에서 파일을 읽고 관리합니다.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        FileHandler 초기화
        
        Args:
            base_path: 기본 경로 (기본값: settings에서 로드)
        """
        self._base_path = Path(base_path or settings.DOCKER_VOLUME_PATH)
        
        # 기본 경로 생성 (존재하지 않으면)
        self._base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FileHandler initialized with base path: {self._base_path}")
    
    def read_file(self, file_path: str) -> bytes:
        """
        파일을 읽어 바이트 데이터를 반환합니다.
        
        Args:
            file_path: 파일 경로 (절대 경로 또는 상대 경로)
        
        Returns:
            bytes: 파일 내용
        
        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            PermissionError: 파일 읽기 권한이 없을 때
        """
        try:
            # 경로 보안 검증
            validated_path = self._validate_and_resolve_path(file_path)
            
            logger.debug(f"Reading file: {validated_path}")
            
            with open(validated_path, 'rb') as f:
                content = f.read()
            
            logger.info(f"Successfully read {len(content)} bytes from {validated_path.name}")
            return content
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except PermissionError:
            logger.error(f"Permission denied: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
    
    def file_exists(self, file_path: str) -> bool:
        """
        파일 존재 여부를 확인합니다.
        
        Args:
            file_path: 파일 경로
        
        Returns:
            bool: 파일이 존재하면 True
        """
        try:
            validated_path = self._validate_and_resolve_path(file_path)
            exists = validated_path.exists() and validated_path.is_file()
            
            logger.debug(f"File exists check for {file_path}: {exists}")
            return exists
            
        except Exception as e:
            logger.warning(f"Error checking file existence {file_path}: {str(e)}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """
        파일을 삭제합니다.
        
        Args:
            file_path: 파일 경로
        
        Returns:
            bool: 삭제 성공 시 True
        """
        try:
            validated_path = self._validate_and_resolve_path(file_path)
            
            if not validated_path.exists():
                logger.warning(f"File not found for deletion: {file_path}")
                return False
            
            validated_path.unlink()
            logger.info(f"Successfully deleted file: {validated_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def delete_directory(self, user_id: str) -> bool:
        """
        사용자 폴더 전체를 삭제합니다.
        
        Args:
            user_id: 사용자 ID (UUID)
        
        Returns:
            bool: 삭제 성공 시 True
        """
        try:
            user_dir = self._base_path / user_id
            
            if not user_dir.exists():
                logger.warning(f"Directory not found for deletion: {user_id}")
                return False
            
            # 보안: base_path 하위인지 확인
            if not str(user_dir.resolve()).startswith(str(self._base_path.resolve())):
                logger.error(f"Security violation: attempted to delete outside base path")
                raise PermissionError("Cannot delete directory outside base path")
            
            shutil.rmtree(user_dir)
            logger.info(f"Successfully deleted directory: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting directory {user_id}: {str(e)}")
            return False
    
    def get_file_size(self, file_path: str) -> Optional[int]:
        """
        파일 크기를 반환합니다.
        
        Args:
            file_path: 파일 경로
        
        Returns:
            Optional[int]: 파일 크기 (bytes), 실패 시 None
        """
        try:
            validated_path = self._validate_and_resolve_path(file_path)
            
            if not validated_path.exists():
                return None
            
            size = validated_path.stat().st_size
            return size
            
        except Exception as e:
            logger.error(f"Error getting file size {file_path}: {str(e)}")
            return None
    
    def _validate_and_resolve_path(self, file_path: str) -> Path:
        """
        경로를 검증하고 절대 경로로 변환합니다.
        Path Traversal 공격 방지.
        
        Args:
            file_path: 파일 경로
        
        Returns:
            Path: 검증된 절대 경로
        
        Raises:
            ValueError: 잘못된 경로일 때
            PermissionError: 보안 위반 시
        """
        # 절대 경로인 경우 Path 객체로 변환
        if Path(file_path).is_absolute():
            full_path = Path(file_path)
        else:
            # 상대 경로인 경우 base_path와 결합
            full_path = self._base_path / file_path
        
        # 경로 정규화
        resolved_path = full_path.resolve()
        
        # 보안 검증: base_path 하위인지 확인 (Path Traversal 방지)
        try:
            resolved_path.relative_to(self._base_path.resolve())
        except ValueError:
            logger.error(f"Security violation: path outside base directory: {file_path}")
            raise PermissionError(
                f"Access denied: path must be within {self._base_path}"
            )
        
        return resolved_path
    
    @property
    def base_path(self) -> Path:
        """
        기본 경로를 반환합니다.
        
        Returns:
            Path: 기본 경로
        """
        return self._base_path
