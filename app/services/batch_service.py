"""
Batch Service for daily portfolio processing.
일일 포트폴리오 배치 처리 서비스.
"""
import time
from typing import List, Dict
from datetime import datetime
from app.services.embedding_service import EmbeddingService
from app.repositories.portfolio_repository import PortfolioRepository
from app.infrastructure.ocr_processor import OCRProcessor
from app.infrastructure.file_handler import FileHandler
from app.schemas.batch import BatchResult
from app.core.logging import get_logger
from app.core.result import Result, Ok, Err, InvalidDataError, NetworkError, SystemError

logger = get_logger(__name__)


class BatchService:
    """
    포트폴리오 임베딩 배치 처리 서비스
    매일 새벽 2시에 실행되어 needsEmbedding=true인 포트폴리오를 처리합니다.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        portfolio_repo: PortfolioRepository,
        ocr_processor: OCRProcessor,
        file_handler: FileHandler
    ):
        """
        BatchService 초기화
        
        Args:
            embedding_service: 임베딩 서비스
            portfolio_repo: 포트폴리오 저장소
            ocr_processor: OCR 처리기
            file_handler: 파일 핸들러
        """
        self._embedding_service = embedding_service
        self._portfolio_repo = portfolio_repo
        self._ocr_processor = ocr_processor
        self._file_handler = file_handler
        
        logger.info("BatchService initialized")
    
    async def process_daily_batch(self) -> BatchResult:
        """
        일일 배치 처리를 실행합니다.
        
        Returns:
            BatchResult: 배치 처리 결과
        """
        start_time = time.time()
        logger.info("Daily batch processing started.")
        
        try:
            portfolios = await self._portfolio_repo.find_needing_embedding()
            total = len(portfolios)
            
            if total == 0:
                logger.info("No portfolios to process today.")
                return BatchResult(total=0, success=0, failed=0, failedIds=[], processingTime="0.0s")
            
            logger.info(f"Found {total} portfolios to process.")
            
            success_count, failed_count = 0, 0
            failed_ids, retry_ids = [], []
            
            for i, portfolio in enumerate(portfolios):
                portfolio_id = str(portfolio.get('_id', 'unknown'))
                logger.info(f"Processing portfolio {i+1}/{total} (ID: {portfolio_id})...")
                
                result = await self.process_single_portfolio(portfolio)
                
                match result:
                    case Ok(processed_id):
                        success_count += 1
                        logger.info(f"✓ Successfully processed portfolio ID: {processed_id}")
                    
                    case Err() if result.is_retryable:
                        failed_count += 1
                        retry_ids.append(portfolio_id)
                        logger.warning(
                            f"⚠ Retryable failure for portfolio ID {portfolio_id}: {result.error_message}. "
                            "This can be retried later."
                        )
                    
                    case Err():
                        failed_count += 1
                        failed_ids.append(portfolio_id)
                        logger.error(f"✗ Permanent failure for portfolio ID {portfolio_id}: {result.error_message}")
            
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
            if retry_ids:
                logger.warning(f"Retryable failures (not retried): {retry_ids}")
            
            return result_summary
            
        except Exception as e:
            logger.error(f"Batch processing failed entirely due to an unexpected error: {str(e)}", exc_info=True)
            elapsed = time.time() - start_time
            return BatchResult(total=0, success=0, failed=0, failedIds=[], processingTime=self._format_time(elapsed))

    async def process_single_portfolio(self, portfolio: Dict) -> Result:
        """
        단일 포트폴리오를 처리합니다.
        
        Args:
            portfolio: 포트폴리오 문서
        
        Returns:
            Result:
                - Ok(str): 처리된 portfolio_id
                - Err: 에러 정보
        """
        portfolio_id = str(portfolio.get('_id', 'unknown'))
        logger.debug(f"Starting single process for portfolio ID: {portfolio_id}")
        
        try:
            texts = self._collect_texts(portfolio)
            attachment_texts = await self._process_attachments(portfolio)
            texts.extend(attachment_texts)
            
            searchable_text = self._create_searchable_text(texts)
            if not searchable_text:
                logger.warning(f"No searchable text generated for portfolio ID: {portfolio_id}. Marking as failure.")
                return Err(InvalidDataError(error=ValueError("No searchable text"), context={"portfolio_id": portfolio_id}))
            
            embedding_result = self._embedding_service.embed_passage(searchable_text)
            
            match embedding_result:
                case Ok(kure_vector):
                    logger.debug(f"Embedding generated for {portfolio_id}, updating in DB...")
                    success = await self._portfolio_repo.update_embeddings(portfolio_id, searchable_text, kure_vector)
                    if success:
                        return Ok(portfolio_id)
                    else:
                        return Err(NetworkError(error=Exception("DB update failed"), context={"portfolio_id": portfolio_id}))
                
                case Err():
                    logger.error(f"Embedding failed for {portfolio_id}: {embedding_result.error_message}")
                    return embedding_result
            
        except Exception as e:
            logger.error(f"Unexpected error processing portfolio {portfolio_id}: {str(e)}", exc_info=True)
            return Err(SystemError(error=e, context={"portfolio_id": portfolio_id}))

    def _collect_texts(self, portfolio: Dict) -> List[str]:
        """포트폴리오에서 모든 텍스트를 수집합니다."""
        texts = []
        basic_info = portfolio.get('basicInfo', {})
        if basic_info.get('name'):
            texts.append(f"이름: {basic_info['name']}")
        if basic_info.get('schoolName'):
            texts.append(f"학교: {basic_info['schoolName']}")
        if basic_info.get('major'):
            texts.append(f"전공: {basic_info['major']}")
        if basic_info.get('desiredPosition'):
            texts.append(f"희망직무: {basic_info['desiredPosition']}")
        for award in basic_info.get('awards', []):
            texts.append(f"수상: {award.get('awardName', '')} - {award.get('achievement', '')}")
        for cert in basic_info.get('certifications', []):
            texts.append(f"자격증: {cert.get('certificationName', '')}")
        for lang in basic_info.get('languages', []):
            texts.append(f"어학: {lang.get('testName', '')} {lang.get('score', '')}")
        for item in portfolio.get('portfolioItems', []):
            if item.get('title'):
                texts.append(f"제목: {item['title']}")
            if item.get('content'):
                texts.append(item['content'])
        return texts

    async def _process_attachments(self, portfolio: Dict) -> List[str]:
        """첨부파일을 처리하여 텍스트를 추출합니다."""
        texts = []
        for item in portfolio.get('portfolioItems', []):
            for attachment in item.get('attachments', []):
                file_path = attachment.get('filePath')
                if not file_path:
                    continue
                try:
                    if not self._file_handler.file_exists(file_path):
                        logger.warning(f"Attachment file not found, skipping: {file_path}")
                        continue
                    
                    file_bytes = self._file_handler.read_file(file_path)
                    file_extension = '.' + file_path.split('.')[-1].lower()
                    
                    extracted_text = self._ocr_processor.extract_text(file_bytes, file_extension)
                    if extracted_text:
                        texts.append(extracted_text)
                        logger.debug(f"Extracted {len(extracted_text)} chars from attachment: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to process attachment {file_path}: {str(e)}", exc_info=True)
                    continue
        return texts

    def _create_searchable_text(self, texts: List[str]) -> str:
        """텍스트 목록을 하나의 검색 가능한 텍스트로 결합합니다."""
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        return '\n\n'.join(clean_texts)

    def _format_time(self, seconds: float) -> str:
        """초를 읽기 쉬운 형식으로 변환합니다."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"