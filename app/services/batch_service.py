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
        self._max_retries = 3
        
        logger.info("BatchService initialized")
    
    async def process_daily_batch(self) -> BatchResult:
        """
        일일 배치 처리를 실행합니다.
        
        Returns:
            BatchResult: 배치 처리 결과
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("Daily batch processing started")
        logger.info("=" * 60)
        
        try:
            # 1. 처리 대상 포트폴리오 조회
            logger.info("Step 1: Finding portfolios needing embedding...")
            portfolios = await self._portfolio_repo.find_needing_embedding()
            total = len(portfolios)
            
            logger.info(f"Found {total} portfolios to process")
            
            if total == 0:
                logger.info("No portfolios to process")
                return BatchResult(
                    total=0,
                    success=0,
                    failed=0,
                    failedIds=[],
                    processingTime="0s"
                )
            
            # 2. 각 포트폴리오 처리
            success_count = 0
            failed_count = 0
            failed_ids = []
            
            for i, portfolio in enumerate(portfolios):
                portfolio_id = str(portfolio['_id'])
                logger.info(f"Processing {i+1}/{total}: {portfolio_id}")
                
                try:
                    result = await self.process_single_portfolio(portfolio)
                    
                    if result:
                        success_count += 1
                        logger.info(f"✓ Success: {portfolio_id}")
                    else:
                        failed_count += 1
                        failed_ids.append(portfolio_id)
                        logger.warning(f"✗ Failed: {portfolio_id}")
                        
                except Exception as e:
                    failed_count += 1
                    failed_ids.append(portfolio_id)
                    logger.error(f"✗ Error processing {portfolio_id}: {str(e)}")
            
            # 3. 결과 생성
            elapsed = time.time() - start_time
            processing_time = self._format_time(elapsed)
            
            result = BatchResult(
                total=total,
                success=success_count,
                failed=failed_count,
                failedIds=failed_ids,
                processingTime=processing_time
            )
            
            logger.info("=" * 60)
            logger.info(f"Batch processing completed in {processing_time}")
            logger.info(f"Success: {success_count}/{total} ({result.success_rate*100:.1f}%)")
            logger.info(f"Failed: {failed_count}/{total}")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            elapsed = time.time() - start_time
            
            return BatchResult(
                total=0,
                success=0,
                failed=0,
                failedIds=[],
                processingTime=self._format_time(elapsed)
            )
    
    async def process_single_portfolio(self, portfolio: Dict) -> bool:
        """
        단일 포트폴리오를 처리합니다.
        
        Args:
            portfolio: 포트폴리오 문서
        
        Returns:
            bool: 처리 성공 여부
        """
        portfolio_id = str(portfolio['_id'])
        
        try:
            logger.debug(f"Processing portfolio: {portfolio_id}")
            
            # 1. 텍스트 수집
            texts = self._collect_texts(portfolio)
            logger.debug(f"Collected {len(texts)} text sections")
            
            # 2. 첨부파일 처리
            attachment_texts = await self._process_attachments(portfolio)
            texts.extend(attachment_texts)
            logger.debug(f"Total {len(texts)} text sections after attachments")
            
            # 3. searchableText 생성
            searchable_text = self._create_searchable_text(texts)
            logger.debug(f"Created searchable text: {len(searchable_text)} characters")
            
            if not searchable_text:
                logger.warning(f"No searchable text for portfolio: {portfolio_id}")
                return False
            
            # 4. 임베딩 생성
            logger.debug("Generating embedding...")
            kure_vector = self._embedding_service.embed_passage(searchable_text)
            
            # 5. MongoDB 업데이트
            logger.debug("Updating portfolio in MongoDB...")
            success = await self._portfolio_repo.update_embeddings(
                portfolio_id,
                searchable_text,
                kure_vector
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to process portfolio {portfolio_id}: {str(e)}")
            return False
    
    def _collect_texts(self, portfolio: Dict) -> List[str]:
        """
        포트폴리오에서 모든 텍스트를 수집합니다.
        
        Args:
            portfolio: 포트폴리오 문서
        
        Returns:
            List[str]: 수집된 텍스트 목록
        """
        texts = []
        
        # basicInfo에서 텍스트 수집
        basic_info = portfolio.get('basicInfo', {})
        
        if basic_info.get('name'):
            texts.append(f"이름: {basic_info['name']}")
        if basic_info.get('schoolName'):
            texts.append(f"학교: {basic_info['schoolName']}")
        if basic_info.get('major'):
            texts.append(f"전공: {basic_info['major']}")
        if basic_info.get('desiredPosition'):
            texts.append(f"희망직무: {basic_info['desiredPosition']}")
        
        # 수상경력
        for award in basic_info.get('awards', []):
            texts.append(f"수상: {award.get('awardName', '')} - {award.get('achievement', '')}")
        
        # 자격증
        for cert in basic_info.get('certifications', []):
            texts.append(f"자격증: {cert.get('certificationName', '')}")
        
        # 어학능력
        for lang in basic_info.get('languages', []):
            texts.append(f"어학: {lang.get('testName', '')} {lang.get('score', '')}")
        
        # portfolioItems에서 텍스트 수집
        for item in portfolio.get('portfolioItems', []):
            if item.get('title'):
                texts.append(f"제목: {item['title']}")
            if item.get('content'):
                texts.append(item['content'])
        
        return texts
    
    async def _process_attachments(self, portfolio: Dict) -> List[str]:
        """
        첨부파일을 처리하여 텍스트를 추출합니다.
        
        Args:
            portfolio: 포트폴리오 문서
        
        Returns:
            List[str]: 추출된 텍스트 목록
        """
        texts = []
        
        for item in portfolio.get('portfolioItems', []):
            for attachment in item.get('attachments', []):
                try:
                    file_path = attachment.get('filePath')
                    
                    if not file_path:
                        continue
                    
                    # 파일 존재 확인
                    if not self._file_handler.file_exists(file_path):
                        logger.warning(f"File not found: {file_path}")
                        continue
                    
                    # 파일 읽기
                    file_bytes = self._file_handler.read_file(file_path)
                    
                    # 파일 확장자 추출
                    file_extension = file_path.split('.')[-1]
                    file_extension = f".{file_extension.lower()}"
                    
                    # OCR 처리
                    extracted_text = self._ocr_processor.extract_text(
                        file_bytes, 
                        file_extension
                    )
                    
                    if extracted_text:
                        texts.append(extracted_text)
                        logger.debug(f"Extracted {len(extracted_text)} chars from {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing attachment {file_path}: {str(e)}")
                    continue
        
        return texts
    
    def _create_searchable_text(self, texts: List[str]) -> str:
        """
        텍스트 목록을 하나의 검색 가능한 텍스트로 결합합니다.
        
        Args:
            texts: 텍스트 목록
        
        Returns:
            str: 결합된 텍스트
        """
        # 빈 텍스트 제거 및 공백 정리
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        
        # '\n\n'로 결합
        searchable_text = '\n\n'.join(clean_texts)
        
        return searchable_text
    
    def _format_time(self, seconds: float) -> str:
        """
        초를 읽기 쉬운 형식으로 변환합니다.
        
        Args:
            seconds: 초 단위 시간
        
        Returns:
            str: 포맷된 시간 (예: "15m 30s")
        """
        if seconds < 60:
            return f"{seconds:.0f}s"
        
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        
        return f"{minutes}m {remaining_seconds}s"
