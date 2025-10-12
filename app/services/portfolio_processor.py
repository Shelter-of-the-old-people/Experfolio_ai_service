"""
Handles the processing logic for a single portfolio.
단일 포트폴리오에 대한 처리 로직을 담당합니다.
"""
from typing import List, Dict
from app.services.embedding_service import EmbeddingService
from app.repositories.portfolio_repository import PortfolioRepository
from app.infrastructure.ocr_processor import OCRProcessor
from app.infrastructure.file_handler import FileHandler
from app.core.logging import get_logger
from app.core.result import Result, Ok, Err, InvalidDataError, NetworkError, SystemError

logger = get_logger(__name__)

class PortfolioProcessor:
    """
    단일 포트폴리오 처리의 비즈니스 로직을 캡슐화하는 클래스.
    '어떻게 포트폴리오를 처리할 것인가'에 대한 책임을 가진다.
    """
    def __init__(
        self,
        embedding_service: EmbeddingService,
        portfolio_repo: PortfolioRepository,
        ocr_processor: OCRProcessor,
        file_handler: FileHandler
    ):
        self._embedding_service = embedding_service
        self._portfolio_repo = portfolio_repo
        self._ocr_processor = ocr_processor
        self._file_handler = file_handler

    async def process(self, portfolio: Dict) -> Result:
        """
        단일 포트폴리오를 처리하는 전체 프로세스를 실행합니다.

        Args:
            portfolio: 처리할 포트폴리오 문서

        Returns:
            Result: 성공 시 Ok(portfolio_id), 실패 시 Err(error_type)
        """
        portfolio_id = str(portfolio.get('_id', 'unknown'))
        logger.debug(f"Starting processing for portfolio ID: {portfolio_id}")

        try:
            texts = self._collect_texts(portfolio)
            attachment_texts = await self._process_attachments(portfolio)
            texts.extend(attachment_texts)
            
            searchable_text = self._create_searchable_text(texts)
            if not searchable_text:
                logger.warning(f"No searchable text for portfolio ID: {portfolio_id}.")
                return Err(InvalidDataError(error=ValueError("No searchable text found"), context={"portfolio_id": portfolio_id}))

            embedding_result = self._embedding_service.embed_passage(searchable_text)
            
            match embedding_result:
                case Ok(kure_vector):
                    success = await self._portfolio_repo.update_embeddings(
                        portfolio_id, searchable_text, kure_vector
                    )
                    if success:
                        return Ok(portfolio_id)
                    else:
                        return Err(NetworkError(error=Exception("DB update failed"), context={"portfolio_id": portfolio_id}))
                
                case Err():
                    return embedding_result
            
        except Exception as e:
            logger.error(f"Unexpected error in PortfolioProcessor for {portfolio_id}: {e}", exc_info=True)
            return Err(SystemError(error=e, context={"portfolio_id": portfolio_id}))

    def _collect_texts(self, portfolio: Dict) -> List[str]:
        """포트폴리오 문서에서 텍스트 콘텐츠를 수집합니다."""
        texts = []
        basic_info = portfolio.get('basicInfo', {})
        if basic_info.get('name'): texts.append(f"이름: {basic_info['name']}")
        if basic_info.get('schoolName'): texts.append(f"학교: {basic_info['schoolName']}")
        if basic_info.get('major'): texts.append(f"전공: {basic_info['major']}")
        if basic_info.get('desiredPosition'): texts.append(f"희망직무: {basic_info['desiredPosition']}")
        for award in basic_info.get('awards', []): texts.append(f"수상: {award.get('awardName', '')} - {award.get('achievement', '')}")
        for cert in basic_info.get('certifications', []): texts.append(f"자격증: {cert.get('certificationName', '')}")
        for lang in basic_info.get('languages', []): texts.append(f"어학: {lang.get('testName', '')} {lang.get('score', '')}")
        for item in portfolio.get('portfolioItems', []):
            if item.get('title'): texts.append(f"제목: {item['title']}")
            if item.get('content'): texts.append(item['content'])
        return texts

    async def _process_attachments(self, portfolio: Dict) -> List[str]:
        """포트폴리오의 첨부 파일에서 텍스트를 추출합니다."""
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
        """수집된 텍스트들을 하나의 문자열로 결합합니다."""
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        return '\n\n'.join(clean_texts)