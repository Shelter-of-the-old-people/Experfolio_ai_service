"""
Handles the processing logic for a single portfolio.
단일 포트폴리오에 대한 처리 로직을 담당합니다.
"""
from typing import List, Dict, Tuple
import httpx
from pathlib import Path
from app.services.embedding_service import EmbeddingService
from app.repositories.portfolio_repository import PortfolioRepository
from app.infrastructure.ocr_processor import OCRProcessor
from app.core.config import settings
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
        ocr_processor: OCRProcessor
        # file_handler는 제거됨
    ):
        self._embedding_service = embedding_service
        self._portfolio_repo = portfolio_repo
        self._ocr_processor = ocr_processor

    async def process(self, portfolio: Dict) -> Result:
        """
        단일 포트폴리오를 처리하는 전체 프로세스를 실행합니다.
        """
        portfolio_id = str(portfolio.get('_id', 'unknown'))
        logger.debug(f"Starting processing for portfolio ID: {portfolio_id}")

        try:
            # 1. 텍스트 수집 (연도 정보 포함)
            texts = self._collect_texts(portfolio)

            # 2. OCR 처리 및 상태 업데이트 (R2 연동)
            attachment_texts, portfolio = await self._process_attachments(portfolio)
            texts.extend(attachment_texts)

            searchable_text = self._create_searchable_text(texts)
            if not searchable_text:
                logger.warning(f"No searchable text for portfolio ID: {portfolio_id}.")
                # 텍스트가 없어도 처리 상태는 업데이트
                await self._portfolio_repo.mark_as_processed(portfolio_id)
                return Ok(portfolio_id)

            embedding_result = self._embedding_service.embed_passage(searchable_text)

            match embedding_result:
                case Ok(kure_vector):
                    # 3. 임베딩 및 처리 완료 상태 업데이트
                    success = await self._portfolio_repo.update_embeddings_and_status(
                        portfolio_id, searchable_text, kure_vector, portfolio.get('portfolioItems', [])
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
        """포트폴리오 문서에서 연도 정보를 포함하여 텍스트 콘텐츠를 수집합니다."""
        texts = []
        basic_info = portfolio.get('basicInfo', {})
        if basic_info.get('name'): texts.append(f"이름: {basic_info['name']}")
        if basic_info.get('schoolName'): texts.append(f"학교: {basic_info['schoolName']}")
        if basic_info.get('major'): texts.append(f"전공: {basic_info['major']}")
        if basic_info.get('desiredPosition'): texts.append(f"희망직무: {basic_info['desiredPosition']}")

        for award in basic_info.get('awards', []):
            award_text = f"수상: {award.get('awardName', '')} - {award.get('achievement', '')}"
            if award.get('awardY'):
                award_text += f" ({award.get('awardY')}년)"
            texts.append(award_text)

        for cert in basic_info.get('certifications', []):
            cert_text = f"자격증: {cert.get('certificationName', '')}"
            if cert.get('issueY'):
                cert_text += f" ({cert.get('issueY')}년 취득)"
            texts.append(cert_text)

        for lang in basic_info.get('languages', []):
            lang_text = f"어학: {lang.get('testName', '')} {lang.get('score', '')}"
            if lang.get('issueY'):
                lang_text += f" ({lang.get('issueY')}년 취득)"
            texts.append(lang_text)

        for item in portfolio.get('portfolioItems', []):
            if item.get('title'): texts.append(f"제목: {item['title']}")
            if item.get('content'): texts.append(item['content'])

        return texts

    async def _process_attachments(self, portfolio: Dict) -> Tuple[List[str], Dict]:
        """
        [수정됨] R2 스토리지의 파일을 다운로드하여 OCR 처리 후, extractionStatus를 업데이트합니다.
        """
        texts = []
        portfolio_items = portfolio.get('portfolioItems', [])
        
        # HTTP 클라이언트를 사용하여 파일 다운로드
        async with httpx.AsyncClient() as client:
            for item in portfolio_items:
                for attachment in item.get('attachments', []):
                    # 이미 완료된 파일 건너뛰기
                    if attachment.get('extractionStatus') == 'completed':
                        continue

                    # objectKey 확인
                    object_key = attachment.get('objectKey')
                    original_filename = attachment.get('originalFilename', 'unknown.pdf')
                    
                    if not object_key:
                        logger.warning(f"Attachment missing objectKey: {attachment}")
                        attachment['extractionStatus'] = 'failed'
                        continue

                    # R2 URL 생성
                    base_url = settings.STORAGE_BASE_URL.rstrip('/')
                    clean_key = object_key.lstrip('/')
                    file_url = f"{base_url}/{clean_key}"

                    try:
                        logger.debug(f"Downloading file from: {file_url}")
                        response = await client.get(file_url)
                        
                        if response.status_code != 200:
                            logger.error(f"Failed to download file: {file_url}, status: {response.status_code}")
                            attachment['extractionStatus'] = 'failed'
                            continue
                        
                        file_bytes = response.content
                        
                        # 확장자 추출 (originalFilename 기반)
                        file_extension = Path(original_filename).suffix.lower()
                        if not file_extension:
                            # 확장자가 없으면 contentType 기반 추론 (간단하게 처리)
                            content_type = attachment.get('contentType', '')
                            if 'pdf' in content_type:
                                file_extension = '.pdf'
                            elif 'image' in content_type:
                                file_extension = '.jpg' # 기본값
                        
                        logger.debug(f"Processing OCR for {original_filename} ({file_extension})")
                        
                        extracted_text = self._ocr_processor.extract_text(file_bytes, file_extension)
                        
                        if extracted_text:
                            texts.append(extracted_text)
                            attachment['extractionStatus'] = 'completed'
                            logger.debug(f"Extracted {len(extracted_text)} chars from: {original_filename}")
                        else:
                            # OCR은 수행했으나 텍스트가 없는 경우 (이미지 위주 등)
                            attachment['extractionStatus'] = 'completed'
                            
                    except Exception as e:
                        logger.error(f"Failed to process attachment {original_filename}: {str(e)}")
                        attachment['extractionStatus'] = 'failed'
                        continue
                        
        return texts, portfolio

    def _create_searchable_text(self, texts: List[str]) -> str:
        """수집된 텍스트들을 하나의 문자열로 결합합니다."""
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        return '\n\n'.join(clean_texts)