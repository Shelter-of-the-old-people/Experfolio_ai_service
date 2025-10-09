"""
OCR Processor using Tesseract and pdf2image.
Tesseract와 pdf2image를 사용한 OCR 처리.
"""
from typing import Optional, Dict
from io import BytesIO
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from app.core.logging import get_logger

logger = get_logger(__name__)


class OCRProcessor:
    """
    OCR 처리를 담당하는 클래스
    PDF와 이미지 파일에서 텍스트를 추출합니다.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        OCR Processor 초기화
        
        Args:
            config: Tesseract 설정 (Optional)
        """
        self._tesseract_config = config or {
            'lang': 'kor+eng',  # 한국어 + 영어
            'config': '--psm 6'  # 단일 블록 텍스트 가정
        }
        
        self._supported_formats = ['.pdf', '.png', '.jpg', '.jpeg']
        
        logger.info(f"OCRProcessor initialized with config: {self._tesseract_config}")
    
    def extract_text(self, file_bytes: bytes, file_extension: str) -> str:
        """
        파일 타입을 자동 감지하여 텍스트를 추출합니다.
        
        Args:
            file_bytes: 파일 바이트 데이터
            file_extension: 파일 확장자 (예: '.pdf', '.jpg')
        
        Returns:
            str: 추출된 텍스트
        
        Raises:
            ValueError: 지원하지 않는 파일 형식
        """
        file_ext = file_extension.lower()
        
        if file_ext not in self._supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {self._supported_formats}"
            )
        
        try:
            if file_ext == '.pdf':
                return self._extract_from_pdf(file_bytes)
            else:
                return self._extract_from_image(file_bytes)
                
        except Exception as e:
            logger.error(f"OCR extraction failed for {file_ext}: {str(e)}")
            return ""  # 실패 시 빈 문자열 반환
    
    def _extract_from_pdf(self, file_bytes: bytes) -> str:
        """
        PDF 파일에서 텍스트를 추출합니다.
        
        Args:
            file_bytes: PDF 파일 바이트 데이터
        
        Returns:
            str: 추출된 텍스트
        """
        try:
            logger.debug("Converting PDF to images...")
            
            # PDF를 이미지로 변환
            images = convert_from_bytes(
                file_bytes,
                dpi=300,  # 고해상도
                fmt='jpeg',
                thread_count=2
            )
            
            logger.debug(f"Converted PDF to {len(images)} images")
            
            # 각 페이지에서 텍스트 추출
            all_text = []
            for i, image in enumerate(images):
                logger.debug(f"Processing page {i+1}/{len(images)}")
                
                # 이미지 전처리
                processed_image = self._preprocess_image(image)
                
                # OCR 수행
                text = pytesseract.image_to_string(
                    processed_image,
                    lang=self._tesseract_config['lang'],
                    config=self._tesseract_config['config']
                )
                
                if text.strip():
                    all_text.append(text.strip())
            
            result = '\n\n'.join(all_text)
            logger.info(f"Extracted {len(result)} characters from PDF")
            
            return result
            
        except Exception as e:
            logger.error(f"PDF OCR failed: {str(e)}")
            return ""
    
    def _extract_from_image(self, file_bytes: bytes) -> str:
        """
        이미지 파일에서 텍스트를 추출합니다.
        
        Args:
            file_bytes: 이미지 파일 바이트 데이터
        
        Returns:
            str: 추출된 텍스트
        """
        try:
            logger.debug("Processing image...")
            
            # 이미지 열기
            image = Image.open(BytesIO(file_bytes))
            
            # 이미지 전처리
            processed_image = self._preprocess_image(image)
            
            # OCR 수행
            text = pytesseract.image_to_string(
                processed_image,
                lang=self._tesseract_config['lang'],
                config=self._tesseract_config['config']
            )
            
            result = text.strip()
            logger.info(f"Extracted {len(result)} characters from image")
            
            return result
            
        except Exception as e:
            logger.error(f"Image OCR failed: {str(e)}")
            return ""
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        OCR 정확도 향상을 위한 이미지 전처리
        
        Args:
            image: PIL Image 객체
        
        Returns:
            Image.Image: 전처리된 이미지
        """
        try:
            # RGB 모드로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 그레이스케일 변환 (OCR 성능 향상)
            image = image.convert('L')
            
            # 이미지 크기 조정 (너무 작으면 확대)
            width, height = image.size
            if width < 1000:
                scale_factor = 1000 / width
                new_size = (int(width * scale_factor), int(height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}, using original")
            return image
