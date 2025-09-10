
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

##  PDF 이미지 추출 클래스
class PDFImageExtractor:
    """PDF에서 이미지를 추출하는 클래스"""
    
    def __init__(self, dpi: int = 300, max_size: int = 2048, target_size: int = None):
        """
        Args:
            dpi: 이미지 해상도 (기본값: 300)
            max_size: 최대 이미지 크기 (기본값: 2048)
            target_size: 목표 이미지 크기 (None이면 자동)
        """
        self.dpi = dpi
        self.max_size = max_size
        self.target_size = target_size
        # A4 크기 정의 (포인트 단위: 72 DPI 기준)
        self.A4_WIDTH = 595  # 210mm
        self.A4_HEIGHT = 842  # 297mm
        
    def extract_images_from_pdf(self, pdf_path: str, output_dir: str = None, split_large_pages: bool = True) -> List[str]:
        """
        PDF에서 각 페이지를 이미지로 추출 (큰 페이지는 분할)
        
        Args:
            pdf_path: PDF 파일 경로
            output_dir: 이미지 저장 디렉토리 (None이면 임시 디렉토리)
            split_large_pages: A4 이상 크기 페이지를 2개로 분할할지 여부
            
        Returns:
            추출된 이미지 파일 경로 리스트
        """
        if output_dir is None:
            output_dir = "temp_images"
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # PDF 파일명 추출 (확장자 제거)
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # PDF 열기
        pdf_document = fitz.open(pdf_path)
        image_paths = []
        
        # 각 페이지를 이미지로 변환
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # 페이지 크기 확인
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height
            
            print(f"페이지 {page_num + 1} 크기: {page_width:.1f} x {page_height:.1f} pt")
            
            # 페이지 분할 여부 결정
            should_split = (split_large_pages and 
                          (page_width > self.A4_WIDTH * 1.5 or page_height > self.A4_HEIGHT * 1.5))
            
            if should_split:
                print(f"  📄 큰 페이지 감지 - 2개로 분할하여 추출")
                split_images = self._extract_split_page(page, page_num, output_dir, pdf_filename)
                image_paths.extend(split_images)
            else:
                # 일반 페이지 추출
                image_path = self._extract_single_page(page, page_num, output_dir, pdf_filename)
                image_paths.append(image_path)
            
            page = None
        
        pdf_document.close()
        pdf_document = None
        return image_paths
    
    def _extract_single_page(self, page, page_num: int, output_dir: str, pdf_filename: str) -> str:
        """
        단일 페이지 이미지 추출
        
        Args:
            page: PDF 페이지 객체
            page_num: 페이지 번호
            output_dir: 출력 디렉토리
            pdf_filename: PDF 파일명 (확장자 제외)
            
        Returns:
            추출된 이미지 파일 경로
        """
        # 페이지를 이미지로 렌더링
        mat = fitz.Matrix(self.dpi/72, self.dpi/72)  # DPI 설정
        pix = page.get_pixmap(matrix=mat)
        
        # 이미지 크기
        width, height = pix.width, pix.height
        optimized_path = os.path.join(output_dir, f"{pdf_filename}_page_{page_num + 1}.png")
        
        # 이미지 저장
        pix.save(optimized_path)
        pix = None
        
        # 크기 최적화 적용
        if self.max_size or self.target_size:
            optimized_path = self._optimize_image_size(optimized_path, width, height)
        
        print(f"페이지 {page_num + 1} 추출 완료: {optimized_path}")
        return optimized_path
    
    def _extract_split_page(self, page, page_num: int, output_dir: str, pdf_filename: str) -> List[str]:
        """
        큰 페이지를 2개로 분할하여 추출
        
        Args:
            page: PDF 페이지 객체
            page_num: 페이지 번호
            output_dir: 출력 디렉토리
            pdf_filename: PDF 파일명 (확장자 제외)
            
        Returns:
            분할된 이미지 파일 경로 리스트
        """
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        split_images = []
        
        # 가로/세로 중 더 긴 쪽을 기준으로 분할 방향 결정
        if page_width > page_height:
            # 가로로 긴 경우 - 세로로 2분할
            split_type = "vertical"
            split_width = page_width / 2
            
            # 왼쪽 반쪽
            left_rect = fitz.Rect(0, 0, split_width, page_height)
            left_image = self._extract_page_region(page, left_rect, page_num, "left", output_dir, pdf_filename)
            split_images.append(left_image)
            
            # 오른쪽 반쪽
            right_rect = fitz.Rect(split_width, 0, page_width, page_height)
            right_image = self._extract_page_region(page, right_rect, page_num, "right", output_dir, pdf_filename)
            split_images.append(right_image)
            
            print(f"  ✂️  세로 분할: {page_width:.1f} x {page_height:.1f} → 2개 ({split_width:.1f} x {page_height:.1f} 각각)")
            
        else:
            # 세로로 긴 경우 - 가로로 2분할
            split_type = "horizontal"
            split_height = page_height / 2
            
            # 위쪽 반쪽
            top_rect = fitz.Rect(0, 0, page_width, split_height)
            top_image = self._extract_page_region(page, top_rect, page_num, "top", output_dir, pdf_filename)
            split_images.append(top_image)
            
            # 아래쪽 반쪽
            bottom_rect = fitz.Rect(0, split_height, page_width, page_height)
            bottom_image = self._extract_page_region(page, bottom_rect, page_num, "bottom", output_dir, pdf_filename)
            split_images.append(bottom_image)
            
            print(f"  ✂️  가로 분할: {page_width:.1f} x {page_height:.1f} → 2개 ({page_width:.1f} x {split_height:.1f} 각각)")
        
        return split_images
    
    def _extract_page_region(self, page, rect: fitz.Rect, page_num: int, region: str, output_dir: str, pdf_filename: str) -> str:
        """
        페이지의 특정 영역을 이미지로 추출
        
        Args:
            page: PDF 페이지 객체
            rect: 추출할 영역 (fitz.Rect)
            page_num: 페이지 번호
            region: 영역 이름 (left, right, top, bottom)
            output_dir: 출력 디렉토리
            pdf_filename: PDF 파일명 (확장자 제외)
            
        Returns:
            추출된 이미지 파일 경로
        """
        # DPI 매트릭스 설정
        mat = fitz.Matrix(self.dpi/72, self.dpi/72)
        
        # 지정된 영역만 렌더링
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        # 이미지 크기
        width, height = pix.width, pix.height
        optimized_path = os.path.join(output_dir, f"{pdf_filename}_page_{page_num + 1}_{region}.png")
        
        # 이미지 저장
        pix.save(optimized_path)
        
        # 크기 최적화 적용
        if self.max_size or self.target_size:
            optimized_path = self._optimize_image_size(optimized_path, width, height)
        
        print(f"    📄 {region} 영역 추출: {optimized_path} ({width}x{height})")
        return optimized_path
    
    def _optimize_image_size(self, image_path: str, original_width: int, original_height: int) -> str:
        """
        PaddleOCR 최적화를 위한 이미지 크기 조정
        
        Args:
            image_path: 원본 이미지 경로
            original_width: 원본 너비
            original_height: 원본 높이
            
        Returns:
            최적화된 이미지 경로
        """
        from PIL import Image
        
        # 이미지 열기
        img = Image.open(image_path)
        
        # 목표 크기 결정
        if self.target_size:
            # 고정 크기로 리사이즈 (비율 유지)
            img.thumbnail((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        elif self.max_size:
            # 최대 크기 제한 (비율 유지)
            max_dim = max(original_width, original_height)
            if max_dim > self.max_size:
                scale_factor = self.max_size / max_dim
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # PaddleOCR 최적 크기 조정 (32의 배수로)
        width, height = img.size
        optimized_width = ((width + 31) // 32) * 32
        optimized_height = ((height + 31) // 32) * 32
        
        if width != optimized_width or height != optimized_height:
            # 패딩 추가하여 32의 배수로 만들기
            new_img = Image.new('RGB', (optimized_width, optimized_height), 'white')
            new_img.paste(img, ((optimized_width - width) // 2, (optimized_height - height) // 2))
            img = new_img
        
        # 최적화된 이미지 저장
        optimized_path = image_path.replace('.png', '_optimized.png')
        img.save(optimized_path, 'PNG', quality=95)

        print(f"    🔧 크기 최적화: {original_width}x{original_height} → {img.size[0]}x{img.size[1]}")
                
        img = None
        # 원본 파일 삭제
        os.remove(image_path)
        return optimized_path
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        OCR을 위한 이미지 전처리 (PaddleOCR 최적화)
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            전처리된 이미지 배열
        """
        # 이미지 읽기
        image = cv2.imread(image_path)
        
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # PaddleOCR은 RGB 입력을 선호하므로 다시 3채널로 변환
        if len(gray.shape) == 2:
            rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
        
        # 이미지 품질 향상
        # 1. 노이즈 제거
        denoised = cv2.bilateralFilter(rgb_image, 9, 75, 75)
        
        # 2. 대비 향상 (각 채널별로)
        enhanced = np.zeros_like(denoised)
        for i in range(3):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced[:,:,i] = clahe.apply(denoised[:,:,i])
        
        # 3. 샤프닝
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        
        sharpened = np.zeros_like(enhanced)
        for i in range(3):
            sharpened[:,:,i] = cv2.filter2D(enhanced[:,:,i], -1, kernel)
        
        return sharpened
    
    def get_optimal_size_info(self, image_path: str) -> Dict:
        """
        이미지의 최적 크기 정보 분석
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            최적화 정보
        """
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # PaddleOCR 효율성 분석
        total_pixels = width * height
        aspect_ratio = width / height
        
        # 최적화 권장사항
        recommendations = []
        
        if total_pixels > 4096 * 4096:
            recommendations.append("이미지가 너무 큽니다. 4096x4096 이하로 축소 권장")
        elif total_pixels < 640 * 480:
            recommendations.append("이미지가 작습니다. 해상도 향상 고려")
        
        if width % 32 != 0 or height % 32 != 0:
            recommendations.append("32의 배수 크기로 패딩 추가 권장")
        
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            recommendations.append("극단적인 종횡비입니다. 분할 처리 고려")
        
        return {
            'current_size': (width, height),
            'total_pixels': total_pixels,
            'aspect_ratio': aspect_ratio,
            'is_optimal_size': len(recommendations) == 0,
            'recommendations': recommendations,
            'optimal_dpi_range': self._calculate_optimal_dpi(width, height),
            'memory_usage_mb': (total_pixels * 3) / (1024 * 1024)  # RGB 기준
        }
    
    def _calculate_optimal_dpi(self, width: int, height: int) -> tuple:
        """
        이미지 크기에 따른 최적 DPI 계산
        """
        total_pixels = width * height
        
        if total_pixels > 2048 * 2048:
            return (150, 250)  # 큰 이미지: 낮은 DPI
        elif total_pixels > 1024 * 1024:
            return (200, 350)  # 중간 이미지: 중간 DPI
        else:
            return (300, 600)  # 작은 이미지: 높은 DPI
    
    def visualize_image(self, image_path: str, title: str = ""):
        """
        이미지 시각화 (크기 정보 포함)
        
        Args:
            image_path: 이미지 파일 경로
            title: 제목
        """
        image = cv2.imread(image_path)
        if image is not None:
            # BGR에서 RGB로 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 크기 정보 가져오기
            size_info = self.get_optimal_size_info(image_path)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb)
            
            # 제목에 크기 정보 추가
            title_with_info = f"{title or os.path.basename(image_path)}\n"
            title_with_info += f"크기: {size_info['current_size'][0]}x{size_info['current_size'][1]} "
            title_with_info += f"({size_info['memory_usage_mb']:.1f}MB)"
            
            plt.title(title_with_info)
            plt.axis('off')
            
            # 최적화 권장사항 출력
            if size_info['recommendations']:
                print("🔍 최적화 권장사항:")
                for rec in size_info['recommendations']:
                    print(f"  • {rec}")
            else:
                print("✅ 현재 이미지 크기는 PaddleOCR에 최적화되어 있습니다.")
            
            plt.show()
        else:
            print(f"이미지를 로드할 수 없습니다: {image_path}")