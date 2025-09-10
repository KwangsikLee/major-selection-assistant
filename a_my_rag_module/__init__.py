"""
a_my_rag_module - RAG (Retrieval-Augmented Generation) 모듈

이 패키지는 RAG 시스템 구현을 위한 핵심 컴포넌트들을 제공합니다:
- TextSplitter: 텍스트 청킹 및 분할
- Embedding: 텍스트 임베딩 생성 및 관리
- Retriever: 문서 검색 및 RAG 파이프라인

Author: kwangsiklee
"""

__version__ = "0.1.0"
__author__ = "kwangsiklee"

# 주요 클래스들을 패키지 레벨에서 임포트하여 사용 편의성 제공
from .splitter import KoreanTextSplitter
from .embedding import MultiEmbeddingManager, VectorStoreManager
from .retriever import MyReranker, AdvancedHybridRetriever
from .pdf_loader import PDFProcessor
from .ocr_korean import KoreanOCR
from .pdf_image import PDFImageExtractor


__all__ = [
    'PDFProcessor',
    'KoreanTextSplitter',
    'MultiEmbeddingManager',
    'VectorStoreManager', 
    'MyReranker',
    'AdvancedHybridRetriever',
    'KoreanOCR',
    'PDFImageExtractor'
]