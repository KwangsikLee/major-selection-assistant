from typing import List, Dict, Any, Optional
from langchain.schema import Document
import PyPDF2
import fitz  # pymupdf

# pip install -q PyPDF2 pymupdf

# ========================================
# PDF 처리 클래스
# ========================================

class PDFProcessor:
    """PDF 문서를 읽고 텍스트를 추출하는 클래스"""

    def __init__(self):
        self.documents = []
    def extract_text_pdfLoader(self, pdf_path: str) -> List[Document]:
        """pdfplumber를 사용한 텍스트 추출"""
        pages = []
        try:
            from langchain_community.document_loaders import PyPDFLoader            

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            total_pages = len(docs)
            for page_num in range(total_pages):
                text = docs[page_num]
                page = Document(
                    page_content=text.page_content,
                    metadata={
                        "source": pdf_path,
                        "page_num": page_num + 1,
                        "total_pages": total_pages + 1
                    }
                )
                pages.append(page)
        except Exception as e:
            print(f"pdfplumber 오류: {e}")
        return pages
    

    def extract_text_pypdf2(self, pdf_path: str) -> List[Document]:
        """PyPDF2를 사용한 텍스트 추출"""
        pages = []
        try:
            with open(pdf_path, 'rb') as file:                
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                for page_num in range(total_pages):
                    text = pdf_reader.pages[page_num]
                    page = Document(
                        page_content=text.extract_text(),
                        metadata={
                            "source": pdf_path,
                            "page_num": page_num + 1,
                            "total_pages": total_pages + 1
                        }
                    )
                    pages.append(page)
                    
        except Exception as e:
            print(f"PyPDF2 오류: {e}")
        return pages

    def extract_text_pymupdf(self, pdf_path: str) -> List[Document]:
        """PyMuPDF를 사용한 텍스트 추출 (더 정확함)"""
        pages = []
        try:
            pdf_document = fitz.open(pdf_path)
            for page_num in range(pdf_document.page_count):
                text = pdf_document.load_page(page_num)
                # text = pdf_document[page_num]
                # print(f"nupdf page({page_num}): {text.get_text()} ")
                page = Document(
                        page_content=text.get_text(),
                        metadata={
                            "source": pdf_path,
                            "page_num": page_num + 1,
                            "total_pages": pdf_document.page_count + 1
                        }
                    )
                pages.append(page)
            pdf_document.close()
        except Exception as e:
            print(f"PyMuPDF 오류: {e}")
        return pages

    def process_pdf(self, pdf_path: str, method='pymupdf') -> List[Document]:
        """PDF를 처리하여 Document 객체 리스트로 변환"""
        if method == 'pymupdf':
            documents = self.extract_text_pymupdf(pdf_path)
            # documents = self.extract_text_pdfLoader(pdf_path=pdf_path)
        else:
            documents = self.extract_text_pypdf2(pdf_path)

        self.documents = documents
        return documents