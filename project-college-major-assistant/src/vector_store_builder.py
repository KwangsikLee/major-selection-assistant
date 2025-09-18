#!/usr/bin/env python3
"""
VectorStoreBuilder - PDF 처리 및 벡터 스토어 구축 전담 클래스

Author: kwangsiklee  
Version: 0.1.0
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import json
from datetime import datetime
import gc

# LangChain imports
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# a_my_rag_module 모듈을 import하기 위한 경로 설정
# 현재 파일: /project-college-major-assistant/src/vector_store_builder.py
# 목표 경로: /AI-Study/a_my_rag_module
sys.path.append(str(Path(__file__).parent.parent.parent))
from a_my_rag_module import PDFImageExtractor, KoreanOCR, VectorStoreManager


class VectorStoreBuilder:
    """PDF 처리 및 벡터 스토어 구축 전담 클래스"""
    
    # 벡터 스토어 설정 상수
    DEFAULT_INDEX_NAME = "college_guide"
    DEFAULT_MODEL_KEY = "embedding-gemma"
    
    def __init__(self, pdf_dir: str, temp_images_dir: str, vector_db_dir: str):
        self.pdf_dir = Path(pdf_dir)
        self.temp_images_dir = Path(temp_images_dir)
        self.vector_db_dir = Path(vector_db_dir)
        
        # temp_texts 디렉토리 추가
        self.temp_texts_dir = Path(temp_images_dir).parent / "temp_texts"
        
        # 디렉토리 생성
        self.temp_images_dir.mkdir(exist_ok=True)
        self.temp_texts_dir.mkdir(exist_ok=True)
        self.vector_db_dir.mkdir(exist_ok=True)
        
        # PDF 처리 구성 요소 초기화
        self.pdf_extractor = PDFImageExtractor(dpi=100, max_size=2048)
        self.ocr = KoreanOCR()
        
        # VectorStoreManager 
        self.vector_manager = None

        # 텍스트 분할기
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # 벡터 스토어 (구축 시에만 사용)
        self.vector_store = None
        
        print(f"VectorStoreBuilder 초기화 완료")
        print(f"PDF 디렉토리: {self.pdf_dir}")
        print(f"벡터 DB 디렉토리: {self.vector_db_dir}")
        
    def force_memory_cleanup(self):
        """강제 메모리 정리"""
        gc.collect()
        
        # GPU 메모리 정리 (PyTorch)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
    
    def initialize_vector_manager(self):
        """Vector Manager 지연 초기화"""
        if self.vector_manager ==  None:        
            # 환경변수에서 HF API 토큰 가져오기 (있다면)
            hf_token = os.getenv('HF_API_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
            
            # VectorStoreManager 초기화 (한국어 특화 모델 사용)
            self.vector_manager = VectorStoreManager(
                embedding_model_key=self.DEFAULT_MODEL_KEY, 
                save_directory=str(self.vector_db_dir),
                hf_api_token=hf_token if hf_token is not None else ""
            )
    
    def vector_store_exists(self) -> bool:
        """벡터 스토어가 이미 존재하는지 확인"""
        self.initialize_vector_manager()
        if self.vector_manager:
            # VectorStoreManager의 index_exists 메서드 사용
            return self.vector_manager.index_exists(self.DEFAULT_INDEX_NAME, self.DEFAULT_MODEL_KEY)
        else:
            return False
    
    def build_vector_store(self, progress_callback: Optional[Callable] = None):
        """PDF 파일들을 처리하여 벡터 스토어 구축 - 2단계 접근법"""
        try:
            print("\n🔄 벡터 스토어 구축 시작 (2단계 접근법)")
            print("=" * 60)
            
            if progress_callback:
                progress_callback("PDF 파일 목록 확인 중...")
            
            # PDF 파일 목록 가져오기
            pdf_files = list(self.pdf_dir.glob("*.pdf"))
            
            if not pdf_files:
                raise ValueError(f"PDF 파일이 없습니다: {self.pdf_dir}")
            
            print(f"📄 발견된 PDF 파일: {len(pdf_files)}개")
            
            # Phase 1: OCR 처리 및 Document 저장
            if progress_callback:
                progress_callback("Phase 1: OCR 처리 및 문서 저장 중...")
            
            processed_pdfs = self._phase1_ocr_processing(pdf_files, progress_callback)
            
            if not processed_pdfs:
                raise ValueError("OCR 처리된 PDF가 없습니다.")
            
            # Phase 2: 배치 벡터 스토어 추가
            if progress_callback:
                progress_callback("Phase 2: 벡터 스토어 배치 처리 중...")
                
            success = self._phase2_batch_vector_addition(processed_pdfs, progress_callback)
            
            if not success:
                raise ValueError("벡터 스토어 배치 처리 실패")
            
            print("\n✅ 2단계 벡터 스토어 구축 완료")
            return True
            
        except Exception as e:
            print(f"❌ 벡터 스토어 구축 실패: {e}")
            return False
    
    def _phase1_ocr_processing(self, pdf_files: List[Path], progress_callback: Optional[Callable] = None) -> List[str]:
        """
        Phase 1: PDF → Image → OCR → Document 저장
        
        Args:
            pdf_files: 처리할 PDF 파일 리스트
            progress_callback: 진행상황 콜백 함수
            
        Returns:
            List[str]: 성공적으로 처리된 PDF 파일명 리스트
        """
        print("\n📋 Phase 1: OCR 처리 및 문서 저장")
        print("-" * 40)
        
        processed_pdfs = []
        
        # 샘플로 처음 5개 파일만 처리 (MVP)
        sample_files = pdf_files[:5]
        # sample_files = [self.pdf_dir / "01-경영대학.pdf"]
        
        for i, pdf_file in enumerate(sample_files):
            try:
                if progress_callback:
                    progress_callback(f"처리 중: {pdf_file.name} ({i+1}/{len(sample_files)})")
                
                print(f"\n📄 처리 중: {pdf_file.name}")
                
                # PDF 파일명 (확장자 제외)
                pdf_filename = pdf_file.stem
                
                # PDF별 텍스트 폴더 생성
                pdf_text_dir = self.temp_texts_dir / pdf_filename
                pdf_text_dir.mkdir(exist_ok=True)
                
                # 1. PDF에서 이미지 추출
                image_paths = self.pdf_extractor.extract_images_from_pdf(
                    str(pdf_file),
                    str(self.temp_images_dir),
                    split_large_pages=True
                )
                
                print(f"  📷 추출된 이미지: {len(image_paths)}개")
                
                check_memory = True # = self.check_memory_threshold()
                if check_memory:
                    print("⚠️ 메모리 임계값 초과 - 강제 정리 및 대기")
                    self.force_memory_cleanup()
                    import time
                    time.sleep(2)  # 메모리 안정화 대기
                
                # 2. OCR로 텍스트 추출 및 저장
                pdf_texts = []
                for page_idx, img_path in enumerate(image_paths):
                    try:
                        text = self.ocr.extract_text(img_path)
                        if text.strip():  # 빈 텍스트가 아닌 경우만
                            clean_text = text.strip()
                            pdf_texts.append(clean_text)                                
                    except Exception as e:
                        print(f"    ⚠️ OCR 실패: {img_path} - {e}")
                        continue
                    
                # 3. 텍스트를 Document 객체로 변환하고 저장
                pdf_documents = []
                for j, text in enumerate(pdf_texts):
                    if len(text) > 50:  # 너무 짧은 텍스트 제외
                        # 텍스트 분할
                        chunks = self.text_splitter.split_text(text)
                    else:
                        chunks = [text]    
                        
                    for k, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": pdf_file.name,
                                "page": j + 1,
                                "chunk": k + 1,
                                "processed_at": datetime.now().isoformat()
                            }
                        )
                        pdf_documents.append(doc)
                
                # 4. PDF별 Document 리스트를 파일로 저장
                if pdf_documents:
                    self._save_pdf_documents(pdf_filename, pdf_documents)
                    processed_pdfs.append(pdf_filename)
                    print(f"  ✅ 완료: {pdf_file.name} - {len(pdf_documents)}개 문서 저장")
                else:
                    print(f"  ⚠️ 처리할 문서 없음: {pdf_file.name}")
                    
            except Exception as e:
                print(f"  ❌ 오류: {pdf_file.name} - {e}")
                continue
            
        # Phase 1 정리
        self.ocr.cleanup_ocr_model()
        
        if not processed_pdfs:
            print("⚠️ OCR 처리된 PDF가 없습니다.")
            return []
        
        print(f"\n📊 Phase 1 완료: {len(processed_pdfs)}개 PDF 처리")
        for pdf_name in processed_pdfs:
            print(f"  ✓ {pdf_name}")
        
        return processed_pdfs
    
    def _save_pdf_documents(self, pdf_filename: str, documents: List[Document]):
        """PDF별 Document 리스트를 JSON 파일로 저장"""
        documents_dir = self.temp_texts_dir / pdf_filename / "documents"
        documents_dir.mkdir(exist_ok=True)
        
        # Document 객체를 직렬화 가능한 딕셔너리로 변환
        doc_data = []
        for doc in documents:
            doc_data.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # JSON 파일로 저장
        json_path = documents_dir / f"{pdf_filename}_documents.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)
        
        print(f"    💾 Document 저장: {json_path}")
    
    def _phase2_batch_vector_addition(self, processed_pdfs: List[str], progress_callback: Optional[Callable] = None) -> bool:
        """
        Phase 2: 저장된 Document들을 배치로 벡터 스토어에 추가
        
        Args:
            processed_pdfs: 처리된 PDF 파일명 리스트
            progress_callback: 진행상황 콜백 함수
            
        Returns:
            bool: 성공 여부
        """
        print("\n🔄 Phase 2: 벡터 스토어 배치 처리")
        print("-" * 40)
        
        try:
            all_documents = []
            
            # 저장된 Document 파일들 로드
            for i, pdf_filename in enumerate(processed_pdfs):
                if progress_callback:
                    progress_callback(f"Document 로드 중: {pdf_filename} ({i+1}/{len(processed_pdfs)})")
                
                documents = self._load_pdf_documents(pdf_filename)
                if documents:
                    all_documents.extend(documents)
                    print(f"  📄 {pdf_filename}: {len(documents)}개 문서 로드")
            
            if not all_documents:
                print("⚠️ 로드된 문서가 없습니다.")
                return False
            
            print(f"\n📊 총 로드된 문서 수: {len(all_documents)}개")
            
            if progress_callback:
                progress_callback(f"벡터 임베딩 생성 중... ({len(all_documents)}개 문서)")

            # VectorStoreManager를 사용하여 벡터 스토어 생성
            try:
                self.initialize_vector_manager()
                
                print(f"   🤖 VectorStoreManager를 사용하여 벡터 스토어 생성 중...")
                
                # 5. 벡터 스토어 생성 및 자동 저장
                # Ensure vector_manager is initialized before use
                self.initialize_vector_manager()
                if self.vector_manager is not None:
                    self.vector_store = self.vector_manager.auto_save_after_creation(
                        documents=all_documents,
                        index_name=self.DEFAULT_INDEX_NAME,
                        model_key=self.DEFAULT_MODEL_KEY
                    )
                else:
                    raise RuntimeError("VectorStoreManager is not initialized.")
                
                print(f"   ✅ VectorStoreManager로 벡터 스토어 생성 완료")
                
            except Exception as vector_manager_error:
                print(f"   ⚠️ VectorStoreManager 실패: {vector_manager_error}")                

            
            # 6. 메타데이터 저장
            metadata = {
                "created_at": datetime.now().isoformat(),
                "total_documents": len(all_documents),
                "processed_pdfs": processed_pdfs,
                "vector_db_path": str(self.vector_db_dir)
            }
            
            metadata_path = self.vector_db_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            if progress_callback:
                progress_callback(f"벡터 스토어 구축 완료! (문서 {len(all_documents)}개)")
            
            print(f"✅ Phase 2 완료: 벡터 스토어 배치 처리 성공!")
            print(f"   저장 위치: {self.vector_db_dir}")
            print(f"   총 문서: {len(all_documents)}개")
            
            return True
            
        except Exception as e:
            error_msg = f"Phase 2 실패: {e}"
            print(f"❌ {error_msg}")
            if progress_callback:
                progress_callback(error_msg)
            return False
    
    def _load_pdf_documents(self, pdf_filename: str) -> List[Document]:
        """저장된 PDF Document 파일을 로드"""
        try:
            json_path = self.temp_texts_dir / pdf_filename / "documents" / f"{pdf_filename}_documents.json"
            
            if not json_path.exists():
                print(f"⚠️ Document 파일 없음: {json_path}")
                return []
            
            with open(json_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # 딕셔너리를 Document 객체로 변환
            documents = []
            for data in doc_data:
                doc = Document(
                    page_content=data["page_content"],
                    metadata=data["metadata"]
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ Document 로드 실패 ({pdf_filename}): {e}")
            return []
    
    def initialize_vector_db(self, force_rebuild: bool = False, progress_callback: Optional[Callable] = None):
        """벡터 DB 초기화 - 독립 실행 가능한 초기화 함수"""
        try:
            if progress_callback:
                progress_callback("벡터 DB 초기화 시작...")
            
            print("🔄 벡터 DB 초기화 시작...")
            
            # 기존 벡터 DB 확인
            if self.vector_store_exists() and not force_rebuild:
                if progress_callback:
                    progress_callback("기존 벡터 DB 발견 - 검증 중...")
                
                print("📁 기존 벡터 DB 발견 - 검증 시도...")
                try:
                    # VectorStoreManager를 먼저 시도
                    self.initialize_vector_manager()
                    
                    if self.vector_manager is not None:
                        # VectorStoreManager로 저장된 인덱스 로드 시도
                        result, message = self.vector_manager.load_vector_store(self.DEFAULT_INDEX_NAME, self.DEFAULT_MODEL_KEY)
                        
                        if result:
                            self.vector_store = self.vector_manager.current_vector_store
                            
                            if progress_callback:
                                progress_callback("VectorStoreManager 벡터 DB 검증 완료!")
                            
                            print("VectorStoreManager 벡터 DB 검증 완료!")
                            return True, "VectorStoreManager 벡터 DB 검증 완료."
                        else:
                            print(f"     VectorStoreManager 로드 실패: {message}")
                        
                    raise ValueError("벡터 스토어가 로드되지 않았습니다.")
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"기존 DB 검증 실패 - 새로 구축: {e}")
                    
                    print(f"⚠️ 기존 DB 검증 실패 - 새로 구축합니다: {e}")
                    force_rebuild = True
            
            # 새 벡터 DB 구축 또는 강제 재구축
            if not self.vector_store_exists() or force_rebuild:
                if force_rebuild:
                    if progress_callback:
                        progress_callback("기존 벡터 DB 삭제 후 새로 구축...")
                    
                    print("🗑️ 기존 벡터 DB 삭제 후 새로 구축...")
                    # VectorStoreManager의 delete_saved_index 메서드 사용
                    if self.vector_manager:
                        delete_result = self.vector_manager.delete_saved_index(self.DEFAULT_INDEX_NAME, self.DEFAULT_MODEL_KEY)
                        print(f"   {delete_result}")
                    else:
                        # vector_manager가 없는 경우에만 수동 삭제
                        import shutil
                        if self.vector_db_dir.exists():
                            shutil.rmtree(self.vector_db_dir)
                            self.vector_db_dir.mkdir(exist_ok=True)
                
                if progress_callback:
                    progress_callback("새 벡터 DB 구축 시작...")
                
                print("🏗️ 새 벡터 DB 구축 시작...")
                self.build_vector_store(progress_callback)
                
                if progress_callback:
                    progress_callback("✅ 새 벡터 DB 구축 완료!")
                
                print("✅ 새 벡터 DB 구축 완료!")
                return True, "새 벡터 DB를 성공적으로 구축했습니다."
            
        except Exception as e:
            error_msg = f"벡터 DB 초기화 실패: {e}"
            print(f"❌ {error_msg}")
            
            if progress_callback:
                progress_callback(f"❌ {error_msg}")
            
            return False, error_msg
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """벡터 스토어 정보 반환"""
        info = {
            "exists": self.vector_store_exists(),
            "initialized": self.vector_store is not None,
            "vector_db_path": str(self.vector_db_dir)
        }
        
        # 메타데이터 정보 추가
        metadata_path = self.vector_db_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                info.update(metadata)
            except Exception as e:
                info["metadata_error"] = str(e)
        
        return info