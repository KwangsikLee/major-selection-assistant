#!/usr/bin/env python3
"""
대학 학과 선택 도우미 RAG 시스템 구현
PDF 이미지 추출 → OCR → 벡터 임베딩 → 검색 → LLM 답변 생성

Author: kwangsiklee  
Version: 0.1.0
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable

# 분리된 클래스들 import
from vector_store_builder import VectorStoreBuilder
from college_qa_system import CollegeQASystem


class CollegeRAGSystem:
    """통합 인터페이스 - 기존 코드 호환성을 위한 wrapper 클래스"""
    
    def __init__(self, pdf_dir: str, temp_images_dir: str, vector_db_dir: str):
        self.pdf_dir = Path(pdf_dir)
        self.temp_images_dir = Path(temp_images_dir)
        self.vector_db_dir = Path(vector_db_dir)
        
        # 구성 요소 초기화
        self.builder = VectorStoreBuilder(pdf_dir, temp_images_dir, vector_db_dir)
        self.qa_system = CollegeQASystem(vector_db_dir)
        
        print(f"CollegeRAGSystem 초기화 완료 (통합 인터페이스)")
        print(f"PDF 디렉토리: {self.pdf_dir}")
        print(f"벡터 DB 디렉토리: {self.vector_db_dir}")
    
    # VectorStoreBuilder 메서드들을 위임
    def vector_store_exists(self) -> bool:
        return self.builder.vector_store_exists()
    
    def build_vector_store(self, progress_callback: Optional[Callable] = None):
        return self.builder.build_vector_store(progress_callback)
    
    def initialize_vector_db(self, force_rebuild: bool = False, progress_callback: Optional[Callable] = None):
        result = self.builder.initialize_vector_db(force_rebuild, progress_callback)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        else:
            # 기본 실패 메시지 반환
            return False, "벡터 DB 초기화 결과가 올바르지 않습니다."
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        return self.builder.get_vector_store_info()
    
    # CollegeQASystem 메서드들을 위임
    def load_vector_store(self):
        return self.qa_system.load_vector_store()
    
    def setup_qa_chain(self):
        return self.qa_system.setup_qa_chain()
    
    def query(self, question: str) -> Dict[str, Any]:
        return self.qa_system.query(question)


# 독립 실행 함수들
def initialize_database(force_rebuild: bool = False):
    """독립 실행 가능한 벡터 DB 초기화 함수"""
    print("🚀 벡터 DB 초기화 시작")
    print("=" * 50)
    
    # 경로 설정 - 프로젝트 루트 디렉토리 기준
    project_root = Path(__file__).parent.parent
    pdf_dir = project_root / "korea_univ_guides"
    temp_images_dir = project_root / "temp_images"
    vector_db_dir = project_root / "vector_db"
    
    # PDF 파일 확인
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ PDF 파일이 없습니다: {pdf_dir}")
        print("   korea_univ_guides/ 폴더에 대학교 안내 PDF 파일을 추가하세요.")
        return False
    
    print(f"📄 발견된 PDF 파일: {len(pdf_files)}개")
    
    # RAG 시스템 초기화
    try:
        rag_system = CollegeRAGSystem(
            pdf_dir=str(pdf_dir),
            temp_images_dir=str(temp_images_dir),
            vector_db_dir=str(vector_db_dir)
        )
        
        # 벡터 DB 초기화 실행
        success, message = rag_system.initialize_vector_db(
            force_rebuild=force_rebuild
        )
        
        if success:
            print(f"\n✅ 성공: {message}")
            
            # 벡터 스토어 정보 출력
            info = rag_system.get_vector_store_info()
            print(f"\n📊 벡터 DB 정보:")
            print(f"   경로: {info['vector_db_path']}")
            print(f"   문서 수: {info.get('total_documents', 'Unknown')}개")
            processed_pdfs = info.get('processed_pdfs', ['Unknown'])
            if isinstance(processed_pdfs, list):
                print(f"   처리된 PDF: {', '.join(processed_pdfs)}")
            else:
                print(f"   처리된 PDF: {processed_pdfs}")
            print(f"   생성일: {info.get('created_at', 'Unknown')}")
            
            return True
        else:
            print(f"\n❌ 실패: {message}")
            return False
            
    except Exception as e:
        print(f"\n❌ 초기화 중 오류 발생: {e}")
        return False

def test_rag_system():
    """RAG 시스템 테스트"""
    print("🧪 RAG 시스템 테스트 시작")
    
    # 경로 설정 - 프로젝트 루트 디렉토리 기준
    project_root = Path(__file__).parent.parent
    pdf_dir = project_root / "korea_univ_guides"
    temp_images_dir = project_root / "temp_images"  
    vector_db_dir = project_root / "vector_db"
    
    # RAG 시스템 초기화
    rag_system = CollegeRAGSystem(
        pdf_dir=str(pdf_dir),
        temp_images_dir=str(temp_images_dir),
        vector_db_dir=str(vector_db_dir)
    )
    
    # 벡터 스토어 구축 또는 로드
    if not rag_system.vector_store_exists():
        print("새로운 벡터 스토어 구축...")
        rag_system.build_vector_store()
    else:
        print("기존 벡터 스토어 로드...")
        rag_system.load_vector_store()
    
    # 테스트 질문들
    test_questions = [
        "컴퓨터공학과는 어떤 공부를 하나요?",
        "경영학과의 주요 과목은 무엇인가요?",
        "의대 진학을 위해 어떤 준비가 필요한가요?"
    ]
    
    print("\n🤖 질문-답변 테스트:")
    for question in test_questions:
        print(f"\n❓ {question}")
        result = rag_system.query(question)
        print(f"💬 {result['answer']}")
        if result['sources']:
            print(f"📚 참고 자료: {', '.join(result['sources'])}")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    # 환경 변수 로드
    load_dotenv()
    
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(description="대학 학과 선택 도우미 RAG 시스템")
    parser.add_argument("--init-db", action="store_true", 
                       help="벡터 DB 초기화")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="기존 DB 삭제 후 강제 재구축")
    parser.add_argument("--test", action="store_true",
                       help="RAG 시스템 테스트 실행")
    
    args = parser.parse_args()
    
    if args.init_db:
        success = initialize_database(force_rebuild=args.force_rebuild)
        if success:
            print("\n🎉 벡터 DB 초기화가 완료되었습니다!")
            print("이제 'python main.py'를 실행하여 웹 인터페이스를 사용할 수 있습니다.")
        sys.exit(0 if success else 1)
    elif args.test:
        test_rag_system()
    else:
        # 기본적으로 테스트 실행
        test_rag_system()