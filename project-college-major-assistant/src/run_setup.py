#!/usr/bin/env python3
"""
고등학생 학과 선택 도우미 - 설정 및 테스트 스크립트
환경 설정부터 RAG 시스템 테스트까지 전체 과정을 실행

Author: kwangsiklee
Version: 0.1.0
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """환경 설정 확인"""
    print("🔍 환경 설정 확인 중...")
    
    # 1. Python 버전 확인
    python_version = sys.version_info
    print(f"Python 버전: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        return False
    
    # 2. .env 파일 확인
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env 파일이 없습니다. .env.example을 참고하여 생성하세요.")
        return False
    
    # 3. 환경 변수 로드
    load_dotenv()
    
    # 4. API 키 확인
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return False
    
    print("✅ 환경 설정 확인 완료")
    return True

def install_dependencies():
    """의존성 패키지 설치"""
    print("📦 의존성 패키지 설치 중...")
    
    try:
        # requirements.txt 설치
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✅ 의존성 패키지 설치 완료")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 실패: {e}")
        return False

def check_directories():
    """필요한 디렉토리 확인 및 생성"""
    print("📁 디렉토리 구조 확인 중...")
    
    directories = [
        "korea_univ_guides",
        "temp_images", 
        "vector_db"
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"  📂 생성: {dir_name}/")
        else:
            print(f"  ✅ 존재: {dir_name}/")
    
    # PDF 파일 확인
    pdf_dir = Path("korea_univ_guides")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"  📄 PDF 파일: {len(pdf_files)}개")
    
    if len(pdf_files) == 0:
        print("  ⚠️ korea_univ_guides 폴더에 PDF 파일이 없습니다.")
        print("     대학교 안내 PDF 파일을 추가하세요.")
        return False
    
    print("✅ 디렉토리 구조 확인 완료")
    return True

def test_modules():
    """모듈 import 테스트"""
    print("🧪 모듈 import 테스트 중...")
    
    try:
        # 필수 모듈들 import 테스트
        import gradio
        print(f"  ✅ Gradio {gradio.__version__}")
        
        import openai
        print(f"  ✅ OpenAI")
        
        import langchain
        print(f"  ✅ LangChain")
        
        # a_my_rag_module 모듈 테스트
        # 현재 파일: /project-college-major-assistant/src/run_setup.py  
        # 목표 경로: /AI-Study/a_my_rag_module
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from a_my_rag_module import PDFImageExtractor, KoreanOCR
        print(f"  ✅ a_my_rag_module")
        
        print("✅ 모듈 import 테스트 완료")
        return True
        
    except ImportError as e:
        print(f"❌ 모듈 import 실패: {e}")
        return False

def test_rag_system():
    """RAG 시스템 기본 테스트"""
    print("🤖 RAG 시스템 테스트 중...")
    
    try:
        from college_rag_system import CollegeRAGSystem
        
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
        
        # 벡터 스토어 상태 확인
        info = rag_system.get_vector_store_info()
        print(f"  📊 벡터 스토어 존재: {info['exists']}")
        
        print("✅ RAG 시스템 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ RAG 시스템 테스트 실패: {e}")
        return False

def initialize_vector_db(force_rebuild=False, from_json=False):
    """벡터 DB 초기화"""
    print("🚀 벡터 DB 초기화 중...")

    try:
        from vector_store_builder import VectorStoreBuilder

        print(f"  🔄 강제 재구축: {'예' if force_rebuild else '아니오'}")
        print(f"  📄 JSON에서 구축: {'예' if from_json else '아니오'}")

        # VectorStoreBuilder 인스턴스 생성
        builder = VectorStoreBuilder(
            pdf_dir="korea_univ_guides",
            temp_images_dir="temp_images",
            vector_db_dir="vector_db"
        )

        # 벡터 DB 초기화 실행
        result = builder.initialize_vector_db(
            force_rebuild=force_rebuild,
            from_json=from_json
        )

        if result is None:
            print("벡터 DB 초기화 실패: 반환값이 없습니다.")
            return False

        success, message = result

        print(f"  결과: {message}")

        if success:
            print("✅ 벡터 DB 초기화 완료")
            return True
        else:
            print("❌ 벡터 DB 초기화 실패")
            return False

    except Exception as e:
        print(f"❌ 벡터 DB 초기화 실패: {e}")
        return False

def main():
    """메인 설정 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="고등학생 학과 선택 도우미 - 환경 설정")
    parser.add_argument("--init-db", action="store_true",
                       help="벡터 DB 초기화만 실행")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="기존 DB 삭제 후 강제 재구축")
    parser.add_argument("--from-json", action="store_true",
                       help="temp_texts 폴더의 documents.json 파일들로부터 벡터 DB 구축 (OCR 단계 건너뛰기)")
    parser.add_argument("--setup-only", action="store_true",
                       help="DB 초기화 없이 환경 설정만 실행")
    
    args = parser.parse_args()
    
    print("🎓 고등학생 학과 선택 도우미 - 환경 설정")
    print("=" * 60)
    
    # DB 초기화만 실행하는 경우
    if args.init_db:
        print("🚀 벡터 DB 초기화 모드")
        print("-" * 40)
        
        # 기본 환경 확인
        if not check_environment():
            print("❌ 환경 설정 문제로 DB 초기화를 중단합니다.")
            return False
        
        if not check_directories():
            print("❌ 디렉토리 구조 문제로 DB 초기화를 중단합니다.")
            return False
            
        # 벡터 DB 초기화 실행
        success = initialize_vector_db(force_rebuild=args.force_rebuild, from_json=args.from_json)
        
        if success:
            print("\n🎉 벡터 DB 초기화가 완료되었습니다!")
            print("이제 'python main.py'를 실행하여 웹 인터페이스를 사용할 수 있습니다.")
        else:
            print("\n❌ 벡터 DB 초기화에 실패했습니다.")
            
        return success
    
    # 일반 설정 과정
    # 단계별 확인
    base_steps = [
        ("환경 설정 확인", check_environment),
        ("디렉토리 구조 확인", check_directories),
        ("의존성 설치", install_dependencies),
        ("모듈 테스트", test_modules),
        ("RAG 시스템 테스트", test_rag_system)
    ]
    
    # DB 초기화 단계 추가 (setup-only가 아닌 경우)
    if not args.setup_only:
        base_steps.append(("벡터 DB 초기화", lambda: initialize_vector_db(force_rebuild=args.force_rebuild, from_json=args.from_json)))
    
    failed_steps = []
    
    for step_name, step_func in base_steps:
        print(f"\n📋 단계: {step_name}")
        print("-" * 40)
        
        if not step_func():
            failed_steps.append(step_name)
            print(f"❌ {step_name} 실패")
        else:
            print(f"✅ {step_name} 성공")
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📋 설정 결과 요약")
    print(f"{'='*60}")
    
    if not failed_steps:
        print("🎉 모든 설정이 완료되었습니다!")
        print("\n다음 단계:")
        if args.setup_only:
            print("1. python run_setup.py --init-db - 벡터 DB 초기화")
            print("2. python college_rag_system.py --init-db - 직접 DB 초기화")
        else:
            print("1. python main.py - Gradio UI 실행")
            print("2. python college_rag_system.py --test - RAG 시스템 테스트")
        return True
    else:
        print(f"❌ 실패한 단계: {', '.join(failed_steps)}")
        print("\n문제를 해결한 후 다시 실행하세요.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)