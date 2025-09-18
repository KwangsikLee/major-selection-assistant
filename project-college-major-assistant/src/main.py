#!/usr/bin/env python3
"""
고등학생 학과 선택 도우미 - 메인 애플리케이션
MVP 버전: PDF 이미지 추출 → OCR → 벡터 임베딩 → RAG → Gradio UI

Author: kwangsiklee
Version: 0.1.0
"""

import os
import sys
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

# a_my_rag_module 모듈을 import하기 위한 경로 설정
# 현재 파일: /project-college-major-assistant/src/main.py
# 목표 경로: /AI-Study/a_my_rag_module
sys.path.append(str(Path(__file__).parent.parent.parent))

from a_my_rag_module import PDFImageExtractor, KoreanOCR, MultiEmbeddingManager, VectorStoreManager
from college_rag_system import CollegeRAGSystem

# 환경 변수 로드
load_dotenv()

class CollegeMajorAssistant:
    """고등학생 학과 선택 도우미 메인 클래스"""
    
    def __init__(self):
        self.setup_paths()
        self.rag_system = None
        self.is_initialized = False
        
    def setup_paths(self):
        """경로 설정"""
        # 프로젝트 루트 디렉토리 (src의 상위 디렉토리)
        self.project_root = Path(__file__).parent.parent
        self.pdf_dir = self.project_root / "korea_univ_guides"
        self.temp_images_dir = self.project_root / "temp_images"
        self.vector_db_dir = self.project_root / "vector_db"
        
        # 디렉토리 생성
        self.temp_images_dir.mkdir(exist_ok=True)
        self.vector_db_dir.mkdir(exist_ok=True)
        
    def initialize_system(self, progress_callback=None):
        """시스템 초기화 - PDF 처리부터 RAG 시스템 구축까지"""
        try:
            if progress_callback:
                progress_callback("시스템 초기화 시작...")
            
            # RAG 시스템 초기화
            self.rag_system = CollegeRAGSystem(
                pdf_dir=str(self.pdf_dir),
                temp_images_dir=str(self.temp_images_dir),
                vector_db_dir=str(self.vector_db_dir)
            )
            
            if progress_callback:
                progress_callback("PDF 파일 처리 중...")
            
            # 벡터 DB가 없으면 처음부터 구축
            if not self.rag_system.vector_store_exists():
                self.rag_system.build_vector_store(progress_callback)
            else:
                if progress_callback:
                    progress_callback("기존 벡터 DB 로드 중...")

                self.rag_system.load_vector_store()
            
            self.is_initialized = True
            
            if progress_callback:
                progress_callback("시스템 초기화 완료!")
            
            return True, "시스템이 성공적으로 초기화되었습니다."
            
        except Exception as e:
            error_msg = f"시스템 초기화 중 오류 발생: {str(e)}"
            if progress_callback:
                progress_callback(error_msg)
            return False, error_msg
    
    def ask_question(self, question: str) -> tuple:
        """질문에 답변하기"""

        if not self.rag_system:
            return "❌ 시스템이 초기화되지 않았습니다. 먼저 '시스템 초기화' 버튼을 클릭하세요.", ""
        if not self.is_initialized or not self.rag_system:
            return "❌ 시스템이 초기화되지 않았습니다. 먼저 '시스템 초기화' 버튼을 클릭하세요.", ""
        
        if not question.strip():
            return "❓ 질문을 입력해주세요.", ""
        
        try:
            # RAG 시스템으로 질문 처리
            result = self.rag_system.query(question)
            
            answer = result.get('answer', '답변을 생성할 수 없습니다.')
            sources = result.get('sources', [])
            
            # 참고 자료 정보 생성
            source_info = ""
            if sources:
                source_info = "\n\n📚 **참고 자료:**\n"
                for i, source in enumerate(sources, 1):
                    source_info += f"{i}. {source}\n"
            
            return answer, source_info
            
        except Exception as e:
            error_msg = f"질문 처리 중 오류가 발생했습니다: {str(e)}"
            return error_msg, ""
    
    def get_system_info(self) -> str:
        """시스템 정보 반환"""
        pdf_count = len(list(self.pdf_dir.glob("*.pdf"))) if self.pdf_dir.exists() else 0
        
        info = f"""
                ### 시스템 정보
                - **버전**: v0.1.0 (MVP)

                ### 기능 소개
                1. **대학교 학과 안내 자료 기반 상담**
                2. **AI 기반 질의응답 시스템**
                3. **개인 맞춤형 전공 추천**

                ### 사용법
                1. 시스템이 자동으로 초기화됩니다. (수동 초기화도 가능)
                2. 전공, 진로, 대학생활에 관한 질문을 입력하세요.
                3. AI가 대학 안내 자료를 바탕으로 답변해드립니다.
                """
        return info

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    assistant = CollegeMajorAssistant()

    def init_system_with_progress():
        """진행상황을 표시하면서 시스템 초기화"""
        progress_messages = []

        def progress_callback(message):
            progress_messages.append(message)
            return "\n".join(progress_messages)

        success, message = assistant.initialize_system(progress_callback)
        final_status = "\n".join(progress_messages)

        if success:
            return (
                f"{final_status}\n✅ {message}",    # init_status
                gr.update(interactive=True),         # question_input
                gr.update(interactive=True)          # ask_btn
            )
        else:
            return (
                f"{final_status}\n❌ {message}",     # init_status
                gr.update(interactive=False),        # question_input
                gr.update(interactive=False)         # ask_btn
            )

    def auto_init_on_load():
        """인터페이스 로드 시 자동으로 시스템 초기화 실행"""
        return init_system_with_progress()
    
    def process_question(question, history):
        """질문 처리 및 채팅 히스토리 업데이트"""
        answer, sources = assistant.ask_question(question)
        
        # 전체 응답 생성 (답변 + 참고자료)
        full_response = answer
        if sources.strip():
            full_response += sources
            
        # 히스토리에 추가
        history.append([question, full_response])
        
        return history, gr.update(value="", interactive=True)  # 채팅창 업데이트, 입력창 초기화 및 활성화 유지
    
    # Gradio 인터페이스 구성
    with gr.Blocks(
        title="🎓 고등학생 학과 선택 도우미",
        css="""
        .gradio-container {max-width: 2000px !important}
        .chat-message {padding: 10px; border-radius: 10px; margin: 5px 0;}
        .user-message {background-color: #e3f2fd;}
        .bot-message {background-color: #f5f5f5;}
        """
    ) as interface:
        
        gr.Markdown("# 🎓 고등학생 학과 선택 도우미")
        gr.Markdown("대학교 학과 안내 자료를 바탕으로 한 AI 상담 서비스입니다.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 시스템 정보 및 제어
                system_info = gr.Markdown(assistant.get_system_info())
                
                init_status = gr.Textbox(
                    label="초기화 상태",
                    lines=5,
                    interactive=False,
                    value="🚀 시스템 자동 초기화를 시작합니다...",
                    placeholder="시스템 초기화가 자동으로 실행됩니다."
                )
                
            with gr.Column(scale=2):
                # 채팅 인터페이스
                chatbot = gr.Chatbot(
                    label="💬 AI 상담사와 대화",
                    height=400,
                    show_copy_button=True
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="질문 입력",
                        placeholder="전공, 진로, 대학생활에 대해 궁금한 점을 질문하세요...",
                        lines=2,
                        interactive=True  # 초기에는 비활성화
                    )
                    ask_btn = gr.Button("📝 질문하기", variant="secondary", interactive=False)
                
                gr.Examples(
                    examples=[
                        "컴퓨터공학과는 어떤 공부를 하나요?",
                        "의대 입학을 위해 어떤 준비가 필요한가요?", 
                        "경영학과의 취업 전망은 어떤가요?",
                        "공대와 이과대학의 차이점은 무엇인가요?",
                        "문과생도 프로그래밍을 배울 수 있나요?"
                    ],
                    inputs=[question_input]
                )

        # 인터페이스 로드 시 자동 초기화
        interface.load(
            fn=auto_init_on_load,
            outputs=[init_status, question_input, ask_btn]
        )
        
        # 질문 처리 (Enter 키 또는 버튼 클릭)
        ask_btn.click(
            fn=process_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input]
        )
        
        question_input.submit(  # Enter 키
            fn=process_question,
            inputs=[question_input, chatbot], 
            outputs=[chatbot, question_input]
        )
    
    return interface

def main():
    """메인 함수"""
    print("🎓 고등학생 학과 선택 도우미 시작")
    print("=" * 50)
    
    # API 키 확인
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 설정하세요.")
        sys.exit(1)
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_gradio_interface()
    
    print("🌐 Gradio 서버 시작...")
    print("브라우저에서 http://localhost:7860 으로 접속하세요.")
    interface.launch(
        server_name="0.0.0.0",  # 외부 접속 허용
        server_port=7860,       # 포트 설정
        share=False,            # 공개 링크 생성 안함
        debug=True,           # 디버그 모드
        show_error=True         # 오류 표시
    )

if __name__ == "__main__":
    main()