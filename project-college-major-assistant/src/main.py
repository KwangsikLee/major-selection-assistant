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
                """
        return info

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    assistant = CollegeMajorAssistant()

    def parse_pdf_files():
        """PDF 파일들을 분석하여 단과대별 학과 데이터 구성"""
        import os
        from pathlib import Path

        pdf_dir = assistant.pdf_dir
        college_departments = {}
        department_pdf_map = {}  # 학과명 -> PDF 파일 경로 매핑

        if not pdf_dir.exists():
            return {}, {}

        pdf_files = list(pdf_dir.glob("*.pdf"))

        # PDF 파일명 분석
        for pdf_file in pdf_files:
            filename = pdf_file.stem

            # 숫자로 시작하는 파일들만 처리 (단과대 구조)
            if filename.split('-')[0].isdigit():
                parts = filename.split('-')
                if len(parts) >= 2:
                    college_name = parts[1]

                    # 학과명 추출
                    if len(parts) >= 3:
                        # 개별 학과 파일 (예: 02-문과대학-01.국어국문.pdf)
                        department_name = parts[2].split('.', 1)[-1] if '.' in parts[2] else parts[2]
                        department_name += "과" if not department_name.endswith(("과", "학과", "학부")) else ""

                        if college_name not in college_departments:
                            college_departments[college_name] = []
                        college_departments[college_name].append(department_name)
                        department_pdf_map[department_name] = str(pdf_file)
                    else:
                        # 단과대 전체 파일 (예: 01-경영대학.pdf, 05-의과대학.pdf)
                        if college_name not in college_departments:
                            college_departments[college_name] = [f"{college_name} 전체"]
                        department_pdf_map[f"{college_name} 전체"] = str(pdf_file)
            else:
                # 특별 학부들 (예: 스마트모빌리티학부, 디자인조형학부 등)
                special_dept = filename.split('_')[0] if '_' in filename else filename
                if "특별학부" not in college_departments:
                    college_departments["특별학부"] = []
                college_departments["특별학부"].append(special_dept)
                department_pdf_map[special_dept] = str(pdf_file)

        # 학과명 정렬
        for college in college_departments:
            college_departments[college] = sorted(college_departments[college])

        return college_departments, department_pdf_map

    # PDF 파일들로부터 실제 데이터 구성
    college_departments, department_pdf_map = parse_pdf_files()

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
                gr.update(interactive=True)          # question_input
            )
        else:
            return (
                f"{final_status}\n❌ {message}",     # init_status
                gr.update(interactive=False)         # question_input
            )

    def auto_init_on_load():
        """인터페이스 로드 시 자동으로 시스템 초기화 실행"""
        return init_system_with_progress()

    def update_departments(college):
        """선택된 단과대에 따라 학과 목록 업데이트"""
        if college in college_departments:
            return gr.update(choices=college_departments[college], value=None)
        return gr.update(choices=[], value=None)

    def generate_auto_question(department):
        """선택된 학과에 대한 자동 질문 생성"""
        if not department:
            return "", gr.update()

        auto_questions = [
            f"{department}에서는 어떤 공부를 하나요?",
            f"{department} 졸업 후 진로는 어떻게 되나요?",
            f"{department} 입학을 위해 어떤 준비가 필요한가요?",
            f"{department}의 취업 전망은 어떤가요?"
        ]

        question = auto_questions[0]  # 기본 질문
        return question, gr.update(value=question)

    def convert_pdf_to_images(pdf_path):
        """PDF를 이미지로 변환"""
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import io
            import tempfile
            doc = fitz.open(pdf_path)
            images = []

            # 최대 5페이지만 변환 (미리보기용)
            max_pages = min(5, len(doc))
            for page_num in range(max_pages):
                page = doc[page_num]
                # 해상도 조정 (1.5배 확대)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_data = pix.tobytes("png")

                # PIL Image로 변환하여 임시 파일로 저장
                img = Image.open(io.BytesIO(img_data))

                # 임시 파일로 저장 (Gradio Gallery가 파일 경로를 요구함)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    img.save(tmp_file.name, 'PNG')
                    images.append(tmp_file.name)

            doc.close()
            return images
        except Exception as e:
            print(f"PDF to image conversion error: {e}")
            return []

    def show_department_info(department):
        """선택된 학과의 PDF 정보 표시"""
        if not department:
            return (
                [["시스템", "학과를 선택해주세요."]],
                gr.update(value=[], visible=False),
                gr.update(value=None, visible=False)
            )

        # PDF 파일 경로 확인
        if department in department_pdf_map:
            pdf_path = department_pdf_map[department]
            pdf_name = Path(pdf_path).name

            # PDF 파일이 존재하는지 확인
            if Path(pdf_path).exists():
                # RAG 시스템을 통해 해당 학과의 정보 검색
                try:
                    # 해당 학과에 특화된 질문으로 검색
                    search_query = f"{department} 학과 소개 교육과정 취업 진로"
                    result = assistant.rag_system.query(search_query) if assistant.rag_system else None

                    # PDF를 이미지로 변환
                    pdf_images = convert_pdf_to_images(pdf_path)

                    if result and result.get('answer'):
                        info_message = f"📖 **{department} 정보** (출처: {pdf_name})\n\n"
                        info_message += result['answer']

                        # 참고 자료 추가
                        if result.get('sources'):
                            info_message += "\n\n📚 **참고 자료:**\n"
                            for i, source in enumerate(result['sources'], 1):
                                info_message += f"{i}. {source}\n"

                        return (
                            [[f"{department} 정보 요청", info_message]],
                            gr.update(value=pdf_images, visible=len(pdf_images) > 0),
                            gr.update(value=pdf_path, visible=True)
                        )
                    else:
                        return (
                            [[f"{department} 정보", f"📖 {department}의 PDF 파일을 찾았습니다: {pdf_name}\n\n시스템 초기화가 완료되면 상세 정보를 제공할 수 있습니다."]],
                            gr.update(value=pdf_images, visible=len(pdf_images) > 0),
                            gr.update(value=pdf_path, visible=True)
                        )

                except Exception as e:
                    return (
                        [[f"{department} 정보", f"📖 {department}의 PDF 파일: {pdf_name}\n\n정보 검색 중 오류가 발생했습니다: {str(e)}"]],
                        gr.update(value=[], visible=False),
                        gr.update(value=None, visible=False)
                    )
            else:
                return (
                    [[f"{department} 정보", f"❌ {department}의 PDF 파일을 찾을 수 없습니다."]],
                    gr.update(value=[], visible=False),
                    gr.update(value=None, visible=False)
                )
        else:
            return (
                [[f"{department} 정보", f"❌ {department}에 해당하는 PDF 파일을 찾을 수 없습니다."]],
                gr.update(value=[], visible=False),
                gr.update(value=None, visible=False)
            )
    
    def process_question(question, history):
        """질문 처리 및 채팅 히스토리 업데이트"""
        # 빈 질문은 처리하지 않음
        if not question.strip():
            return history, gr.update(value="", interactive=True)

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
        
        system_info = gr.Markdown(assistant.get_system_info())        
        with gr.Row():
            with gr.Column(scale=1):
                # 단과대 및 학과 선택
                gr.Markdown("### 📚 단과대별 학과 선택")

                college_dropdown = gr.Dropdown(
                    label="단과대 선택",
                    choices=list(college_departments.keys()) if college_departments else ["데이터 로딩 중..."],
                    value=None,
                    interactive=True
                )

                department_dropdown = gr.Dropdown(
                    label="학과 선택",
                    choices=[],
                    value=None,
                    interactive=True
                )

                with gr.Row():
                    dept_info_btn = gr.Button("📖 학과정보 바로보기", size="sm")
                    auto_question_btn = gr.Button("❓ 자동질문하기", size="sm")

                # PDF 미리보기 및 다운로드
                pdf_gallery = gr.Gallery(
                    label="📄 PDF 미리보기",
                    columns=1,
                    rows=2,
                    height=400,
                    visible=False
                )

                pdf_download = gr.File(
                    label="📁 PDF 다운로드",
                    visible=False
                )

            with gr.Column(scale=2):
                # 채팅 인터페이스
                chatbot = gr.Chatbot(
                    label="💬 AI 상담사와 대화",
                    height=400,
                    show_copy_button=True
                )
                
                question_input = gr.Textbox(
                    label="질문 입력 (Enter 키로 전송)",
                    placeholder="전공, 진로, 대학생활에 대해 궁금한 점을 질문하세요...",
                    interactive=True
                )
                
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

        # 페이지 하단 상태 표시
        init_status = gr.Textbox(
            label="상태",
            lines=1,
            interactive=False,
            value="🚀 시스템 자동 초기화를 시작합니다...",
            placeholder="시스템 상태가 여기에 표시됩니다."
        )

        # 인터페이스 로드 시 자동 초기화
        interface.load(
            fn=auto_init_on_load,
            outputs=[init_status, question_input]
        )

        # 단과대 선택 시 학과 목록 업데이트
        college_dropdown.change(
            fn=update_departments,
            inputs=[college_dropdown],
            outputs=[department_dropdown]
        )

        # 자동 질문하기 버튼
        auto_question_btn.click(
            fn=generate_auto_question,
            inputs=[department_dropdown],
            outputs=[question_input, question_input]
        )

        # 학과 정보 바로보기 버튼
        dept_info_btn.click(
            fn=show_department_info,
            inputs=[department_dropdown],
            outputs=[chatbot, pdf_gallery, pdf_download]
        )

        # 질문 처리 (Enter 키)
        question_input.submit(
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