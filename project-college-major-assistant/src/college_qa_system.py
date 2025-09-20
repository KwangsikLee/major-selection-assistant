#!/usr/bin/env python3
"""
CollegeQASystem - 구축된 벡터 스토어를 사용한 질문답변 전담 클래스

Author: kwangsiklee  
Version: 0.1.0
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# a_my_rag_module 모듈을 import하기 위한 경로 설정
# 현재 파일: /project-college-major-assistant/src/college_qa_system.py
# 목표 경로: /AI-Study/a_my_rag_module
sys.path.append(str(Path(__file__).parent.parent.parent))
from a_my_rag_module import VectorStoreManager
from a_my_rag_module.retriever import HybridRetrieverWrapper


class CollegeQASystem:
    """구축된 벡터 스토어를 사용한 질문답변 전담 클래스"""
    
    # 벡터 스토어 설정 상수
    DEFAULT_INDEX_NAME = "college_guide"
    DEFAULT_MODEL_KEY = "embedding-gemma"
    
    def __init__(self, vector_db_dir: str):
        self.vector_db_dir = Path(vector_db_dir)
        
        # LLM 설정 (필요시 지연 로딩)
        self.llm = None
        
        # 벡터 스토어와 QA 체인
        self.vector_store = None
        self.qa_chain = None
        
        # VectorStoreManager
        self.vector_manager = None
        
        # HybridRetrieverWrapper
        self.hybrid_retriever_wrapper = None
        
        # 프롬프트 템플릿
        self.setup_prompt_template()
        
        print(f"CollegeQASystem 초기화 완료")
        print(f"벡터 DB 디렉토리: {self.vector_db_dir}")
    
    def initialize_llm_components(self):
        """LLM 구성요소 지연 초기화"""
        if self.llm is None:
            print("🤖 LLM 구성요소 초기화 중...")
            
            # OpenAI 설정
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2
            )
    
    def setup_prompt_template(self):
        """프롬프트 템플릿 설정"""
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""당신은 고등학생들의 대학교 전공 선택을 도와주는 전문 상담사입니다.

아래 대학교 학과 안내 자료를 바탕으로 학생의 질문에 정확하고 도움이 되는 답변을 해주세요.

참고 자료:
{context}

질문: {question}

답변 시 다음 사항을 고려해주세요:
1. 고등학생이 이해하기 쉬운 언어로 설명해주세요
2. 구체적이고 실용적인 정보를 제공해주세요  
3. 장황하지 않게 표현해주세요
4. 친근하고 격려하는 톤으로 답변해주세요
5. 답변은 오직 참고자료에 있는 내용에 한해서만 해야 한다.
5. 참고 자료에 없는 내용은 정보가 없어 답변못해 미안하다고 하고 끝낸다
6. 참고 자료에 있는 답변은 근거를 표시해주세요.
답변:"""
        )
    
    def initialize_vector_manager(self):
        """VectorStoreManager 지연 초기화"""
        if self.vector_manager is None:
            import os
            # 환경변수에서 HF API 토큰 가져오기 (있다면)
            hf_token = os.getenv('HF_API_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
            
            # VectorStoreManager 초기화
            self.vector_manager = VectorStoreManager(
                embedding_model_key=self.DEFAULT_MODEL_KEY, 
                save_directory=str(self.vector_db_dir),
                hf_api_token=hf_token if hf_token is not None else ""
            )
    
    def load_vector_store(self):
        """기존 벡터 스토어 로드"""
        try:
            print("기존 벡터 스토어 로드 중...")
            
            # LLM 구성요소 초기화
            self.initialize_llm_components()
            
            # VectorStoreManager 초기화
            self.initialize_vector_manager()
            if self.vector_manager is not None:
                # VectorStoreManager로 벡터 스토어 로드 시도
                result, message = self.vector_manager.load_vector_store(
                    self.DEFAULT_INDEX_NAME, 
                    self.DEFAULT_MODEL_KEY
                )
                
                if result:
                    self.vector_store = self.vector_manager.current_vector_store                    
                    print(f"   {message}")                    
                    # HybridRetrieverWrapper 초기화
                    self.setup_hybrid_retriever()
                
            if not self.vector_store:
                # VectorStoreManager 로드 실패 시 기존 방식으로 시도
                print(f"⚠️ VectorStoreManager 로드 실패: {message}")
                raise ValueError("벡터 스토어가 로드되지 않았습니다.")
            
            # QA 체인 설정
            self.setup_qa_chain()
            
            # 메타데이터 로드
            metadata_path = self.vector_db_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"   생성일: {metadata.get('created_at', 'Unknown')}")
                print(f"   문서 수: {metadata.get('total_documents', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ 벡터 스토어 로드 실패: {e}")
            raise
    
    def setup_hybrid_retriever(self):
        """HybridRetrieverWrapper 설정"""
        try:
            if self.vector_store is not None:
                self.hybrid_retriever_wrapper = HybridRetrieverWrapper(
                    vector_store=self.vector_store,
                    search_method="ensemble"
                )
            else:
                raise ValueError("vector_store가 None입니다.")
        except Exception as e:
            print(f"⚠️ HybridRetrieverWrapper 설정 실패: {e}")
            print("   기본 벡터 검색기를 사용합니다.")
            self.hybrid_retriever_wrapper = None
    
    def setup_qa_chain(self):
        """QA 체인 설정"""
        if self.vector_store is None:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        
        # 리트리버 선택
        retriever = self.hybrid_retriever_wrapper or self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        
        # QA 체인 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
        
        retriever_name = "HybridRetrieverWrapper" if self.hybrid_retriever_wrapper else "기본 벡터 검색기"
        print(f"✅ QA 체인 설정 완료 ({retriever_name})")
    
    def query(self, question: str) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        if self.qa_chain is None:
            raise ValueError("QA 체인이 설정되지 않았습니다. 먼저 벡터 스토어를 로드하세요.")
        
        try:
            print(f"🔍 질문 처리: {question}")
            
            # QA 체인으로 질문 처리
            result = self.qa_chain.invoke({"query": question})
            
            # 답변 추출
            answer = result.get("result", "답변을 생성할 수 없습니다.")
            source_docs = result.get("source_documents", [])
            
            # 검색된 context 로그 출력
            retriever_type = "HybridRetrieverWrapper (Hybrid + Reranking)" if self.hybrid_retriever_wrapper else "기본 벡터 검색기"
            print(f"\n📖 {retriever_type}가 검색한 Context 정보:")
            print("=" * 80)
            for i, doc in enumerate(source_docs):
                metadata = doc.metadata
                content = doc.page_content
                print(f"[Context {i+1}]")
                print(f"출처: {metadata.get('source', 'Unknown')} (페이지 {metadata.get('page', '?')})")
                print(f"내용 길이: {len(content)} 글자")
                print(f"내용 미리보기: {content[:200]}..." if len(content) > 200 else f"전체 내용: {content}")
                print("-" * 40)
            print("=" * 80)
            
            # 소스 정보 생성
            sources = []
            for doc in source_docs:
                metadata = doc.metadata
                source_info = f"{metadata.get('source', 'Unknown')} (페이지 {metadata.get('page', '?')})"
                if source_info not in sources:
                    sources.append(source_info)
            
            print(f"✅ 답변 생성 완료 (참고 자료: {len(sources)}개)")
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "source_documents": source_docs
            }
            
        except Exception as e:
            error_msg = f"질문 처리 중 오류 발생: {e}"
            print(f"❌ {error_msg}")
            
            return {
                "question": question,
                "answer": f"죄송합니다. {error_msg}",
                "sources": [],
                "source_documents": []
            }
    
    def switch_reranker(self, reranker_model: str) -> str:
        """Reranker 모델 변경"""
        if not self.hybrid_retriever_wrapper:
            return "⚠️ HybridRetrieverWrapper가 초기화되지 않았습니다."
        
        return self.hybrid_retriever_wrapper.switch_reranker(reranker_model)
    
    def get_retriever_info(self) -> str:
        """현재 검색기 정보 반환"""
        if not self.hybrid_retriever_wrapper:
            return "📊 현재 검색기: 기본 벡터 검색기"
        
        return self.hybrid_retriever_wrapper.get_retriever_info()