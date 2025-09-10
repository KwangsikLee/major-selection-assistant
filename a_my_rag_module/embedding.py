from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import json
import os
import pickle
# LangChain imports
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from sentence_transformers import SentenceTransformer

# =============================================================================
# 다중 임베딩 및 벡터 DB 관리 클래스 (임베딩 & 벡터 DB화)
# =============================================================================

class MultiEmbeddingManager:
    def __init__(self, api_token):
        # 사용 가능한 임베딩 모델들
        self.embedding_models = {
            "embedding-gemma": {
                "name": "google/embeddinggemma-300m",
                "description": "다국어 지원 SentenceTransformer 모델",
                "language": "Multilingual",
                "size": "~1.2G"
            },
            "ko-sroberta-multitask": {
                "name": "jhgan/ko-sroberta-multitask",
                "description": "한국어 특화 SentenceTransformer 모델",
                "language": "Korean",
                "size": "~300MB"
            },
            "paraphrase-multilingual": {
                "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "description": "다국어 지원 경량 모델",
                "language": "Multilingual",
                "size": "~420MB"
            },
            "ko-sentence-transformer": {
                "name": "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
                "description": "100개 언어 지원 BERT 기반 모델",
                "language": "Multilingual",
                "size": "~1.1GB"
            },
            "distiluse-multilingual": {
                "name": "sentence-transformers/distiluse-base-multilingual-cased-v2",
                "description": "다국어 DistilUSE 모델 (빠름)",
                "language": "Multilingual",
                "size": "~540MB"
            },
            "ko-electra": {
                "name": "bongsoo/kpf-sbert-128d",
                "description": "한국어 ELECTRA 기반 경량 모델",
                "language": "Korean",
                "size": "~50MB"
            }
        }

        self.loaded_embeddings = {}
        self.current_model = None
        self.hf_api_token = api_token
        print(f"embedding manager with key: {self.hf_api_token}")

    def load_embedding_model(self, model_key: str) -> HuggingFaceEmbeddings:
        """임베딩 모델 로드"""
        if model_key in self.loaded_embeddings:
            return self.loaded_embeddings[model_key]

        if model_key not in self.embedding_models:
            raise ValueError(f"Unknown embedding model: {model_key}")

        model_info = self.embedding_models[model_key]
        print(f"Loading embedding model: {model_info['name']} ({model_info['size']})")
        print(f"my hf api token = {self.hf_api_token}")   
        try:

            if model_key == "embedding-gemma":
                
                # # Download from the 🤗 Hub
                # model = SentenceTransformer("google/embeddinggemma-300m")
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_info['name'],
                    model_kwargs={'device': 'cpu', "token": self.hf_api_token}
                )
            else:
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_info['name'],
                    model_kwargs={'device': 'cpu', "token": self.hf_api_token}
                )
            self.loaded_embeddings[model_key] = embeddings
            self.current_model = model_key
            return embeddings
        except Exception as e:
            print(f"Failed to load {model_key}: {e}")
            # fallback to default model
            if model_key != "paraphrase-multilingual":
                return self.load_embedding_model("paraphrase-multilingual")
            raise e

    def get_available_models(self) -> Dict[str, str]:
        """사용 가능한 모델 목록 반환"""
        return {key: f"{info['description']} ({info['language']}, {info['size']})"
                for key, info in self.embedding_models.items()}

    def get_current_model_info(self) -> str:
        """현재 사용 중인 모델 정보 반환"""
        if self.current_model:
            info = self.embedding_models[self.current_model]
            return f"🤖 {info['description']} ({info['language']}, {info['size']})"
        return "모델이 로드되지 않음"

class VectorStoreManager:
    def __init__(self, embedding_model_key: str = "embedding-gemma", save_directory: str = "faiss_indexes", hf_api_token: str = None):
        self.embedding_manager = MultiEmbeddingManager(api_token=hf_api_token)
        
        self.vector_stores = {}  # 모델별 벡터 스토어 저장
        self.current_vector_store = None
        self.current_model_key = embedding_model_key
        self.save_directory = save_directory
        self.embeddings = None
        # 저장 디렉토리 생성
        os.makedirs(self.save_directory, exist_ok=True)

    def lazy_init_embedding(self, embedding_model_key):
        self.embeddings = self.embedding_manager.load_embedding_model(embedding_model_key)

    def switch_embedding_model(self, model_key: str) -> str:
        """임베딩 모델 변경"""
        try:
            print(f"Switching to embedding model: {model_key}")
            self.embeddings = self.embedding_manager.load_embedding_model(model_key)
            self.current_model_key = model_key

            # 기존 벡터 스토어가 있다면 사용, 없다면 새로 생성 필요
            if model_key in self.vector_stores:
                self.current_vector_store = self.vector_stores[model_key]
                return f"✅ 모델 변경 완료: {self.embedding_manager.get_current_model_info()}"
            else:
                return f"✅ 모델 변경 완료 (벡터 스토어 재생성 필요): {self.embedding_manager.get_current_model_info()}"
        except Exception as e:
            return f"❌ 모델 변경 실패: {str(e)}"

    def create_vector_store(self, documents: List[Document], model_key: str = None) -> FAISS:
        """벡터 스토어 생성 - Document 메타데이터 강화"""
        if model_key and model_key != self.current_model_key:
            self.switch_embedding_model(model_key)

        if self.embeddings is None:
            self.lazy_init_embedding(self.current_model_key)
        print(f"임베딩 및 벡터 스토어 생성 중... (모델: {self.current_model_key})")
        
        # Document 메타데이터 강화
        enhanced_documents = self._enhance_document_metadata(documents)
        
        print(f"문서 {len(enhanced_documents)}개의 메타데이터 강화 완료")
        
        try:
            vector_store = FAISS.from_documents(enhanced_documents, self.embeddings)
            self.vector_stores[self.current_model_key] = vector_store
            self.current_vector_store = vector_store
            
            # 벡터 스토어에 전체 문서 메타데이터 통계 추가
            self._add_vector_store_metadata(vector_store, enhanced_documents)
            
            return vector_store
        except Exception as e:
            print(f"벡터 스토어 생성 실패: {e}")
            # fallback model로 재시도
            if self.current_model_key != "paraphrase-multilingual":
                print("기본 모델로 재시도...")
                self.switch_embedding_model("paraphrase-multilingual")
                vector_store = FAISS.from_documents(enhanced_documents, self.embeddings)
                self.vector_stores[self.current_model_key] = vector_store
                self.current_vector_store = vector_store
                self._add_vector_store_metadata(vector_store, enhanced_documents)
                return vector_store
            raise e

    def add_documents(self, documents: List[Document]):
        """기존 벡터 스토어에 문서 추가 - Document 메타데이터 강화"""
        # Document 메타데이터 강화
        enhanced_documents = self._enhance_document_metadata(documents)
        
        print(f"추가할 문서 {len(enhanced_documents)}개의 메타데이터 강화 완료")
        
        if self.current_vector_store:
            self.current_vector_store.add_documents(enhanced_documents)
            self.vector_stores[self.current_model_key] = self.current_vector_store
            
            # 벡터 스토어 메타데이터 업데이트
            self._update_vector_store_metadata(self.current_vector_store, enhanced_documents)
        else:
            self.create_vector_store(enhanced_documents)

    def get_available_models(self) -> Dict[str, str]:
        """사용 가능한 임베딩 모델 목록"""
        return self.embedding_manager.get_available_models()

    def get_current_model_info(self) -> str:
        """현재 모델 정보"""
        return self.embedding_manager.get_current_model_info()
    
    def save_vector_store(self, index_name: str, model_key: str = None) -> tuple[bool, str]:
        """벡터 스토어를 파일로 저장"""
        if model_key is None:
            model_key = self.current_model_key
            
        if model_key not in self.vector_stores:
            return False, f"❌ 저장할 벡터 스토어가 없습니다: {model_key}"
        
        try:
            # FAISS 인덱스 저장 경로
            index_path = os.path.join(self.save_directory, f"{index_name}_{model_key}")
            
            index_dir = Path(index_path)
            index_dir.mkdir(exist_ok=True)

            # FAISS 인덱스 저장
            vector_store = self.vector_stores[model_key]
            vector_store.save_local(index_path)
            
            # 메타데이터 저장 (모델 정보, 생성일시, 문서 통계 등)
            document_metadata = getattr(vector_store, '_document_metadata', {})
            metadata = {
                'model_key': model_key,
                'model_name': self.embedding_manager.embedding_models[model_key]['name'],
                'index_name': index_name,
                'created_at': __import__('datetime').datetime.now().isoformat(),
                'document_count': len(vector_store.docstore._dict) if hasattr(vector_store.docstore, '_dict') else 'unknown',
                'document_sources': document_metadata.get('sources', []),
                'document_types': document_metadata.get('types', []),
                'content_length_stats': document_metadata.get('content_stats', {}),
                'unique_metadata_keys': document_metadata.get('metadata_keys', []),
                'last_modified': __import__('datetime').datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.save_directory, f"{index_name}_{model_key}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Debug용 JSON 파일로도 저장
            json_metadata_path = os.path.join(self.save_directory, f"{index_name}_{model_key}_metadata_debug.json")
            with open(json_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
            
            return True, f"✅ 벡터 스토어 저장 완료: {index_path}"
            
        except Exception as e:
            return False, f"❌ 벡터 스토어 저장 실패: {str(e)}"
    
    def load_vector_store(self, index_name: str, model_key: str = None) -> tuple[bool, str]:
        """저장된 벡터 스토어를 로드"""
        if model_key is None:
            model_key = self.current_model_key
        
        try:
            # FAISS 인덱스 로드 경로
            index_path = os.path.join(self.save_directory, f"{index_name}_{model_key}")

            faiss_file = os.path.join(index_path, "index.faiss")
            pkl_file = os.path.join(index_path, "index.pkl")            
            if not os.path.exists(faiss_file):
                return False, f"❌ 벡터 스토어 파일이 없습니다: {index_path}"
            
            # 해당 모델의 임베딩이 로드되지 않았다면 로드
            if model_key != self.current_model_key:
                self.switch_embedding_model(model_key)
            
            if self.embeddings is None:
                self.lazy_init_embedding(self.current_model_key)

            # FAISS 인덱스 로드
            vector_store = FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            self.vector_stores[model_key] = vector_store
            self.current_vector_store = vector_store
            
            # 메타데이터 로드 (있다면)
            metadata_path = os.path.join(self.save_directory, f"{index_name}_{model_key}_metadata.pkl")
            metadata_info = ""
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    metadata_info = f" (생성일: {metadata.get('created_at', 'Unknown')}, 문서수: {metadata.get('document_count', 'Unknown')})"
            
            return True, f"✅ 벡터 스토어 로드 완료: {index_name}_{model_key}{metadata_info}"
            
        except Exception as e:
            return False, f"❌ 벡터 스토어 로드 실패: {str(e)}"
    
    def list_saved_indexes(self) -> Dict[str, Dict[str, Any]]:
        """저장된 인덱스 목록 반환"""
        saved_indexes = {}
        
        if not os.path.exists(self.save_directory):
            return saved_indexes
        
        for filename in os.listdir(self.save_directory):
            if filename.endswith('.faiss'):
                # 파일명에서 인덱스명과 모델키 추출
                base_name = filename.replace('.faiss', '')
                # 마지막 '_'를 기준으로 분리
                parts = base_name.rsplit('_', 1)
                if len(parts) == 2:
                    index_name, model_key = parts
                    
                    # 메타데이터 정보 로드
                    metadata_path = os.path.join(self.save_directory, f"{base_name}_metadata.pkl")
                    metadata = {}
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'rb') as f:
                                metadata = pickle.load(f)
                        except:
                            pass
                    
                    if index_name not in saved_indexes:
                        saved_indexes[index_name] = {}
                    
                    saved_indexes[index_name][model_key] = {
                        'file_path': os.path.join(self.save_directory, filename),
                        'created_at': metadata.get('created_at', 'Unknown'),
                        'document_count': metadata.get('document_count', 'Unknown'),
                        'model_name': metadata.get('model_name', 'Unknown')
                    }
        
        return saved_indexes
    
    def delete_saved_index(self, index_name: str, model_key: str) -> str:
        """저장된 인덱스 삭제"""
        try:
            index_path = os.path.join(self.save_directory, f"{index_name}_{model_key}")
            metadata_path = os.path.join(self.save_directory, f"{index_name}_{model_key}_metadata.pkl")
            
            # FAISS 관련 파일들 삭제
            files_to_delete = [
                f"{index_path}.faiss",
                f"{index_path}.pkl",
                metadata_path
            ]
            
            deleted_files = []
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(os.path.basename(file_path))
            
            if deleted_files:
                return f"✅ 인덱스 삭제 완료: {', '.join(deleted_files)}"
            else:
                return f"❌ 삭제할 파일이 없습니다: {index_name}_{model_key}"
                
        except Exception as e:
            return f"❌ 인덱스 삭제 실패: {str(e)}"
    
    def index_exists(self, index_name: str, model_key: str = None) -> bool:
        """저장된 인덱스가 존재하는지 확인"""
        if model_key is None:
            model_key = self.current_model_key
        
        try:
            index_path = os.path.join(self.save_directory, f"{index_name}_{model_key}")
            faiss_file = os.path.join(index_path, "index.faiss")
            pkl_file = os.path.join(index_path, "index.pkl")
            
            # 두 파일 모두 존재해야 유효한 인덱스
            return os.path.exists(faiss_file) and os.path.exists(pkl_file)
        except Exception:
            return False
    
    def auto_save_after_creation(self, documents: List[Document], index_name: str, model_key: str = None) -> FAISS:
        """문서로부터 벡터 스토어 생성 후 자동 저장"""
        vector_store = self.create_vector_store(documents, model_key)
        save_result = self.save_vector_store(index_name, model_key)
        print(save_result)
        return vector_store


    def _enhance_document_metadata(self, documents: List[Document]) -> List[Document]:
        """Document 메타데이터 강화"""
        enhanced_docs = []
        
        for i, doc in enumerate(documents):
            # 기존 메타데이터 복사
            enhanced_metadata = doc.metadata.copy()
            
            # 기본 정보 추가
            enhanced_metadata.update({
                'doc_index': i,
                'content_length': len(doc.page_content),
                'content_word_count': len(doc.page_content.split()),
                'processing_timestamp': __import__('datetime').datetime.now().isoformat(),
                'embedding_model': self.current_model_key,
                'embedding_model_name': self.embedding_manager.embedding_models[self.current_model_key]['name']
            })
            
            # 내용 기반 메타데이터 추가
            content = doc.page_content
            enhanced_metadata.update({
                'has_korean': any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in content),
                'has_english': any(char.isascii() and char.isalpha() for char in content),
                'has_numbers': any(char.isdigit() for char in content),
                'line_count': len(content.split('\n')),
                'paragraph_count': len([p for p in content.split('\n\n') if p.strip()])
            })
            
            # 소스 정보 정규화
            if 'source' not in enhanced_metadata and 'file_path' in enhanced_metadata:
                enhanced_metadata['source'] = enhanced_metadata['file_path']
            
            # 문서 타입 추론
            doc_type = self._infer_document_type(enhanced_metadata, content)
            enhanced_metadata['inferred_type'] = doc_type
            
            enhanced_docs.append(Document(
                page_content=content,
                metadata=enhanced_metadata
            ))
        
        return enhanced_docs
    
    def _infer_document_type(self, metadata: dict, content: str) -> str:
        """문서 타입 추론"""
        # 파일 확장자 기반 추론
        source = metadata.get('source', '')
        if source.endswith('.pdf'):
            return 'PDF'
        elif source.endswith(('.txt', '.md')):
            return 'Text'
        elif source.endswith('.html'):
            return 'HTML'
        
        # 메타데이터 기반 추론
        if 'page' in metadata:
            return 'PDF'
        elif 'url' in metadata or source.startswith('http'):
            return 'Web'
        
        # 내용 기반 추론
        if '<!DOCTYPE html>' in content or '<html>' in content:
            return 'HTML'
        elif len(content.split('\n')) > 50 and len(content.split()) > 1000:
            return 'Long Document'
        else:
            return 'Text'
    
    def _add_vector_store_metadata(self, vector_store: FAISS, documents: List[Document]):
        """벡터 스토어에 전체 문서 메타데이터 통계 추가"""
        # 문서 통계 계산
        sources = list(set(doc.metadata.get('source', 'unknown') for doc in documents))
        doc_types = list(set(doc.metadata.get('inferred_type', 'unknown') for doc in documents))
        
        content_lengths = [len(doc.page_content) for doc in documents]
        content_stats = {
            'min_length': min(content_lengths) if content_lengths else 0,
            'max_length': max(content_lengths) if content_lengths else 0,
            'avg_length': sum(content_lengths) / len(content_lengths) if content_lengths else 0,
            'total_length': sum(content_lengths)
        }
        
        # 모든 메타데이터 키 수집
        all_metadata_keys = set()
        for doc in documents:
            all_metadata_keys.update(doc.metadata.keys())
        
        # 벡터 스토어에 메타데이터 저장
        vector_store._document_metadata = {
            'sources': sources,
            'types': doc_types,
            'content_stats': content_stats,
            'metadata_keys': list(all_metadata_keys),
            'document_count': len(documents),
            'created_at': __import__('datetime').datetime.now().isoformat()
        }
    
    def _update_vector_store_metadata(self, vector_store: FAISS, new_documents: List[Document]):
        """벡터 스토어 메타데이터 업데이트 (문서 추가 시)"""
        existing_metadata = getattr(vector_store, '_document_metadata', {})
        
        # 새로운 문서들의 통계 계산
        new_sources = list(set(doc.metadata.get('source', 'unknown') for doc in new_documents))
        new_types = list(set(doc.metadata.get('inferred_type', 'unknown') for doc in new_documents))
        
        new_content_lengths = [len(doc.page_content) for doc in new_documents]
        
        # 기존 통계와 합병
        all_sources = list(set(existing_metadata.get('sources', []) + new_sources))
        all_types = list(set(existing_metadata.get('types', []) + new_types))
        
        # 메타데이터 키 업데이트
        all_metadata_keys = set(existing_metadata.get('metadata_keys', []))
        for doc in new_documents:
            all_metadata_keys.update(doc.metadata.keys())
        
        # 전체 문서 수 업데이트
        total_docs = existing_metadata.get('document_count', 0) + len(new_documents)
        
        # 업데이트된 메타데이터 저장
        vector_store._document_metadata = {
            'sources': all_sources,
            'types': all_types,
            'content_stats': {
                'new_docs_count': len(new_documents),
                'new_docs_total_length': sum(new_content_lengths),
                'new_docs_avg_length': sum(new_content_lengths) / len(new_content_lengths) if new_content_lengths else 0
            },
            'metadata_keys': list(all_metadata_keys),
            'document_count': total_docs,
            'last_updated': __import__('datetime').datetime.now().isoformat(),
            'created_at': existing_metadata.get('created_at', __import__('datetime').datetime.now().isoformat())
        }
    
    def get_document_metadata_summary(self) -> Dict[str, Any]:
        """현재 벡터 스토어의 문서 메타데이터 요약 반환"""
        if not self.current_vector_store:
            return {'error': '로드된 벡터 스토어가 없습니다'}
        
        metadata = getattr(self.current_vector_store, '_document_metadata', {})
        if not metadata:
            return {'error': '메타데이터 정보가 없습니다'}
        
        return {
            '문서 수': metadata.get('document_count', 0),
            '소스 목록': metadata.get('sources', []),
            '문서 타입': metadata.get('types', []),
            '내용 통계': metadata.get('content_stats', {}),
            '메타데이터 키': metadata.get('metadata_keys', []),
            '생성일': metadata.get('created_at', 'Unknown'),
            '최근 업데이트': metadata.get('last_updated', metadata.get('created_at', 'Unknown'))
        }
    
    def search_documents_by_metadata(self, metadata_key: str, metadata_value: Any, top_k: int = 10) -> List[Document]:
        """메타데이터를 기준으로 문서 검색"""
        if not self.current_vector_store:
            return []
        
        # FAISS에서 모든 문서 가져오기
        all_docs = []
        if hasattr(self.current_vector_store, 'docstore') and hasattr(self.current_vector_store.docstore, '_dict'):
            for doc_id, doc in self.current_vector_store.docstore._dict.items():
                if metadata_key in doc.metadata and doc.metadata[metadata_key] == metadata_value:
                    all_docs.append(doc)
        
        return all_docs[:top_k]


if __name__ == "__main__":
    vector_store_manager = VectorStoreManager()