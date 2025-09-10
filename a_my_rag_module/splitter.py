from typing import List, Dict, Any, Optional
# 한국어 처리
from konlpy.tag import Okt
import re
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============================================================================
# 향상된 한국어 텍스트 전처리 및 분할 클래스 (분할 - 청크 단위 분할)
# =============================================================================

class KoreanTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.okt = Okt()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def preprocess_text(self, text: str) -> str:
        """한국어 텍스트 전처리"""
        # 불필요한 문자 제거하되 검색을 위해 일부 특수문자 보존
        text = re.sub(r'[^\w\s가-힣.,!?()-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_keywords(self, text: str) -> List[str]:
        """검색용 키워드 추출"""
        try:
            # 형태소 분석
            morphs = self.okt.pos(text, stem=True)

            # 중요한 품사만 추출 (명사, 형용사, 동사, 영어)
            keywords = []
            for word, pos in morphs:
                if pos in ['Noun', 'Adjective', 'Verb'] and len(word) > 1:
                    keywords.append(word)
                elif pos == 'Alpha' and len(word) > 2:  # 영어 단어
                    keywords.append(word.lower())

            # 중복 제거 및 빈도순 정렬
            from collections import Counter
            keyword_counts = Counter(keywords)
            return [word for word, count in keyword_counts.most_common()]

        except Exception as e:
            print(f"키워드 추출 실패: {e}")
            return text.split()

    def morphological_analysis(self, text: str) -> str:
        """형태소 분석 및 정규화"""
        try:
            # 명사와 형용사, 동사 추출
            morphs = self.okt.pos(text, stem=True)
            important_morphs = []

            for word, pos in morphs:
                if pos in ['Noun', 'Adjective', 'Verb'] and len(word) > 1:
                    important_morphs.append(word)
                elif pos == 'Alpha' and len(word) > 2:
                    important_morphs.append(word.lower())

            return ' '.join(important_morphs)
        except:
            return text

    def create_searchable_content(self, text: str) -> str:
        """검색 최적화된 콘텐츠 생성"""
        # 원본 텍스트
        original = self.preprocess_text(text)
        # 형태소 분석 결과
        morphed = self.morphological_analysis(text)
        # 키워드 추출 결과
        keywords = ' '.join(self.extract_keywords(text))

        # 모든 버전을 결합하여 검색 성능 향상
        return f"{original} {morphed} {keywords}"

    def create_documents(
        self, texts: list[str], metadatas: Optional[list[dict[Any, Any]]] = None
    ) -> list[Document]:
        """문서를 청크 단위로 분할 (검색 최적화)"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", ".", " "]
        )
        documents = []
        documents = splitter.create_documents(texts, metadatas)
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 청크 단위로 분할 (검색 최적화)"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", ".", " "]
        )

        split_docs = []
        for doc in documents:
            # 텍스트 전처리
            preprocessed_text = self.preprocess_text(doc.page_content)            
            # 검색용 콘텐츠 생성
            searchable_content = self.create_searchable_content(doc.page_content)

            # 키워드 추출
            keywords = self.extract_keywords(doc.page_content)

            # 문서 분할
            chunks = splitter.split_text(preprocessed_text)

            for i, chunk in enumerate(chunks):
                # 청크별 키워드도 추출
                chunk_keywords = self.extract_keywords(chunk)
                chunk_searchable = self.create_searchable_content(chunk)

                split_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_id': i,
                        'keywords': keywords + chunk_keywords,  # 전체 + 청크 키워드
                        'searchable_content': chunk_searchable,  # 검색 최적화 콘텐츠
                        'morphed_content': self.morphological_analysis(chunk)
                    }
                )
                split_docs.append(split_doc)

        return split_docs


if __name__ == "__main__":
    splitter = KoreanTextSplitter()
