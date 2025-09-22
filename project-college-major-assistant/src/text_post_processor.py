import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib
from collections import Counter
import numpy as np


@dataclass
class Chunk:
    """청크 데이터 구조"""
    id: str
    text: str
    metadata: Dict[str, Any]
    dense_text: str = ""
    sparse_tokens: List[str] = None
    parent_id: str = None
    child_ids: List[str] = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata,
            'dense_text': self.dense_text,
            'sparse_tokens': self.sparse_tokens,
            'parent_id': self.parent_id,
            'child_ids': self.child_ids
        }


class OCRCorrector:
    """OCR 오류 정정 클래스"""
    
    def __init__(self):
        # 일반적인 OCR 오류 패턴
        self.common_errors = {
            "동신": "통신",
            "회곡": "희곡",
            "용합": "융합",
            "음용": "응용",
            "동해": "통해",
            "않": "있",
            "좋이": "종이",
            "날성": "낯선",
            "끌": "클",
            "털": "털",
            "날힐": "넓힐",
            "1T": "IT",
            "OS": "QS",
            "니": "L",
            "굿": "못"
        }
        
        # 노이즈 패턴
        self.noise_patterns = [
            r'ALIS[A-Z\s]+VINON[^\n]*\n?',
            r'[^\s]{30,}',  # 너무 긴 연속 문자
            r'[\s]{4,}',     # 과도한 공백
            r'[0-9\s]{10,}', # 의미없는 숫자 나열
            r'[·\-]{5,}'     # 반복되는 특수문자
        ]
    
    def correct(self, text: str) -> str:
        """OCR 오류 정정"""
        # 1. 노이즈 제거
        for pattern in self.noise_patterns:
            text = re.sub(pattern, ' ', text)
        
        # 2. 일반적인 오류 교정
        for error, correction in self.common_errors.items():
            text = text.replace(error, correction)
        
        # 3. 특수문자 정리
        text = re.sub(r'[\s]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 4. 잘못된 띄어쓰기 교정
        text = self._fix_spacing(text)
        
        return text.strip()
    
    def _fix_spacing(self, text: str) -> str:
        """띄어쓰기 교정"""
        # 숫자와 한글 사이 띄어쓰기
        text = re.sub(r'(\d)([가-힣])', r'\1 \2', text)
        text = re.sub(r'([가-힣])(\d)', r'\1 \2', text)
        
        # 영문과 한글 사이 띄어쓰기
        text = re.sub(r'([A-Za-z])([가-힣])', r'\1 \2', text)
        text = re.sub(r'([가-힣])([A-Za-z])', r'\1 \2', text)
        
        return text


class MetadataExtractor:
    """메타데이터 추출 클래스"""
    
    def extract(self, text: str, source: str = None) -> Dict[str, Any]:
        """텍스트에서 메타데이터 추출"""
        metadata = {}
        
        # 학과명 추출
        dept_pattern = r'([\w]+학과|[\w]+학부)'
        dept_match = re.search(dept_pattern, text)
        if dept_match:
            metadata['department'] = dept_match.group(1)
        
        # 대학명 추출
        college_pattern = r'([\w]+대학)'
        college_match = re.search(college_pattern, text)
        if college_match:
            metadata['college'] = college_match.group(1)
        
        # 연도 추출
        year_pattern = r'(19\d{2}|20\d{2})년'
        years = re.findall(year_pattern, text)
        if years:
            metadata['years'] = list(set(years))
        
        # 섹션 타입 분류
        metadata['section_type'] = self._classify_section(text)
        
        # 키워드 추출
        metadata['keywords'] = self._extract_keywords(text)
        
        # 소스 정보
        if source:
            metadata['source'] = source
        
        # 텍스트 길이
        metadata['text_length'] = len(text)
        
        return metadata
    
    def _classify_section(self, text: str) -> str:
        """섹션 타입 분류"""
        section_keywords = {
            '교육과정': ['교육과정', '커리큘럼', '전공과목', '학년', '과목'],
            '진로': ['진로', '졸업', '취업', '경력', '직업'],
            '입학': ['입학', '지원', '전형', '모집'],
            '연구': ['연구', '연구소', '연구실', 'LAB'],
            '프로그램': ['프로그램', '특별', '교환학생', '인턴'],
            '소개': ['소개', '역사', '설립', '비전']
        }
        
        for section_type, keywords in section_keywords.items():
            if any(keyword in text for keyword in keywords):
                return section_type
        
        return '일반'
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """키워드 추출"""
        # 명사 패턴 (간단한 휴리스틱)
        noun_pattern = r'[가-힣]{2,}'
        nouns = re.findall(noun_pattern, text)
        
        # 불용어 제거
        stopwords = {'있다', '있는', '하는', '되는', '이다', '그리고', '또한', '통해'}
        nouns = [n for n in nouns if n not in stopwords and len(n) > 1]
        
        # 빈도수 기반 상위 키워드
        counter = Counter(nouns)
        return [word for word, _ in counter.most_common(top_n)]


class HybridChunker:
    """하이브리드 청킹 클래스"""
    
    def __init__(self, 
                 min_size: int = 200, 
                 max_size: int = 800, 
                 overlap: int = 100):
        self.min_size = min_size
        self.max_size = max_size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """텍스트를 청크로 분할"""
        chunks = []
        
        # 1. 의미적 경계 찾기
        sections = self._find_semantic_sections(text)
        
        # 2. 각 섹션을 청킹
        for section in sections:
            section_chunks = self._chunk_section(section, metadata)
            chunks.extend(section_chunks)
        
        # 3. Parent-Child 관계 설정
        chunks = self._create_parent_child_relationships(chunks)
        
        return chunks
    
    def _find_semantic_sections(self, text: str) -> List[str]:
        """의미적 섹션 찾기"""
        # 제목 패턴
        title_patterns = [
            r'\n#{1,3}\s+.+\n',  # Markdown 헤더
            r'\n[가-힣\s]{2,20}\n(?=[^\n])',  # 짧은 제목
            r'\n\d+\.\s+.+\n',  # 번호가 있는 제목
        ]
        
        sections = []
        current_section = []
        lines = text.split('\n')
        
        for line in lines:
            is_title = any(re.match(pattern.strip('\n'), line) 
                          for pattern in title_patterns)
            
            if is_title and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        # 섹션이 없으면 전체 텍스트를 하나의 섹션으로
        if not sections:
            sections = [text]
        
        return sections
    
    def _chunk_section(self, section: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """섹션을 청크로 분할"""
        chunks = []
        sentences = self._split_sentences(section)
        
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.max_size and current_chunk:
                # 청크 생성
                chunk_text = ' '.join(current_chunk)
                chunk_id = self._generate_chunk_id(chunk_text)
                
                chunk = Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                
                # 오버랩 처리
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size < self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # 마지막 청크
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_size or not chunks:  # 최소 크기 확인
                chunk_id = self._generate_chunk_id(chunk_text)
                chunk = Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=metadata or {}
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리"""
        # 한국어 문장 종결 패턴
        sentence_endings = r'[.!?]\s+|[\n]+'
        sentences = re.split(sentence_endings, text)
        
        # 빈 문장 제거
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _create_parent_child_relationships(self, chunks: List[Chunk]) -> List[Chunk]:
        """Parent-Child 관계 설정"""
        # 2개 이상의 청크를 합쳐서 parent 생성
        parent_chunks = []
        
        for i in range(0, len(chunks), 2):
            if i + 1 < len(chunks):
                # 두 개의 청크를 합쳐서 parent 생성
                parent_text = chunks[i].text + " " + chunks[i+1].text
                parent_id = self._generate_chunk_id(parent_text)
                
                parent_metadata = chunks[i].metadata.copy()
                parent_metadata['is_parent'] = True
                
                parent = Chunk(
                    id=parent_id,
                    text=parent_text,
                    metadata=parent_metadata,
                    child_ids=[chunks[i].id, chunks[i+1].id]
                )
                parent_chunks.append(parent)
                
                # 자식 청크에 부모 ID 설정
                chunks[i].parent_id = parent_id
                chunks[i+1].parent_id = parent_id
        
        return chunks + parent_chunks
    
    def _generate_chunk_id(self, text: str) -> str:
        """청크 ID 생성"""
        return hashlib.md5(text.encode()).hexdigest()[:12]


class AdvancedRAGPreprocessor:
    """고급 RAG 전처리기"""
    
    def __init__(self):
        self.ocr_corrector = OCRCorrector()
        self.metadata_extractor = MetadataExtractor()
        self.chunker = HybridChunker()
    
    def process_json_file(self, file_path: str) -> Dict[str, Any]:
        """JSON 파일 처리"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self.process_documents(data)
    
    def process_documents(self, documents: List[Dict]) -> Dict[str, Any]:
        """문서 리스트 처리"""
        all_chunks = []
        document_metadata = {}
        
        for doc in documents:
            # 문서 정보 추출
            page_content = doc.get('page_content', '')
            doc_metadata = doc.get('metadata', {})
            
            # OCR 오류 정정
            cleaned_text = self.ocr_corrector.correct(page_content)
            
            # 메타데이터 추출
            extracted_metadata = self.metadata_extractor.extract(
                cleaned_text, 
                doc_metadata.get('source', '')
            )
            
            # 기존 메타데이터와 병합
            combined_metadata = {**doc_metadata, **extracted_metadata}
            
            # 청킹
            chunks = self.chunker.chunk(cleaned_text, combined_metadata)
            
            # 각 청크 후처리
            for chunk in chunks:
                # Dense retrieval용 텍스트 준비
                chunk.dense_text = self._prepare_for_dense_retrieval(chunk)
                
                # Sparse retrieval용 토큰 준비
                chunk.sparse_tokens = self._prepare_for_sparse_retrieval(chunk)
            
            all_chunks.extend(chunks)
            
            # 문서 메타데이터 저장
            if doc_metadata.get('source'):
                document_metadata[doc_metadata['source']] = combined_metadata
        
        # 통계 생성
        stats = self._generate_statistics(all_chunks)
        
        return {
            'chunks': [chunk.to_dict() for chunk in all_chunks],
            'document_metadata': document_metadata,
            'statistics': stats,
            'total_chunks': len(all_chunks)
        }
    
    def _prepare_for_dense_retrieval(self, chunk: Chunk) -> str:
        """Dense retrieval용 텍스트 준비"""
        # 메타데이터 컨텍스트 추가
        context_parts = []
        
        if chunk.metadata.get('department'):
            context_parts.append(f"[학과: {chunk.metadata['department']}]")
        
        if chunk.metadata.get('college'):
            context_parts.append(f"[대학: {chunk.metadata['college']}]")
        
        if chunk.metadata.get('section_type'):
            context_parts.append(f"[섹션: {chunk.metadata['section_type']}]")
        
        # 컨텍스트와 텍스트 결합
        context = ' '.join(context_parts)
        enhanced_text = f"{context} {chunk.text}" if context else chunk.text
        
        # 약어 확장
        enhanced_text = self._expand_abbreviations(enhanced_text)
        
        return enhanced_text
    
    def _prepare_for_sparse_retrieval(self, chunk: Chunk) -> List[str]:
        """Sparse retrieval용 토큰 준비"""
        tokens = []
        
        # 기본 토큰화
        base_tokens = re.findall(r'[\w가-힣]+', chunk.text.lower())
        tokens.extend(base_tokens)
        
        # 키워드 추가
        if chunk.metadata.get('keywords'):
            tokens.extend(chunk.metadata['keywords'])
        
        # n-gram 생성 (2-gram, 3-gram)
        for n in [2, 3]:
            ngrams = self._generate_ngrams(base_tokens, n)
            tokens.extend(ngrams[:10])  # 상위 10개만
        
        # 중복 제거
        return list(set(tokens))
    
    def _expand_abbreviations(self, text: str) -> str:
        """약어 확장"""
        abbreviations = {
            'AI': 'AI 인공지능 Artificial Intelligence',
            'ML': 'ML 머신러닝 Machine Learning',
            'IoT': 'IoT 사물인터넷 Internet of Things',
            'IT': 'IT 정보기술 Information Technology',
            'CS': 'CS 컴퓨터과학 Computer Science',
            'DB': 'DB 데이터베이스 Database',
            'OS': 'OS 운영체제 Operating System',
            'SW': 'SW 소프트웨어 Software',
            'HW': 'HW 하드웨어 Hardware'
        }
        
        for abbr, expansion in abbreviations.items():
            # 단어 경계에서만 치환
            pattern = r'\b' + abbr + r'\b'
            text = re.sub(pattern, expansion, text)
        
        return text
    
    def _generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """n-gram 생성"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = '_'.join(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    def _generate_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """통계 생성"""
        stats = {
            'total_chunks': len(chunks),
            'parent_chunks': sum(1 for c in chunks if c.child_ids),
            'child_chunks': sum(1 for c in chunks if c.parent_id),
            'avg_chunk_length': np.mean([len(c.text) for c in chunks]),
            'min_chunk_length': min(len(c.text) for c in chunks),
            'max_chunk_length': max(len(c.text) for c in chunks),
            'section_types': Counter(c.metadata.get('section_type', '일반') 
                                   for c in chunks),
            'departments': list(set(c.metadata.get('department', '') 
                                  for c in chunks if c.metadata.get('department')))
        }
        
        return stats
    
    def save_processed_data(self, processed_data: Dict[str, Any], output_path: str):
        """처리된 데이터 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processed data saved to {output_path}")
        print(f"Total chunks: {processed_data['total_chunks']}")
        print(f"Statistics: {processed_data['statistics']}")


# 사용 예제
if __name__ == "__main__":
    # 전처리기 생성
    preprocessor = AdvancedRAGPreprocessor()
    
    # JSON 파일 처리 예제
    doc_json = "./temp_texts/01-경영대학/documents/01-경영대학_documents.json"
    processed_data = preprocessor.process_json_file(doc_json)
    
    # # 또는 직접 데이터 처리
    # sample_documents = [
    #     {
    #         "page_content": "Since1905 THE FIRST & THE BEST 경영학과 Korea University Business School...",
    #         "metadata": {
    #             "source": "01-경영대학.pdf",
    #             "page": 1,
    #             "processed_at": "2025-09-19T16:58:12.004500"
    #         }
    #     }
    # ]    
    # processed_data = preprocessor.process_documents(sample_documents)
    
    # 결과 저장
    preprocessor.save_processed_data(processed_data, "processed_rag_data.json")
    
    # 처리된 청크 예시 출력
    if processed_data['chunks']:
        print("\n=== 첫 번째 청크 예시 ===")
        first_chunk = processed_data['chunks'][0]
        print(f"ID: {first_chunk['id']}")
        print(f"Text: {first_chunk['text'][:200]}...")
        print(f"Metadata: {first_chunk['metadata']}")
        print(f"Dense Text: {first_chunk['dense_text'][:200]}...")
        print(f"Sparse Tokens (sample): {first_chunk['sparse_tokens'][:10]}")