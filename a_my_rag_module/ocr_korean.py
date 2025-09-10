

from paddleocr import PPStructureV3 #, draw_structure_result, save_structure_res
import easyocr
import cv2, os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt


from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import kss
import re

import torch
from transformers import BertTokenizer, BertForMaskedLM

# 설치
# pip install easyocr
# pip install sentencepiece
# pip install kss, python-mecab-kor,  mecab-python3
# pip install paddleocr
# pip install "paddleocr[doc-parser]"

class KoreanOCRCorrector:
    def __init__(self, replace_mode = True):
        self.replace_mode = replace_mode
        if replace_mode :
            # KoBERT 로드
            self.model_name = "klue/bert-base" # "monologg/kobert" #klue/bert-base
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForMaskedLM.from_pretrained(self.model_name)
            self.model.eval()            
        else :
            # KoBART 또는 mT5 사용 (한글 지원)
            self.model_name = "gogamza/kobart-base-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
       
        #         
        # 한글 맞춤법 검사기
        # from hanspell import spell_checker
        # self.spell_checker = spell_checker
        
    def correct_with_context(self, text):
        """문맥 기반 보정"""
        # 1. 기본 전처리
        text = self.preprocess_text(text)
        
        # 2. 문장 단위 분할
        sentences = self.split_sentences(text)
        
        corrected_sentences = []
        for sent in sentences:
            # 3. 문맥 기반 보정
            if self.replace_mode:
                corrected = self._detect_and_correct(sent)
            else:    
                corrected = self._apply_context_correction(sent)
            
            # 4. 맞춤법 검사
            corrected = self.apply_spell_check(corrected)
            
            corrected_sentences.append(corrected)
        
        return ' '.join(corrected_sentences)
    
    def preprocess_text(self, text):
        """기본 전처리"""
        # 일반적인 OCR 오류 패턴 수정
        replacements = {
            r'([가-힣])(\d)([가-힣])': r'\1\3',  # 한글 사이 숫자 제거
            r'([가-힣])\s{2,}([가-힣])': r'\1 \2',  # 과도한 공백 정리
            r'[ㄱ-ㅎㅏ-ㅣ]+': '',  # 단독 자모음 제거
            '０': '0', '１': '1', '２': '2',  # 전각 숫자 변환
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def split_sentences(self, text):
        return kss.split_sentences(text)
    
    def apply_spell_check(self, text):
        return text

    def _apply_context_correction(self, text: str):
        """Transformer 모델로 문맥 보정"""
        inputs = self.tokenizer(
            f"correct: {text}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # token_type_ids가 있으면 제거
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True,
            temperature=0.7
        )
        
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected

    def _detect_and_correct(self, text: str, topk=3, threshold=-5.0) -> str:
            
        tokens = self.tokenizer.tokenize(text)
        corrected_tokens = tokens[:]
        
        for i, tok in enumerate(tokens):
            # 각 토큰을 마스크로 교체
            masked_tokens = tokens[:]
            masked_tokens[i] = '[MASK]'
            input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + masked_tokens + ['[SEP]'])
            input_tensor = torch.tensor([input_ids])
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                predictions = outputs.logits
            
            mask_index = masked_tokens.index('[MASK]') + 1
            softmax_probs = torch.nn.functional.log_softmax(predictions[0, mask_index], dim=-1)
            
            # 현재 단어 점수 확인
            orig_id = self.tokenizer.convert_tokens_to_ids([tok])[0]
            orig_score = softmax_probs[orig_id].item()
            
            # 점수가 너무 낮으면 보정 후보
            if orig_score < threshold:
                # Top-k 후보 뽑기
                top_ids = torch.topk(softmax_probs, topk).indices.tolist()
                candidates = self.tokenizer.convert_ids_to_tokens(top_ids)
                print(f"[의심 단어] {tok} → 후보: {candidates}")
                
                # 여기서는 단순히 Top-1 교체
                corrected_tokens[i] = candidates[0]
        
        return self.tokenizer.convert_tokens_to_string(corrected_tokens)


class KoreanOCR:
    def __init__(self, use_paddle=True):
        self.use_paddle = use_paddle
        self.paddle_ocr = None
        self.eacy_ocr = None
        #self.corrector = KoreanOCRCorrector(replace_mode = False)
        self.corrector = None

    def _load_ocr_model(self):
        if self.use_paddle:
            self.paddle_ocr = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                lang="korean"
            )
        else:
            # GPU 사용 (없으면 자동으로 CPU 사용)
            self.eacy_ocr = easyocr.Reader(['ko', 'en'], gpu=True)

    def cleanup_ocr_model(self):
        """OCR 모델 해제 (필요시)"""
        if self.eacy_ocr is not None:
            self.ocr_model = None
        if self.paddle_ocr is not None:
            self.paddle_ocr = None

    # 후처리: 한글 정규화
    def normalize_korean(text: str) -> str:
        """
        - 불필요한 공백 제거
        - 중복된 특수문자 정리
        - OCR에서 흔히 생기는 잘못된 분리 정리
        """
        text = re.sub(r"\s+", " ", text)          # 다중 공백 제거
        text = re.sub(r"[~!@#%^&*()_+=<>?/]{2,}", " ", text)  # 특수문자 연속 제거
        text = re.sub(r"([가-힣])\s+([가-힣])", r"\1\2", text) # 한글 사이 불필요한 공백 제거
        return text.strip()

    def extract_text(self, image_path, threshold=0.5):
        """
        이미지에서 텍스트 추출
        
        Args:
            image_path: 이미지 파일 경로
            threshold: 신뢰도 임계값
        """
        if self.use_paddle and self.paddle_ocr is None:
            self._load_ocr_model()
        elif not self.use_paddle and self.eacy_ocr is None:
            self._load_ocr_model()

        if self.paddle_ocr:
            layout_info = self.extract_with_paddleocr(image_path)
        else: 
            layout_info = self.extract_with_layout(image_path)
        
        all_text = '\n'.join( para['text'] for para in layout_info)        
        return all_text

    def extract_with_layout(self, image_path):
        """레이아웃 정보와 함께 추출"""
        # 상세 정보 포함 추출
        results = self.eacy_ocr.readtext(
            image_path,
            detail=1,
            paragraph=True,  # 단락 병합 활성화
            x_ths=1.0,  # x축 거리 임계값
            y_ths=0.5   # y축 거리 임계값  
        )
        
        confidence = 0.8 # not code
        # 레이아웃 분석
        layout_info = []
        for i, (bbox, text) in enumerate(results):
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            post_text = self.correct_with_context(text)
            if text != post_text:
                print(f"correct text : {text} \n --> {post_text}")

            layout_info.append({
                'para_id': i,
                'text': post_text,
                'confidence': confidence,
                'bbox': bbox,
                'x_min': min(x_coords),
                'x_max': max(x_coords),
                'y_min': min(y_coords),
                'y_max': max(y_coords),
                'width': max(x_coords) - min(x_coords),
                'height': max(y_coords) - min(y_coords),
                'center_x': (min(x_coords) + max(x_coords)) / 2,
                'center_y': (min(y_coords) + max(y_coords)) / 2
            })
        
        # Y 좌표로 정렬 (위에서 아래로)
        layout_info.sort(key=lambda x: x['y_min'])
        
        return layout_info

   
    def calculate_bbox_overlap(self, bbox1, bbox2):
        """
        두 바운딩 박스의 겹침 비율 계산
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2] 형태의 바운딩 박스
        
        Returns:
            float: 겹침 비율 (0~1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 겹치는 영역 계산
        x1_overlap = max(x1_1, x1_2)
        y1_overlap = max(y1_1, y1_2)
        x2_overlap = min(x2_1, x2_2)
        y2_overlap = min(y2_1, y2_2)
        
        # 겹치지 않는 경우
        if x1_overlap >= x2_overlap or y1_overlap >= y2_overlap:
            return 0.0
        
        # 겹치는 면적
        overlap_area = (x2_overlap - x1_overlap) * (y2_overlap - y1_overlap)
        
        # OCR 바운딩 박스 면적
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        # OCR 박스 대비 겹침 비율
        return overlap_area / bbox1_area if bbox1_area > 0 else 0.0
    
    def is_bbox_completely_inside(inner_bbox, outer_bbox):
        """바운딩 박스가 완전히 내부에 있는지 확인"""
        x1_i, y1_i, x2_i, y2_i = inner_bbox
        x1_o, y1_o, x2_o, y2_o = outer_bbox
        
        return (x1_i >= x1_o and y1_i >= y1_o and 
                x2_i <= x2_o and y2_i <= y2_o)

    def is_center_inside_bbox(bbox, target_bbox):
        """바운딩 박스의 중심점이 대상 영역 내부에 있는지 확인"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        tx1, ty1, tx2, ty2 = target_bbox
        return tx1 <= center_x <= tx2 and ty1 <= center_y <= ty2

    def remove_nested_boxes(self, layout_boxes):
        """
        다른 박스 안에 완전히 포함되는 작은 박스들을 제거
        
        Args:
            layout_boxes: PPStructureV3의 layout detection 결과 박스들
            
        Returns:
            list: 중복 제거된 박스들
        """
        filtered_boxes = []
        
        for i, box in enumerate(layout_boxes):
            coord = box['coordinate']
            x1, y1, x2, y2 = coord
            
            # 현재 박스가 다른 박스에 완전히 포함되는지 확인
            is_nested = False
            
            for j, other_box in enumerate(layout_boxes):
                if i == j:  # 자기 자신과는 비교하지 않음
                    continue
                    
                other_coord = other_box['coordinate']
                ox1, oy1, ox2, oy2 = other_coord
                
                # 현재 박스가 다른 박스에 완전히 포함되는지 확인
                if (ox1 <= x1 and oy1 <= y1 and ox2 >= x2 and oy2 >= y2):
                    # 완전히 같은 크기인 경우는 제외 (동일한 박스)
                    if not (ox1 == x1 and oy1 == y1 and ox2 == x2 and oy2 == y2):
                        is_nested = True
                        break
            
            # 중첩되지 않은 박스만 유지
            if not is_nested:
                filtered_boxes.append(box)
        
        return filtered_boxes

    
    def extract_with_paddleocr(self, image_path):
        # For Image
        results = self.paddle_ocr.predict(
            input=image_path
        )

        # 레이아웃 분석
        
        layout_info = []
        for i, result in enumerate(results):
            # PPStructureV3 결과 구조에서 layout detection과 OCR 결과 접근
            res_data = result

            # Layout detection 결과
            layout_boxes = res_data['layout_det_res']['boxes']
            
            # 중복된 박스 제거 (다른 박스 안에 완전히 포함되는 작은 박스 제거)
            filtered_boxes = self.remove_nested_boxes(layout_boxes)
            
            # Overall OCR 결과
            ocr_texts = res_data['overall_ocr_res']['rec_texts']
            ocr_boxes = res_data['overall_ocr_res']['rec_boxes']
            ocr_scores = res_data['overall_ocr_res']['rec_scores']
            
            # 각 layout box에 대해 해당하는 OCR 텍스트 찾기
            for j, layout_box in enumerate(filtered_boxes):
                # Layout box 좌표 (x_min, y_min, x_max, y_max)
                layout_coord = layout_box['coordinate']
                x_min, y_min, x_max, y_max = layout_coord
                
                # 해당 레이아웃 영역 내의 OCR 텍스트들 찾기
                texts_in_region = []
                total_confidence = 0
                box_count = 0
                
                for k, ocr_box in enumerate(ocr_boxes):
                    ocr_x_min, ocr_y_min, ocr_x_max, ocr_y_max = ocr_box
                    
                    # OCR 박스가 layout 박스 안에 있는지 확인 (중심점 기준)
                    ocr_center_x = (ocr_x_min + ocr_x_max) / 2
                    ocr_center_y = (ocr_y_min + ocr_y_max) / 2
                    
                    if (x_min <= ocr_center_x <= x_max and 
                        y_min <= ocr_center_y <= y_max):
                        if k < len(ocr_texts):
                            texts_in_region.append(ocr_texts[k])
                            if k < len(ocr_scores):
                                total_confidence += ocr_scores[k]
                                box_count += 1
                
                # 텍스트 결합 및 보정
                combined_text = ' '.join(texts_in_region)
                post_text = self.correct_with_context(combined_text) if combined_text else ""
                
                if combined_text != post_text and post_text:
                    print(f"correct text : {combined_text} \n --> {post_text}")
                
                # 평균 신뢰도 계산
                avg_confidence = total_confidence / box_count if box_count > 0 else layout_box['score']
                
                # bbox를 EasyOCR 형식으로 변환 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [x_min, x_max, x_max, x_min]
                y_coords = [y_min, y_min, y_max, y_max]
                
                layout_info.append({
                    'para_id': len(layout_info),
                    'text': post_text,
                    'block_type': layout_box['label'],  # text, image, doc_title 등
                    'confidence': float(avg_confidence),
                    'bbox': [[x_coords[i], y_coords[i]] for i in range(4)],
                    'x_min': float(x_min),
                    'x_max': float(x_max),
                    'y_min': float(y_min),
                    'y_max': float(y_max),
                    'width': float(x_max - x_min),
                    'height': float(y_max - y_min),
                    'center_x': float((x_min + x_max) / 2),
                    'center_y': float((y_min + y_max) / 2)
                })

            # Y 좌표로 정렬 (위에서 아래로)
            layout_info.sort(key=lambda x: x['y_min'])
        
        return layout_info

    def correct_with_context(self, text: str) -> str:
        if self.corrector:
            return self.corrector.correct_with_context(text)
        else:
            return text
        
    def visualize_results(self, image_path, results):
        """결과 시각화"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for i, result in enumerate(results):
            id = result['para_id']
            bbox = result['bbox']
            text = result['text']
            # confidence = result['confidence']
            
            # 바운딩 박스 그리기
            pts = np.array(bbox, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            
            # 텍스트 표시
            cv2.putText(image, f"{ i + 1}", 
                       (int(bbox[0][0]), int(bbox[0][1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return image

