import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from model_detection import initialize_model, detect_objects
import ujson as json
from rdtdet_log import logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 모델 초기화
ocr = PaddleOCR(use_angle_cls=False, lang='korean')
rtmdet = initialize_model()
IMAGE_THRESHOLD = 0.6
COLUMN_THRESHOLD = 0.6

def resize_image(image, target_size=1000):
    height, width = image.shape[:2]
    scale = target_size / max(height, width)
    new_size = (int(width * scale), int(height * scale))
    resized_img = cv2.resize(image, new_size)
    return resized_img, scale

def merge_ocr_in_cell(ocr_results):
    if not ocr_results:
        return ""
    
    # y 좌표를 기준으로 정렬
    sorted_results = sorted(ocr_results, key=lambda x: x['bbox'][0][1])
    
    merged_text = ""
    prev_y = None
    for word in sorted_results:
        if prev_y is not None and word['bbox'][0][1] - prev_y > 20:  # 줄바꿈 기준 (20픽셀)
            merged_text += "\n"
        merged_text += word['text'] + " "
        prev_y = word['bbox'][2][1]
    
    return merged_text.strip()

def process_image(image_path):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"{image_path}에서 이미지를 불러오는데 실패했습니다.")
    
    # 이미지 리사이즈
    resized_img, scale = resize_image(original_img)
    
    # 표 모델 실행 및 열 인식
    detected_objects = detect_objects(rtmdet, resized_img, IMAGE_THRESHOLD)
    columns = [obj for obj in detected_objects if obj['class_name'] == 'column' and obj['score'] > COLUMN_THRESHOLD]
    columns.sort(key=lambda x: x['bbox'][0])  # x 좌표로 정렬
    
    # OCR 실행
    ocr_result = ocr.ocr(resized_img, cls=False)
    
    # 열별로 단어 할당 및 병합
    for column in columns:
        column['words'] = []
        x1, y1, x2, y2 = column['bbox']
        for line in ocr_result:
            for word_info in line:
                bbox, (text, confidence) = word_info
                word_center_x = (bbox[0][0] + bbox[2][0]) / 2
                if x1 <= word_center_x <= x2:
                    column['words'].append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence
                    })
        
        # 열 내의 단어들 병합
        column['merged_text'] = merge_ocr_in_cell(column['words'])
    
    # 원본 크기로 좌표 변환
    for column in columns:
        column['bbox'] = [int(coord / scale) for coord in column['bbox']]
        for word in column['words']:
            word['bbox'] = [[int(x / scale), int(y / scale)] for x, y in word['bbox']]
    
    return original_img, columns

def save_results(original_img, columns, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 원본 이미지 저장
    cv2.imwrite(os.path.join(output_dir, 'original.jpg'), original_img)
    
    # 결과 시각화
    result_img = original_img.copy()
    for i, column in enumerate(columns):
        color = (0, 255, 0)  # 초록색
        x1, y1, x2, y2 = column['bbox']
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result_img, f"Column {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 병합된 텍스트 표시
        lines = column['merged_text'].split('\n')
        for j, line in enumerate(lines):
            cv2.putText(result_img, line, (x1, y1 + 20 + j*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.imwrite(os.path.join(output_dir, 'result.jpg'), result_img)
    
    # JSON 결과 저장
    results = {
        "columns": [
            {
                "column_id": i + 1,
                "bbox": column['bbox'],
                "merged_text": column['merged_text'],
                "words": [
                    {
                        "text": word['text'],
                        "bbox": word['bbox'],
                        "confidence": word['confidence']
                    } for word in column['words']
                ]
            } for i, column in enumerate(columns)
        ]
    }
    
    with open(os.path.join(output_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    img_path = r"20220304_174813461_61050.jpeg"
    output_dir = r"./output"
    
    original_img, columns = process_image(img_path)
    save_results(original_img, columns, output_dir)
