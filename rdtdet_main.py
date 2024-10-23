import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from rdtdet_log import logger
from model_detection import initialize_model, detect_objects
from rdtdet_merge import merge_ocr_results
from rdtdet_io import save_cropped_images, save_json
from rdtdet_process import process_image
from rdtdet_analyze import analyze_and_create_timetable
from rdtdet_display import display_results, print_merged_ocr_results
from rdtdet_day_utils import get_day_of_week
import uuid, time

# 모델 초기화
rtmdet = initialize_model()
IMAGE_THRESHOLD = 0.6
if __name__ == "__main__":
    start_time = time.time()

    img_path = r"joeun.jpg"
    output_dir = "ocr_cell"
    os.makedirs(output_dir, exist_ok=True)

    original_img, detected_objects, ocr_results = process_image(img_path, rtmdet, IMAGE_THRESHOLD)
    unique_labels = set(obj['label'] for obj in detected_objects)
    logger.info(f"감지된 객체의 고유 라벨: {unique_labels}")
    # 행, 열, 셀 이미지 저장
    save_cropped_images(original_img, detected_objects, output_dir)
    
    logger.info("OCR 글자들의 병합을 시작합니다.")
    
    merged_ocr_results = merge_ocr_results(ocr_results)
    
    # merged_ocr_results 출력
    print_merged_ocr_results(merged_ocr_results)
    
    # merged_ocr_results를 JSON 파일로 저장
    merged_ocr_json = [
        {
            "text": result[1][0],
            "confidence": result[1][1],
            "bbox": result[0],
            "merge_info": merge_info
        }
        for result, merge_info in merged_ocr_results
    ]
    save_json(merged_ocr_json, os.path.join(output_dir, "merged_ocr_results.json"))
    
    # 시간표 분석 및 생성

    timetable, header_row, header_col = analyze_and_create_timetable(detected_objects, merged_ocr_results)

        # timetable_entries 생성 부분
    timetable_entries = []
    for i, row in enumerate(timetable):
        for j, cell in enumerate(row):
            if cell['content']:
                is_day_header = cell.get('is_day_header', False)
                
                entry = {
                    "id": str(uuid.uuid4()),
                    "day_of_week": cell.get('day', ''),
                    "period": str(i),
                    "start_time": cell.get('start_time', ''),
                    "end_time": cell.get('end_time', ''),
                    "subject": cell['content'],
                    "row": i,
                    "column": cell['column'],  # 'column' 값을 cell에서 가져옴
                    "consecutive_classes": cell.get('consecutive_classes', 1),
                    "cell_type": cell['type']
                }
                if is_day_header:
                    entry["is_day_header"] = True
                    entry["ocr_confidence"] = cell.get('ocr_confidence', 0.0)
                    entry['subject'] = entry['day_of_week']

                timetable_entries.append(entry)

    # 요일 정보 확인을 위한 추가 로깅
    for entry in timetable_entries:
        logger.info(f"엔트리: 행={entry['row']}, 열={entry['column']}, 요일={entry['day_of_week']}, 내용={entry['subject']}")

    # JSON으로 저장
    save_json(timetable_entries, os.path.join(output_dir, "timetable.json"))
        
    # 결과 표시
    display_results(original_img, detected_objects, merged_ocr_results)
    cell_labels = [0, 4, 5, 6]  # 일반 셀, 병합된 셀, 오버플로우된 셀, 병합+오버플로우된 셀
    detected_cells = [obj for obj in detected_objects if obj['label'] in cell_labels]
    logger.info(f"원본 객체 {len(detected_objects)}개를 감지했습니다.")
    logger.info(f"셀 {len(detected_cells)}개를 감지했습니다.")
    logger.info(f"병합된 OCR 결과 {len(merged_ocr_results)}개를 생성했습니다.")
    logger.info(f"시간표 항목 {len(timetable_entries)}개를 생성했습니다.")
    # 측정하려는 코드
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"실행 시간: {execution_time} 초")
