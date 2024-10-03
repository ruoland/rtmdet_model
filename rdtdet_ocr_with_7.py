import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from rdtdet_log import logger
from model_detection import initialize_model, detect_objects
from rdtdet_merge import merge_ocr_results, organize_cells_into_grid, merge_new_cells
from rdtdet_io import save_cropped_images
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 모델 초기화
ocr = PaddleOCR(use_angle_cls=False, lang='korean')
rtmdet = initialize_model()
IMAGE_THRESHOLD = 0.5

def process_image(image_path, target_size=1000):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"{image_path}에서 이미지를 불러오는데 실패했습니다.")
    
    # 이미지 리사이즈
    height, width = original_img.shape[:2]
    scale = target_size / max(height, width)
    new_size = (int(width * scale), int(height * scale))
    resized_img = cv2.resize(original_img, new_size)
    
    # 표 모델 실행 및 객체 감지
    detected_objects = detect_objects(rtmdet, resized_img, IMAGE_THRESHOLD)
    
    # OCR 실행
    ocr_result = ocr.ocr(resized_img, cls=False)
    
    # 원본 크기로 좌표 변환
    for obj in detected_objects:
        obj['bbox'] = [int(coord / scale) for coord in obj['bbox']]
    
    for i in range(len(ocr_result)):
        for j in range(len(ocr_result[i])):
            box = ocr_result[i][j][0]
            ocr_result[i][j] = (
                [[int(x / scale), int(y / scale)] for x, y in box],
                ocr_result[i][j][1]
            )
    
    logger.info(f"{len(detected_objects)}개의 객체와 {len(ocr_result[0])}개의 OCR 결과를 감지했습니다.")
    return original_img, detected_objects, ocr_result[0]


def print_formatted_table_structure(table_structure):
    for i, row in enumerate(table_structure):
        for j, cell in enumerate(row):
            if cell:
                content = cell.get('text', '')[:20]  # 첫 20자만 표시
                print(f"{i+1}행 {j+1}열: {content}")
            else:
                print(f"{i+1}행 {j+1}열: 빈 셀")
        print()  # 행 사이에 빈 줄 추가
import json
import uuid
from collections import defaultdict

def create_grid_from_rows_and_columns(rows, columns, image_height, image_width):
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])
    
    row_boundaries = [0] + [row['bbox'][3] for row in rows] + [image_height]
    col_boundaries = [0] + [col['bbox'][2] for col in columns] + [image_width]
    
    return row_boundaries, col_boundaries

def assign_cells_to_grid(cells, row_boundaries, col_boundaries):
    grid = defaultdict(dict)
    for cell in cells:
        x1, y1, x2, y2 = cell['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        row = next(i for i, boundary in enumerate(row_boundaries[1:]) if center_y < boundary) - 1
        col = next(i for i, boundary in enumerate(col_boundaries[1:]) if center_x < boundary) - 1
        
        grid[row][col] = cell
    
    return grid
def create_timetable_entries(grid, row_boundaries, col_boundaries, merged_ocr_results):
    entries = []
    days_of_week = ['월', '화', '수', '목', '금', '토', '일']
    
    for row in range(1, len(row_boundaries) - 1):  # Skip header row
        for col in range(1, len(col_boundaries) - 1):  # Skip time column
            cell = grid.get(row, {}).get(col)
            if cell:
                start_time = grid.get(row, {}).get(0, {}).get('text', '')
                
                # merged_ocr_results에서 해당 셀의 텍스트 찾기
                cell_text = cell.get('text', '')
                for ocr_result, _ in merged_ocr_results:
                    ocr_box, (ocr_text, ocr_confidence) = ocr_result
                    if is_inside(cell['bbox'], ocr_box):
                        cell_text = ocr_text
                        break
                
                entry = {
                    "id": str(uuid.uuid4()),
                    "day_of_week": days_of_week[col-1] if col-1 < len(days_of_week) else "",
                    "period": str(row),
                    "start_time": start_time,
                    "end_time": "",  # 종료 시간 정보가 없으므로 비워둠
                    "subject": cell_text,
                    "row": row,
                    "column": col,
                    "consecutive_classes": calculate_consecutive_classes(cell, row_boundaries)
                }
                entries.append(entry)
    
    return entries

def is_inside(outer_box, inner_box):
    return (outer_box[0] <= inner_box[0] and outer_box[1] <= inner_box[1] and
            outer_box[2] >= inner_box[2] and outer_box[3] >= inner_box[3])

def calculate_consecutive_classes(cell, row_boundaries):
    if not cell or 'bbox' not in cell:
        return 1
    cell_height = cell['bbox'][3] - cell['bbox'][1]
    avg_row_height = (row_boundaries[-1] - row_boundaries[0]) / (len(row_boundaries) - 1)
    return max(1, round(cell_height / avg_row_height))

def print_merged_ocr_results(merged_ocr_results):
    print("Merged OCR Results:")
    for i, (result, merge_info) in enumerate(merged_ocr_results):
        box, (text, confidence) = result
        print(f"Result {i+1}:")
        print(f"  Text: {text}")
        print(f"  Confidence: {confidence}")
        print(f"  Bounding Box: {box}")
        print(f"  Merge Info: {merge_info}")
        print()
def save_timetable_to_json(timetable_entries, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(timetable_entries, f, ensure_ascii=False, indent=2)
    logger.info(f"시간표 데이터를 {output_file}에 저장했습니다.")
# 메인 실행 부분
if __name__ == "__main__":
    img_path = r"original.png"
    output_dir = "ocr_cell"
    os.makedirs(output_dir, exist_ok=True)

    original_img, detected_objects, ocr_results = process_image(img_path)
    
    # 행, 열, 셀 이미지 저장
    save_cropped_images(original_img, detected_objects, output_dir)
    logger.info("OCR 결과 병합을 시작합니다.")
    detected_cells = [obj for obj in detected_objects if obj['label'] == 0]  # 셀만 선택
    merged_ocr_results = merge_ocr_results(ocr_results, detected_cells)
    
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
    with open(os.path.join(output_dir, "merged_ocr_results.json"), 'w', encoding='utf-8') as f:
        json.dump(merged_ocr_json, f, ensure_ascii=False, indent=2)
    
    # 시간표 분석 및 생성
    timetable = analyze_and_create_timetable(detected_objects, merged_ocr_results)
    # 시간표를 JSON 형식으로 변환
    timetable_entries = []
    for i, row in enumerate(timetable[1:], start=1):  # 첫 번째 행(요일)은 건너뜁니다
        for j, subject in enumerate(row[1:], start=1):  # 첫 번째 열(시간)은 건너뜁니다
            if subject:
                entry = {
                    "id": str(uuid.uuid4()),
                    "day_of_week": timetable[0][j],
                    "period": str(i),
                    "start_time": row[0],  # 첫 번째 열의 값을 시작 시간으로 사용
                    "end_time": "",
                    "subject": subject,
                    "row": i,
                    "column": j,
                    "consecutive_classes": 1  # 기본값, 필요시 나중에 조정
                }
                timetable_entries.append(entry)
    
    # JSON으로 저장
    json_output_file = os.path.join(output_dir, "timetable.json")
    save_timetable_to_json(timetable_entries, json_output_file)
    
    # 결과 표시
    display_results(original_img, detected_objects, merged_ocr_results)
    
    logger.info(f"원본 객체 {len(detected_objects)}개를 감지했습니다.")
    logger.info(f"셀 {len(detected_cells)}개를 감지했습니다.")
    logger.info(f"병합된 OCR 결과 {len(merged_ocr_results)}개를 생성했습니다.")
    logger.info(f"시간표 항목 {len(timetable_entries)}개를 생성했습니다.")