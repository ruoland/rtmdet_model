import cv2
import os
import json
from rdtdet_log import logger

def save_cropped_images(original_img, detected_objects, output_dir):
    # 디렉토리 생성
    cols_dir = os.path.join(output_dir, 'columns')
    cells_dir = os.path.join(output_dir, 'cells')
    os.makedirs(cols_dir, exist_ok=True)
    os.makedirs(cells_dir, exist_ok=True)

    # 열 정보 추출 및 정렬
    columns = [obj for obj in detected_objects if obj['label'] == 3]
    columns.sort(key=lambda x: x['bbox'][0])  # x1 좌표로 정렬

    # 열 이미지 저장
    for i, col in enumerate(columns):
        x1, y1, x2, y2 = map(int, col['bbox'])
        cropped_col = original_img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(cols_dir, f'column_{i:03d}.png'), cropped_col)

    # 셀 정보 추출 및 정렬
    cell_types = {0: 'cell', 4: 'merged_cell', 5: 'overflow_cell', 6: 'merged_overflow_cell'}
    cells = [obj for obj in detected_objects if obj['label'] in cell_types]
    
    # 셀을 열의 순서와 y 좌표에 따라 정렬
    def cell_sort_key(cell):
        if columns:
            col_index = min(range(len(columns)), key=lambda i: abs(cell['bbox'][0] - columns[i]['bbox'][0]))
        col_index = 0
        return (col_index, cell['bbox'][1])  # 열 인덱스와 y 좌표로 정렬

    cells.sort(key=cell_sort_key)

    # 셀 이미지 저장
    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = map(int, cell['bbox'])
        cropped_cell = original_img[y1:y2, x1:x2]
        cell_type = cell_types.get(cell['label'], 'unknown_cell')
        
        # 셀이 속한 열 인덱스 찾기
        
        col_index = min(range(len(columns)), key=lambda i: abs(cell['bbox'][0] - columns[i]['bbox'][0]))
        if not columns:
            col_index = 0
        # 파일명 생성: 열번호_셀타입_순서
        filename = f'col{col_index:03d}_{cell_type}_{i:03d}.png'
        cv2.imwrite(os.path.join(cells_dir, filename), cropped_cell)

    logger.info(f"열 {len(columns)}개, 셀 {len(cells)}개의 이미지를 저장했습니다.")

def save_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"데이터를 {output_file}에 저장했습니다.")
