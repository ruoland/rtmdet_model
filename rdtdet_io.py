import cv2
import os
import numpy as np
import cv2
import os
import json, logging
import cv2
import os
from rdtdet_log import logger

def save_cropped_images(original_img, detected_objects, output_dir):
    # 디렉토리 생성
    rows_dir = os.path.join(output_dir, 'rows')
    cols_dir = os.path.join(output_dir, 'columns')
    cells_dir = os.path.join(output_dir, 'cells')
    os.makedirs(rows_dir, exist_ok=True)
    os.makedirs(cols_dir, exist_ok=True)
    os.makedirs(cells_dir, exist_ok=True)

    # 행 이미지 저장
    rows = [obj for obj in detected_objects if obj['label'] == 2]
    for i, row in enumerate(rows):
        x1, y1, x2, y2 = map(int, row['bbox'])
        cropped_row = original_img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(rows_dir, f'row_{i}.png'), cropped_row)

    # 열 이미지 저장
    columns = [obj for obj in detected_objects if obj['label'] == 3]
    for i, col in enumerate(columns):
        x1, y1, x2, y2 = map(int, col['bbox'])
        cropped_col = original_img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(cols_dir, f'column_{i}.png'), cropped_col)

    # 셀 이미지 저장
    cell_types = {0: 'cell', 4: 'merged_cell', 5: 'overflow_cell', 6: 'merged_overflow_cell'}
    cells = [obj for obj in detected_objects if obj['label'] in cell_types]
    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = map(int, cell['bbox'])
        cropped_cell = original_img[y1:y2, x1:x2]
        cell_type = cell_types.get(cell['label'], 'unknown_cell')
        cv2.imwrite(os.path.join(cells_dir, f'{cell_type}_{i}.png'), cropped_cell)

    logger.info(f"행 {len(rows)}개, 열 {len(columns)}개, 셀 {len(cells)}개의 이미지를 저장했습니다.")
def save_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"데이터를 {output_file}에 저장했습니다.")