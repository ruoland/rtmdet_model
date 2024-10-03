import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from rdtdet_log import logger
from model_detection import initialize_model, detect_objects
import ujson as json
from datetime import datetime, timedelta
import re
from sklearn.cluster import DBSCAN
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 모델 초기화
ocr = PaddleOCR(use_angle_cls=False, lang='korean')
config_file = 'mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
checkpoint_file = r"c:\Users\opron\Downloads\epoch_18.pth"
rtmdet = initialize_model(config_file, checkpoint_file)
IMAGE_THRESHOLD = 0.8

class Cell:
    def __init__(self, bbox, label):
        self.bbox = bbox
        self.label = label
        self.text = ""
        self.ocr_results = []

class Table:
    def __init__(self, bbox):
        self.bbox = bbox
        self.rows = []
        self.columns = []
        self.cells = []
        self.titles = []

    def add_row(self, row_bbox):
        self.rows.append(row_bbox)

    def add_column(self, col_bbox):
        self.columns.append(col_bbox)

    def add_cell(self, cell):
        self.cells.append(cell)

    def add_title(self, title_text, title_bbox):
        self.titles.append((title_text, title_bbox))
def draw_category_image(image, objects, category):
    result_image = image.copy()
    for obj in objects:
        if obj['label'] == category:
            cv2.rectangle(result_image, 
                          (int(obj['bbox'][0]), int(obj['bbox'][1])), 
                          (int(obj['bbox'][2]), int(obj['bbox'][3])), 
                          (0, 255, 0), 2)
    return result_image
def process_table_detection(image_path):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"{image_path}에서 이미지를 불러오는데 실패했습니다.")
    
    detected_objects = detect_objects(rtmdet, original_img, IMAGE_THRESHOLD)
    
    tables = []
    for obj in detected_objects:
        if obj['label'] == 1:  # table
            table = Table(obj['bbox'])
            tables.append(table)
        elif obj['label'] == 2:  # row
            if tables:
                tables[-1].add_row(obj['bbox'])
        elif obj['label'] == 3:  # column
            if tables:
                tables[-1].add_column(obj['bbox'])
        elif obj['label'] in [0, 4, 5, 6]:  # cell, merged_cell, overflow_cell, merged_overflow_cell
            if tables:
                cell = Cell(obj['bbox'], obj['label'])
                tables[-1].add_cell(cell)
    
    return original_img, tables

def process_ocr(image, tables):
    ocr_result = ocr.ocr(image, cls=False)
    
    for table in tables:
        for ocr_box, (text, confidence) in ocr_result[0]:
            ocr_bbox = [ocr_box[0][0], ocr_box[0][1], ocr_box[2][0], ocr_box[2][1]]
            
            if is_potential_title(ocr_bbox, table.bbox):
                table.add_title(text, ocr_bbox)
                continue
            
            for cell in table.cells:
                if is_inside(ocr_bbox, cell.bbox):
                    cell.ocr_results.append((text, confidence, ocr_bbox))
    
    for table in tables:
        for cell in table.cells:
            cell.text = merge_ocr_in_cell(cell.ocr_results)

def is_potential_title(ocr_bbox, table_bbox):
    ocr_center_y = (ocr_bbox[1] + ocr_bbox[3]) / 2
    table_top = table_bbox[1]
    
    # OCR 결과가 표의 상단에서 표 높이의 10% 이내에 있는지 확인
    if ocr_center_y < table_top + (table_bbox[3] - table_bbox[1]) * 0.1:
        return True
    return False

def is_inside(ocr_bbox, cell_bbox):
    ocr_center_x = (ocr_bbox[0] + ocr_bbox[2]) / 2
    ocr_center_y = (ocr_bbox[1] + ocr_bbox[3]) / 2
    
    return (cell_bbox[0] <= ocr_center_x <= cell_bbox[2] and
            cell_bbox[1] <= ocr_center_y <= cell_bbox[3])

def merge_ocr_in_cell(ocr_results):
    if not ocr_results:
        return ""
    
    # y 좌표를 기준으로 정렬
    sorted_results = sorted(ocr_results, key=lambda x: x[2][1])
    
    merged_text = ""
    prev_y = None
    for text, _, bbox in sorted_results:
        if prev_y is not None and bbox[1] - prev_y > 20:  # 줄바꿈 기준 (20픽셀)
            merged_text += "\n"
        merged_text += text + " "
        prev_y = bbox[3]
    
    return merged_text.strip()

def merge_tables(tables):
    if len(tables) <= 1:
        return tables

    merged_table = Table(bbox=[
        min(table.bbox[0] for table in tables),
        min(table.bbox[1] for table in tables),
        max(table.bbox[2] for table in tables),
        max(table.bbox[3] for table in tables)
    ])

    for table in tables:
        merged_table.rows.extend(table.rows)
        merged_table.columns.extend(table.columns)
        merged_table.cells.extend(table.cells)
        merged_table.titles.extend(table.titles)

    return [merged_table]

import numpy as np
from sklearn.cluster import DBSCAN

def cluster_lines(lines, axis=0, eps=10):
    if not lines:
        return []
    
    coordinates = np.array([[line[axis], line[axis+2]] for line in lines])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(coordinates)
    
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(lines[i])
    
    return [np.mean(cluster, axis=0).tolist() for cluster in clusters.values()]
def extract_time_info(cells):
    time_pattern = re.compile(r'(\d{1,2}):(\d{2})')
    time_info = []
    
    for i, cell in enumerate(cells):
        if cell.bbox[0] == min(c.bbox[0] for c in cells):  # 첫 번째 열
            match = time_pattern.search(cell.text)
            if match:
                hour, minute = map(int, match.groups())
                time_info.append((i, f"{hour:02d}:{minute:02d}"))
    
    return time_info
def extract_row_col_info(cells, rows, columns):
    first_row = [cell for cell in cells if cell.bbox[1] <= rows[0][1]]
    first_col = [cell for cell in cells if cell.bbox[0] <= columns[0][0]]
    
    print("1행 인식 결과:")
    for cell in first_row:
        print(f"  - {cell.text}")
    
    print("\n1열 인식 결과:")
    for cell in first_col:
        print(f"  - {cell.text}")
    
    return first_row, first_col
def cluster_lines(lines, axis=0, eps=10):
    if not lines:
        return []
    
    coordinates = np.array([[line[axis], line[axis+2]] for line in lines])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(coordinates)
    
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(lines[i])
    
    return [np.mean(cluster, axis=0).tolist() for cluster in clusters.values()]
def parse_timetable(table, ocr_results):
    rows = table.rows
    columns = table.columns
    
    timetable_grid = [[[] for _ in range(len(columns) + 1)] for _ in range(len(rows) + 1)]
    
    for bbox, (text, confidence) in ocr_results:
        row_index, col_index = assign_cell_to_row_col(bbox, rows, columns)
        if row_index is not None and col_index is not None:
            timetable_grid[row_index][col_index].append(text)
    
    return timetable_grid, rows, columns

def assign_cell_to_row_col(cell_bbox, rows, columns):
    # PaddleOCR bbox 형식: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    cell_center_y = (cell_bbox[0][1] + cell_bbox[2][1]) / 2
    cell_center_x = (cell_bbox[0][0] + cell_bbox[2][0]) / 2
    
    row_index = next((i for i, row in enumerate(rows) if row[1] <= cell_center_y <= row[3]), None)
    col_index = next((i for i, col in enumerate(columns) if col[0] <= cell_center_x <= col[2]), None)
    
    return row_index, col_index

def save_results(original_img, tables, ocr_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 원본 이미지 저장
    cv2.imwrite(os.path.join(output_dir, 'original.jpg'), original_img)
    
    # 테이블 병합
    merged_tables = merge_tables(tables)
    
    # 시간표 파싱
    timetable_grid, rows, columns = parse_timetable(merged_tables[0], ocr_results)
    
    # 결과 JSON 파일 저장
    results = {
        'timetableData': timetable_grid
    }
    
    with open(os.path.join(output_dir, 'timetable_data.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 인식 결과를 원본 이미지에 그리기
    result_image = original_img.copy()
    for table in merged_tables:
        # 테이블 경계 그리기
        cv2.rectangle(result_image, (int(table.bbox[0]), int(table.bbox[1])), 
                      (int(table.bbox[2]), int(table.bbox[3])), (255, 0, 0), 2)
        
        # 행 그리기
        for row in table.rows:
            cv2.line(result_image, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (0, 255, 0), 1)
        
        # 열 그리기
        for col in table.columns:
            cv2.line(result_image, (int(col[0]), int(col[1])), (int(col[2]), int(col[3])), (0, 0, 255), 1)
        
        # 셀 그리기
        for cell in table.cells:
            cv2.rectangle(result_image, (int(cell.bbox[0]), int(cell.bbox[1])), 
                          (int(cell.bbox[2]), int(cell.bbox[3])), (0, 255, 255), 1)
    
    cv2.imwrite(os.path.join(output_dir, 'result_with_detection.jpg'), result_image)

    # 콘솔에 결과 출력
    print("Timetable Grid:")
    for i, row in enumerate(timetable_grid):
        non_empty_cells = [cell for cell in row if cell]
        print(f"Row {i}:")
        print(f"  Number of cells: {len(non_empty_cells)}")
        print("  Cell contents:")
        for j, cell in enumerate(row):
            if cell:
                print(f"    Column {j}: {', '.join(cell)}")
        print()

# 메인 함수 수정
if __name__ == "__main__":
    img_path = r"jeunjae.jpg"
    output_dir = r"C:\project\output_results"
    
    original_img, tables = process_table_detection(img_path)
    ocr_results = ocr.ocr(original_img, cls=False)[0]
    
    save_results(original_img, tables, ocr_results, output_dir)
    
    print(f"Results saved in {output_dir}")
