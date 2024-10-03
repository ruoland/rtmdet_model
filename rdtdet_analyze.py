import numpy as np
from rdtdet_log import logger
import math

from rdtdet_time_utils import process_time_column, extract_time_info, calculate_end_time
from rdtdet_day_utils import process_day_row, get_day_of_week

def analyze_and_create_timetable(detected_objects, merged_ocr_results):
    rows = [obj for obj in detected_objects if obj['label'] == 2]
    columns = [obj for obj in detected_objects if obj['label'] == 3]
    
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])
    
    # 행과 열의 크기 계산 및 헤더 식별
    row_heights = [row['bbox'][3] - row['bbox'][1] for row in rows]
    col_widths = [col['bbox'][2] - col['bbox'][0] for col in columns]
    median_row_height = np.median(row_heights)
    median_col_width = np.median(col_widths)
    header_row = row_heights[0] < 0.7 * median_row_height if row_heights else False
    header_col = col_widths[0] < 0.7 * median_col_width if col_widths else False
    
    logger.info(f"헤더 행 감지: {header_row}, 헤더 열 감지: {header_col}")
    
    timetable = create_initial_timetable(rows, columns, detected_objects, merged_ocr_results)
    
    # 요일 및 시간 정보 처리
    timetable = process_day_row(timetable)
    timetable = process_time_column(timetable)
    
    # 셀 내부의 시간 정보 처리 및 consecutive_classes 계산
    process_cell_info(timetable, rows)
    
    logger.info(f"최종 시간표 구조: {len(timetable)}행 x {len(timetable[0]) if timetable else 0}열")
    return timetable, header_row, header_col

def create_initial_timetable(rows, columns, detected_objects, merged_ocr_results):
    timetable = []
    for row_index, row in enumerate(rows):
        row_content = []
        for col_index, column in enumerate(columns):
            cell_bbox = [max(row['bbox'][0], column['bbox'][0]),
                         row['bbox'][1],
                         min(row['bbox'][2], column['bbox'][2]),
                         row['bbox'][3]]
            
            cell_obj = find_cell_in_bbox(detected_objects, cell_bbox)
            
            if cell_obj:
                cell_type = get_cell_type(cell_obj['label'])
                cell_content = get_text_in_bbox(cell_obj['bbox'], merged_ocr_results)
                
                cell_info = {
                    'content': cell_content,
                    'type': cell_type,
                    'bbox': cell_obj['bbox'],
                    'row': row_index,
                    'column': col_index
                }
                
                logger.info(f"셀 정보 ({row_index+1}행 {col_index+1}열): {cell_info}")
            else:
                cell_info = {'content': '', 'type': 'empty'}
            
            row_content.append(cell_info)
        timetable.append(row_content)
    return timetable
def process_cell_info(timetable, rows):
    for i, row in enumerate(timetable):
        for j, cell in enumerate(row):
            if 'bbox' not in cell:
                logger.warning(f"셀 ({i}, {j})에 bbox 정보가 없습니다. 건너뜁니다.")
                continue

            try:
                cell['consecutive_classes'], cell['row_ratios'] = calculate_cell_span_and_ratios(cell, rows)
                
                time_info = extract_time_info(cell['content'])
                start_time = time_info if time_info else timetable[i][0].get('time', '')
                end_time = calculate_cell_end_time(timetable, i, cell)
                
                cell['start_time'] = start_time
                cell['end_time'] = end_time
                
                logger.info(f"셀 ({i}, {j}) 정보: 타입 = {cell['type']}, 연속 수업 = {cell['consecutive_classes']}, "
                            f"행 비율 = {cell['row_ratios']}, 시작 시간 = {start_time}, 종료 시간 = {end_time}")
            except Exception as e:
                logger.error(f"셀 ({i}, {j}) 처리 중 오류 발생: {str(e)}")


def calculate_cell_span_and_ratios(cell, rows):
    cell_top = cell['bbox'][1]
    cell_bottom = cell['bbox'][3]
    
    span = 0
    ratios = []
    epsilon = 2  # 약간의 여유를 두기 위한 값 (픽셀 단위)

    for row in rows:
        row_top = row['bbox'][1]
        row_bottom = row['bbox'][3]
        
        if cell_bottom < row_top - epsilon:
            break
        
        if cell_top > row_bottom + epsilon:
            continue
        
        overlap = max(0, min(cell_bottom, row_bottom) - max(cell_top, row_top))
        row_height = row_bottom - row_top
        
        if overlap > 0:
            span += 1
            ratio = overlap / row_height
            ratios.append(ratio)

    logger.info(f"셀 span 계산: span = {span}, ratios = {ratios}")
    return span, ratios
def calculate_cell_end_time(timetable, start_row, cell):
    start_time = timetable[start_row][0].get('time', '')
    if not start_time:
        logger.warning(f"행 {start_row}에 시작 시간 정보가 없습니다.")
        return ''
    
    start_hours, start_minutes = map(int, start_time.split(':'))
    total_minutes = 0
    
    for ratio in cell['row_ratios']:
        total_minutes += int(60 * ratio)
    
    # 총 시간이 0분이면 최소 1분으로 설정
    if total_minutes == 0:
        total_minutes = 1
    
    end_hours = (start_hours + total_minutes // 60) % 24
    end_minutes = (start_minutes + total_minutes % 60) % 60
    
    logger.info(f"셀 시간 계산: 시작 = {start_time}, 총 분 = {total_minutes}, 종료 = {end_hours:02d}:{end_minutes:02d}")
    return f"{end_hours:02d}:{end_minutes:02d}"


def find_cell_in_bbox(detected_objects, bbox):
    for obj in detected_objects:
        if obj['label'] in [0, 4, 5, 6]:  # cell, merged_cell, overflow_cell, merged_overflow_cell
            if is_cell_in_grid(obj['bbox'], bbox):
                return obj
    return None

def get_cell_type(label):
    cell_types = {0: 'cell', 4: 'merged_cell', 5: 'overflow_cell', 6: 'merged_overflow_cell'}
    return cell_types.get(label, 'unknown_cell')


def is_overlapping(bbox1, bbox2):
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])
def is_cell_in_grid(cell_bbox, grid_bbox, threshold=0.7):
    intersection_area = max(0, min(cell_bbox[2], grid_bbox[2]) - max(cell_bbox[0], grid_bbox[0])) * \
                        max(0, min(cell_bbox[3], grid_bbox[3]) - max(cell_bbox[1], grid_bbox[1]))
    cell_area = (cell_bbox[2] - cell_bbox[0]) * (cell_bbox[3] - cell_bbox[1])
    result = intersection_area / cell_area >= threshold
    return result

def get_text_in_bbox(bbox, merged_ocr_results):
    text = ""
    for result, _ in merged_ocr_results:
        if is_overlapping(bbox, result[0]):
            text += result[1][0] + " "
    return text.strip()
