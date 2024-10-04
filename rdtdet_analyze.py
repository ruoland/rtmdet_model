import numpy as np
from rdtdet_log import logger
import math
from datetime import datetime, timedelta
from rdtdet_time_utils import process_time_column, extract_time_info, calculate_end_time
from rdtdet_day_utils import process_day_row, get_day_of_week, is_day_cell
import re
from rdtdet_time_utils import is_time_format
def analyze_and_create_timetable(detected_objects, merged_ocr_results):
    rows = [obj for obj in detected_objects if obj['label'] == 2]
    columns = [obj for obj in detected_objects if obj['label'] == 3]
    
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])
    
    row_heights = [row['bbox'][3] - row['bbox'][1] for row in rows]
    col_widths = [col['bbox'][2] - col['bbox'][0] for col in columns]
    median_row_height = np.median(row_heights)
    median_col_width = np.median(col_widths)
    
    header_row = -1
    header_col = -1
    days_of_week = []
    start_time = None
    end_time = None

    # 헤더 행 감지 및 요일 정보 추출
    for i, row in enumerate(rows):
        row_content = get_text_in_bbox(row['bbox'], merged_ocr_results)
        if row_heights[i] < 0.7 * median_row_height or any(is_day_cell(cell) for cell in row_content.split()):
            header_row = i
            days_of_week = [get_day_of_week(cell) for cell in row_content.split() if is_day_cell(cell)]
            break
    
    # 헤더 열 감지 및 시간 정보 추출
    for i, col in enumerate(columns):
        col_content = get_text_in_bbox(col['bbox'], merged_ocr_results)
        if col_widths[i] < 0.7 * median_col_width or is_time_format(col_content):
            header_col = i
            times = extract_time_info(col_content)
            if times:
                start_time = times[0] if len(times) > 0 else None
                end_time = times[-1] if len(times) > 1 else None
            break
    
    logger.info(f"헤더 행 감지: {header_row}, 헤더 열 감지: {header_col}")
    logger.info(f"요일 정보: {days_of_week}")
    logger.info(f"시작 시간: {start_time}, 종료 시간: {end_time}")
    
    timetable = create_initial_timetable(rows, columns, detected_objects, merged_ocr_results, 
                                         header_row, header_col, days_of_week, start_time, end_time)
    
    # 요일 및 시간 정보 처리
    timetable = process_day_row(timetable, merged_ocr_results)
    timetable = process_time_column(timetable)
    
    # 셀 내부의 시간 정보 처리 및 consecutive_classes 계산
    process_cell_info(timetable, rows)
    
    logger.info(f"최종 시간표 구조: {len(timetable)}행 x {len(timetable[0]) if timetable else 0}열")
    return timetable, header_row != -1, header_col != -1

def create_initial_timetable(rows, columns, detected_objects, merged_ocr_results, 
                             header_row, header_col, days_of_week, start_time, end_time):
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
                
                # 헤더 행이나 열에 대한 추가 정보
                if row_index == header_row:
                    cell_info['is_day_header'] = True
                    if col_index < len(days_of_week):
                        cell_info['day'] = days_of_week[col_index]
                if col_index == header_col:
                    cell_info['is_time_header'] = True
                    if row_index == 0:
                        cell_info['start_time'] = start_time
                    elif row_index == len(rows) - 1:
                        cell_info['end_time'] = end_time
                
                logger.info(f"셀 정보 ({row_index+1}행 {col_index+1}열): {cell_info}")
            else:
                cell_info = {
                    'content': '',
                    'type': 'empty',
                    'row': row_index,
                    'column': col_index
                }
            
            row_content.append(cell_info)
        timetable.append(row_content)
    return timetable
def process_cell_info(timetable, rows):
    for i, row in enumerate(timetable):
        for j, cell in enumerate(row):
            # 헤더 행(요일) 또는 헤더 열(시간)인 경우 별도 처리
            if cell.get('is_day_header') or cell.get('is_time_header'):
                cell['consecutive_classes'] = 1
                cell['row_ratios'] = [1.0]
                if cell.get('is_time_header'):
                    cell['start_time'] = cell.get('start_time', '')
                    cell['end_time'] = cell.get('end_time', '')
                continue

            if 'bbox' not in cell or not cell['content']:
                cell['consecutive_classes'] = 1
                cell['row_ratios'] = [1.0]
                cell['start_time'] = ''
                cell['end_time'] = ''
                continue

            try:
                content = cell['content'].replace('\n', '')
                cell['consecutive_classes'], cell['row_ratios'] = calculate_cell_span_and_ratios(cell, rows)
                
                time_info = extract_time_info(content)
                start_time = time_info[0] if time_info else timetable[i][0].get('start_time', '')
                
                # 시작 시간이 없으면 9시를 기준으로 설정
                if not start_time:
                    base_time = datetime.strptime("09:00", "%H:%M")
                    start_time = (base_time + timedelta(hours=i)).strftime("%H:%M")
                    logger.warning(f"셀 {content}의 시작 시간을 찾지 못했습니다. 기본값으로 설정: {start_time}")
                
                end_time = calculate_end_time(start_time, cell['consecutive_classes'])
                
                cell['start_time'] = start_time
                cell['end_time'] = end_time
                
                logger.info(f"셀 {content} 정보: 타입 = {cell['type']}, 연속 수업 = {cell['consecutive_classes']}, "
                            f"행 비율 = {cell['row_ratios']}, 시작 시간 = {start_time}, 종료 시간 = {end_time}")
            except Exception as e:
                logger.error(f"셀 ({i}, {j}) 처리 중 오류 발생: {str(e)}")
                cell['start_time'] = ''
                cell['end_time'] = ''



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

    
    return span, ratios


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
