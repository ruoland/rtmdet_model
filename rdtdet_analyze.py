import time
import numpy as np
from rdtdet_log import logger
from datetime import datetime, timedelta
from rdtdet_time_utils import extract_time_info, calculate_end_time
from rdtdet_day_utils import process_day_row
def analyze_and_create_timetable(detected_objects, merged_ocr_results):
    rows = [obj for obj in detected_objects if obj['label'] == 2]
    columns = [obj for obj in detected_objects if obj['label'] == 3]
    cells = [obj for obj in detected_objects if obj['label'] in [0, 4, 5, 6]]
    
    columns.sort(key=lambda x: x['bbox'][0])
    
    # 기본 시간표 구조 생성
    timetable = create_initial_timetable(columns, cells, merged_ocr_results)
    
    # 요일 및 시간 정보 처리
    timetable = process_day_row(timetable, merged_ocr_results)
    
    # 요일 정보 로깅
    logger.info("요일 정보 확인:")
    for i, row in enumerate(timetable):
        for j, cell in enumerate(row):
            if 'day' in cell:
                logger.info(f"행 {i}, 열 {j}: 요일 = {cell['day']}")
            else:
                logger.info(f"행 {i}, 열 {j}: 요일 정보 없음")

    # 셀 내부의 시간 정보 처리 및 consecutive_classes 계산
    timetable = process_timetable_cells(timetable, columns, detected_objects)
    
    # 요일 정보 로깅 (process_timetable_cells 이후)
    logger.info("process_timetable_cells 이후 요일 정보 확인:")
    for i, row in enumerate(timetable):
        for j, cell in enumerate(row):
            if 'day' in cell:
                logger.info(f"행 {i}, 열 {j}: 요일 = {cell['day']}")
            else:
                logger.info(f"행 {i}, 열 {j}: 요일 정보 없음")

    logger.info(f"최종 시간표 구조: {len(timetable)}행 x {len(timetable[0]) if timetable else 0}열")
    header_row, header_col = identify_headers(cells)
    return timetable, header_row, header_col


def create_initial_timetable(columns, cells, merged_ocr_results):
    timetable = []
    header_row, _ = identify_headers(cells)
    
    # 헤더 행 처리
    header_row_info = []
    for i, cell in enumerate(header_row):
        cell_info = process_cell(cell, merged_ocr_results, columns, cells)
        cell_info['is_header'] = True
        cell_info['row'] = 0
        if i == 0:
            cell_info['is_time_header'] = True
        else:
            cell_info['is_day_header'] = True
        header_row_info.append(cell_info)
    timetable.append(header_row_info)

    # 나머지 행 처리
    current_row = []
    current_y = header_row[0]['bbox'][3] + 5
    row_index = 1  # 헤더 행 다음부터 1로 시작
    
    for cell in cells:
        if cell not in header_row and cell['bbox'][1] >= current_y:
            if abs(cell['bbox'][1] - current_y) > 5:  # 새로운 행 시작
                if current_row:
                    timetable.append(current_row)
                    row_index += 1
                current_row = []
                current_y = cell['bbox'][1]
            
            cell_info = process_cell(cell, merged_ocr_results, columns, cells)
            cell_info['row'] = row_index
            
            if cell_info['column'] == 0:
                cell_info['is_time_header'] = True
                time_value = extract_time_from_header(cell_info['content'])
                if time_value:
                    cell_info['start_time'] = time_value
            
            current_row.append(cell_info)
    
    if current_row:
        timetable.append(current_row)
    
    return timetable


def identify_headers(cells, tolerance=5):
    cells.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))  # y좌표 먼저, 그 다음 x좌표로 정렬
    
    # 헤더 행 식별
    
    header_row_y = cells[0]['bbox'][1]
    header_row = [cell for cell in cells if abs(cell['bbox'][1] - header_row_y) <= tolerance]
    
    # 헤더 열 식별
    header_col_x = cells[0]['bbox'][0]
    header_col = [cell for cell in cells if abs(cell['bbox'][0] - header_col_x) <= tolerance]
    
    return header_row, header_col

def process_cell(cell, merged_ocr_results, columns, cells):
    cell_content = get_text_in_bbox(cell['bbox'], merged_ocr_results)
    cell_type = get_cell_type(cell['label'])
    
    column_index = find_column_index(cell, columns)
    
    return {
        'content': cell_content,
        'type': cell_type,
        'bbox': cell['bbox'],
        'column': column_index,  # 열 인덱스를 올바르게 설정
        'consecutive_classes': calculate_cell_span(cell, columns, cells) if cell_type in ['merged_cell', 'merged_overflow_cell'] else 1
    }
def find_column_index(cell, columns):
    cell_center_x = (cell['bbox'][0] + cell['bbox'][2]) / 2
    for i, col in enumerate(columns):
        if col['bbox'][0] <= cell_center_x <= col['bbox'][2]:
            return i
    return -1  # 열을 찾지 못한 경우

def extract_time_from_header(content):
    # 헤더 열의 내용(9, 10, 11 등)을 시간으로 변환
    try:
        hour = int(content)
        if 1 <= hour <= 12:
            return f"{hour:02d}:00"
    except ValueError:
        pass
    return None
from datetime import datetime, timedelta

def process_timetable_cells(timetable, columns, detected_objects):
    for i, row in enumerate(timetable):
        if i == 0:  # 헤더 행 처리
            continue  # 헤더 행은 이미 처리되었으므로 건너뜁니다.

        time_cell = row[0] if row else None
        if time_cell:
            start_time, end_time = extract_time_info(time_cell.get('content', ''))
            time_cell['start_time'] = start_time if start_time else ""
            time_cell['end_time'] = end_time if end_time else ""
            time_cell['is_time_header'] = True
            logger.info(f"시간 열 정보: 시작={start_time}, 종료={end_time}")
        else:
            logger.warning(f"시간 열을 찾을 수 없습니다: {row}")
            start_time, end_time = "", ""

        for j, cell in enumerate(row):
            if j == 0:  # 시간 열은 이미 처리했으므로 건너뜁니다.
                continue

            if not cell.get('content'):
                cell['consecutive_classes'] = 1
                cell['start_time'] = ''
                cell['end_time'] = ''
                continue

            try:
                content = cell.get('content', '').replace('\n', '')
                cell['consecutive_classes'] = calculate_cell_span(cell, columns, detected_objects)
                
                # 요일 정보 확인 (이미 처리되었을 수 있음)
                if 'day' not in cell and j < len(timetable[0]):
                    cell['day'] = timetable[0][j].get('day', '')

                # 셀 내용에서 시간 정보 추출 시도
                cell_time_info = extract_time_info(content)
                if cell_time_info[0]:
                    cell_start_time, cell_end_time = cell_time_info
                else:
                    cell_start_time, cell_end_time = start_time, ""

                # 연속 수업에 따른 종료 시간 계산
                if cell['consecutive_classes'] > 1 and not cell_end_time:
                    end_row_index = min(i + cell['consecutive_classes'], len(timetable) - 1)
                    end_time_cell = timetable[end_row_index][0] if timetable[end_row_index] else None
                    if end_time_cell:
                        _, cell_end_time = extract_time_info(end_time_cell.get('content', ''))

                # 여전히 시간 정보가 없는 경우에만 기본값 사용
                if not cell_start_time:
                    base_time = datetime.strptime("09:00", "%H:%M")
                    cell_start_time = (base_time + timedelta(hours=i-1)).strftime("%H:%M")
                if not cell_end_time:
                    cell_end_time = calculate_end_time(cell_start_time, cell['consecutive_classes'])
                    logger.warning(f"셀 {content}의 종료 시간을 찾지 못했습니다. 계산된 값으로 설정: {cell_end_time}")
                
                cell['start_time'] = cell_start_time
                cell['end_time'] = cell_end_time
                
                logger.info(f"셀 {content} 정보: 타입 = {cell.get('type', 'Unknown')}, 연속 수업 = {cell['consecutive_classes']}, "
                            f"시작 시간 = {cell_start_time}, 종료 시간 = {cell_end_time}, 요일 = {cell.get('day', 'Unknown')}")
            except Exception as e:
                logger.error(f"셀 ({i}, {j}) 처리 중 오류 발생: {str(e)}")
                cell['start_time'] = ''
                cell['end_time'] = ''

    logger.info("시간표 셀 처리 완료")
    return timetable



def calculate_cell_span(cell, columns, cells):
    cell_height = cell['bbox'][3] - cell['bbox'][1]
    column = next((col for col in columns if is_cell_in_grid(cell['bbox'], col['bbox'])), None)
    if column:
        column_height = column['bbox'][3] - column['bbox'][1]
        average_cell_height = column_height / len([c for c in cells if is_cell_in_grid(c['bbox'], column['bbox'])])
        return max(1, round(cell_height / average_cell_height))
    return 1


def is_overlapping(bbox1, bbox2):
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])
def is_cell_in_grid(cell_bbox, grid_bbox, threshold=0.7):
    intersection_area = max(0, min(cell_bbox[2], grid_bbox[2]) - max(cell_bbox[0], grid_bbox[0])) * \
                        max(0, min(cell_bbox[3], grid_bbox[3]) - max(cell_bbox[1], grid_bbox[1]))
    cell_area = (cell_bbox[2] - cell_bbox[0]) * (cell_bbox[3] - cell_bbox[1])
    return intersection_area / cell_area >= threshold

def get_text_in_bbox(bbox, merged_ocr_results):
    text = ""
    for result, _ in merged_ocr_results:
        if is_cell_in_grid(result[0], bbox, threshold=0.1):  # 낮은 임계값 사용
            text += result[1][0] + " "
    return text.strip()

def find_cell_in_bbox(detected_objects, bbox):
    for obj in detected_objects:
        if obj['label'] in [0, 4, 5, 6]:  # cell, merged_cell, overflow_cell, merged_overflow_cell
            if is_cell_in_grid(obj['bbox'], bbox):
                return obj
    return None

def get_cell_type(label):
    cell_types = {0: 'cell', 4: 'merged_cell', 5: 'overflow_cell', 6: 'merged_overflow_cell'}
    return cell_types.get(label, 'unknown_cell')

