import re
from rdtdet_log import logger
def get_day_of_week(text):
    if isinstance(text, int):
        # 정수를 요일로 변환 (예: 0 -> '월', 1 -> '화' 등)
        days_kr = ['월', '화', '수', '목', '금', '토', '일']
        return days_kr[text % 7]
    
    days_kr = ['월', '화', '수', '목', '금', '토', '일']
    days_kr_full = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    days_en = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    days_en_full = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    text = str(text).lower()
    
    for i, day in enumerate(days_kr + days_kr_full + days_en + days_en_full):
        if day in text:
            return days_kr[i % 7]
    
    return None

def is_day_cell(cell_content):
    return get_day_of_week(cell_content) is not None

def process_day_row(timetable, merged_ocr_results):
    if not timetable or len(timetable) < 2:
        logger.warning("시간표가 비어있거나 행이 충분하지 않습니다.")
        return timetable

    day_row = timetable[0]
    days = ['','월', '화', '수', '목', '금', '토', '일']
    day_index = 1

    for j, cell in enumerate(day_row):
        if j == 0:  # 첫 번째 열은 시간 정보용이므로 건너뜁니다
            cell['is_time_header'] = True
            cell['day'] = ''
            continue
        
        ocr_result = get_ocr_result_for_cell(cell, merged_ocr_results)
        
        if ocr_result:
            content, confidence = ocr_result
            detected_day = get_day_of_week(content)
            
            if detected_day and confidence > 0.8:
                cell['day'] = detected_day
                cell['content'] = content
                cell['ocr_confidence'] = confidence
                logger.info(f"열 {j}에 요일 할당 (OCR): {cell['day']}, 신뢰도: {confidence:.2f}")
            else:
                cell['content'] = content
                cell['ocr_confidence'] = confidence
                cell['day'] = days[day_index]
                logger.info(f"열 {j}에 요일 자동 할당: {cell['day']}, OCR 내용: {content}, 신뢰도: {confidence:.2f}")
        else:
            cell['day'] = days[day_index]
            cell['content'] = days[day_index]
            cell['ocr_confidence'] = 0.0
            logger.info(f"열 {j}에 요일 자동 할당 (OCR 없음): {cell['day']}")
        
        cell['is_day_header'] = True
        day_index = (day_index + 1) % 7

    # 나머지 행에 요일 정보 전파
    for row in timetable[1:]:
        for j, cell in enumerate(row):
            if j == 0:
                cell['is_time_header'] = True
                cell['day'] = ''
            elif j < len(day_row):
                cell['day'] = day_row[j]['day']

    return timetable

def get_ocr_result_for_cell(cell, merged_ocr_results):
    for result, _ in merged_ocr_results:
       if 'bbox' in cell and not cell['bbox']:
            if is_overlapping(cell['bbox'], result[0]):
                return result[1]
    return None

def is_overlapping(bbox1, bbox2):
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])
