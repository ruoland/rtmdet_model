import re
from rdtdet_log import logger

def get_day_of_week(text):
    days_kr = ['월', '화', '수', '목', '금', '토', '일']
    days_kr_full = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    days_en = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    days_en_full = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    text = text.lower()
    
    for i, day in enumerate(days_kr + days_kr_full + days_en + days_en_full):
        if day in text:
            return days_kr[i % 7]
    
    return None

def is_day_cell(cell_content):
    return get_day_of_week(cell_content) is not None

def process_day_row(timetable):
    if not timetable:
        return timetable
    
    header = timetable[0]
    for cell in header[1:]:  # 첫 번째 열(시간 열) 제외
        day = get_day_of_week(cell['content'])
        if day:
            cell['day'] = day
        else:
            cell['day'] = ""
    
    logger.info("요일 행 처리 완료")
    return timetable