import re
from rdtdet_log import logger

def extract_time_info(text):
    patterns = [
        r'(\d{1,2})교시',
        r'(\d{1,2})강',
        r'(\d{1,2}):(\d{2})',
        r'(\d{1,2})시',
        # 필요에 따라 더 많은 패턴을 추가할 수 있습니다.
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if ':' in pattern:
                return f"{match.group(1)}:{match.group(2)}"
            else:
                return f"{match.group(1)}:00"
    
    return None

def is_time_cell(cell_content):
    return extract_time_info(cell_content) is not None

def process_time_column(timetable):
    for row in timetable[1:]:  # 헤더 행 제외
        cell = row[0]
        time_info = extract_time_info(cell['content'])
        if time_info:
            cell['time'] = time_info
        else:
            cell['time'] = ""
    
    logger.info("시간 열 처리 완료")
    return timetable

def calculate_end_time(start_time, consecutive_classes):
    hours, minutes = map(int, start_time.split(':'))
    total_minutes = hours * 60 + minutes + (consecutive_classes * 50)  # 한 교시를 50분으로 가정
    end_hours = total_minutes // 60
    end_minutes = total_minutes % 60
    return f"{end_hours:02d}:{end_minutes:02d}"