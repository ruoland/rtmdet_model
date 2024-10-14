import re
from rdtdet_log import logger

def extract_time_info(text):
    patterns = [
        r'(\d{1,2}):(\d{2})\s*[~-]\s*(\d{1,2}):(\d{2})',  # 시간 범위 (예: 18:50-19:25)
        r'(\d{1,2})교시',
        r'(\d{1,2})강',
        r'(\d{1,2}):(\d{2})',
        r'(\d{1,2})시(\d{2})분',
        r'(\d{1,2})시',
        r'(\d{1,2})분',
        r'(\d{1,2})[.:](\d{2})',  # 점(.) 또는 콜론(:)으로 구분된 시간
        r'(\d{1,2})시\s*[~-]\s*(\d{1,2})시',  # 시간 범위 (예: 9시~10시)
        r'(\d{1,2}):\d{2}\s*(AM|PM)',  # AM/PM 포함 (예: 9:00 AM)
        r'(\d{1,2})\s*(AM|PM)',  # AM/PM 포함 (예: 9 AM)
        r'([0-9A-Za-z]+)교시',  # 알파벳과 숫자 조합의 교시 (예: A교시, 1A교시)
        r'([0-9A-Za-z]+)강',    # 알파벳과 숫자 조합의 강의 (예: A강, 1A강)
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if '-' in pattern:
                return f"{match.group(1)}:{match.group(2)}", f"{match.group(3)}:{match.group(4)}"
            elif ':' in pattern:
                return f"{match.group(1)}:{match.group(2)}", None
            elif '교시' in pattern or '강' in pattern:
                return f"{match.group(1)}:00", None
            else:
                return f"{match.group(1)}:00", None
    
    return None, None
def is_time_format(text):
    start_time, end_time = extract_time_info(text)
    return start_time is not None

def is_time_cell(cell_content):
    start_time, _ = extract_time_info(cell_content)
    return start_time is not None


def calculate_end_time(start_time, consecutive_classes, class_duration=50):
    if not start_time:
        return ''
    
    try:
        start_datetime = datetime.strptime(start_time, "%H:%M")
        end_datetime = start_datetime + timedelta(minutes=consecutive_classes * class_duration)
        return end_datetime.strftime("%H:%M")
    except ValueError:
        logger.error(f"잘못된 시작 시간 형식: {start_time}")
        return ''
