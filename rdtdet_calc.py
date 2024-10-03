   
def is_inside_cell(point, cell):
    x, y = point
    x1, y1, x2, y2 = cell['bbox']
    return x1 <= x <= x2 and y1 <= y <= y2

def calculate_text_height(box):
    return box[3][1] - box[0][1]
def is_vertically_aligned(box1, box2, y_threshold):
    """
    두 박스가 수직으로 정렬되어 있는지 확인합니다.
    
    Args:
        box1 (list): 첫 번째 박스의 좌표
        box2 (list): 두 번째 박스의 좌표
        y_threshold (int): 수직 정렬 판단을 위한 y축 임계값
    
    Returns:
        bool: 수직 정렬 여부
    
    사용 상황:
    - OCR 결과를 병합할 때 텍스트가 수직으로 정렬되어 있는지 확인하는 데 사용
    - 여러 줄의 텍스트를 하나로 묶을 때 유용
    """
    return (box2[0][1] - box1[2][1] < y_threshold and 
            abs((box1[0][0] + box1[2][0]) / 2 - (box2[0][0] + box2[2][0]) / 2) < (box1[2][0] - box1[0][0]) / 2)
def is_inside_any_cell(point, detected_cells):
    """
    주어진 점이 감지된 셀 내부에 있는지 확인합니다.
    
    Args:
        point (tuple): 확인할 점의 (x, y) 좌표
        detected_cells (list): 감지된 셀 정보 리스트
    
    Returns:
        bool: 점이 셀 내부에 있는지 여부
    
    사용 상황:
    - OCR 결과가 이미 감지된 셀 내부에 있는지 확인할 때 사용
    - 새로운 셀을 추가해야 하는지 판단할 때 유용
    """
    return any(is_inside_cell(point, cell) for cell in detected_cells)