from rdtdet_calc import is_inside_cell
from sklearn.cluster import KMeans
from rdtdet_log import *
from rdtdet_merge import merge_new_cells

def cluster_cell_sizes(cell_sizes, n_clusters=2):
    if not cell_sizes:
        return None
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(cell_sizes)
    logger.info(f"Clustered cell sizes into {n_clusters} groups.")
    return kmeans.cluster_centers_



def get_cell_sizes(cells, ocr_results):
    cell_sizes = []
    for cell in cells:
        for line in ocr_results:
            box = line[0]
            text = line[1][0]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            if is_inside_cell((center_x, center_y), cell):
                x1, y1, x2, y2 = cell['bbox']
                width = x2 - x1
                height = y2 - y1
                text_length = len(text)
                cell_sizes.append((width, height, text_length))
                break
    logger.info(f"Collected {len(cell_sizes)} cell sizes.")
    return cell_sizes

def adjust_new_cell(new_cell, detected_cells, image_shape):
    x1, y1, x2, y2 = map(int, new_cell['bbox'])
    height, width = image_shape[:2]
    
    # 상하좌우로 셀 확장
    while True:
        expanded = False
        
        # 위로 확장
        if y1 > 0 and not any(int(cell['bbox'][1]) < y1 < int(cell['bbox'][3]) for cell in detected_cells):
            y1 -= 1
            expanded = True
        
        # 아래로 확장
        if y2 < height - 1 and not any(int(cell['bbox'][1]) < y2 < int(cell['bbox'][3]) for cell in detected_cells):
            y2 += 1
            expanded = True
        
        # 왼쪽으로 확장
        if x1 > 0 and not any(int(cell['bbox'][0]) < x1 < int(cell['bbox'][2]) for cell in detected_cells):
            x1 -= 1
            expanded = True
        
        # 오른쪽으로 확장
        if x2 < width - 1 and not any(int(cell['bbox'][0]) < x2 < int(cell['bbox'][2]) for cell in detected_cells):
            x2 += 1
            expanded = True
        
        if not expanded:
            break
    
    # 기존 셀과 겹치는 경우 축소
    for cell in detected_cells:
        cx1, cy1, cx2, cy2 = map(int, cell['bbox'])
        
        # 위쪽으로 겹치는 경우
        if cy1 < y1 < cy2:
            y1 = cy2
        
        # 아래쪽으로 겹치는 경우
        if cy1 < y2 < cy2:
            y2 = cy1
        
        # 왼쪽으로 겹치는 경우
        if cx1 < x1 < cx2:
            x1 = cx2
        
        # 오른쪽으로 겹치는 경우
        if cx1 < x2 < cx2:
            x2 = cx1
    
    return {
        'bbox': (int(x1), int(y1), int(x2), int(y2)),
        'score': new_cell['score'],
        'text': new_cell['text'],
        'added': True
    }
 
def adjust_new_cell_size(new_cell, detected_cells, merged_ocr_results):
    text = new_cell['text']
    text_lines = text.split('\n')
    max_line_length = max(len(line) for line in text_lines)
    line_count = len(text_lines)

    suitable_cells = []
    for cell in detected_cells:
        # RTMDET로 인식된 셀만 고려 (이미 detected_cells에 포함되어 있음)
        if 'label' in cell and cell['label'] == 0:  # 0은 'normal_cell'을 나타낸다고 가정
            cell_width = cell['bbox'][2] - cell['bbox'][0]
            cell_height = cell['bbox'][3] - cell['bbox'][1]
            
            # 최소 크기 조건
            if cell_width >= max_line_length * 8 and cell_height >= line_count * 15:
                # 적합도 계산 (값이 작을수록 더 적합)
                suitability = abs(cell_width - max_line_length * 10) + abs(cell_height - line_count * 20)
                suitable_cells.append((cell, suitability))

    if suitable_cells:
        # 가장 적합한 셀 선택
        best_cell, _ = min(suitable_cells, key=lambda x: x[1])
        new_width = best_cell['bbox'][2] - best_cell['bbox'][0]
        new_height = best_cell['bbox'][3] - best_cell['bbox'][1]
        
        x1, y1 = new_cell['bbox'][:2]
        new_cell['bbox'] = (x1, y1, x1 + new_width, y1 + new_height)
        return new_cell
    else:
        # 적합한 셀이 없으면 None 반환
        return None

def add_missing_cells(detected_cells, merged_ocr_results, image_shape):
    """
    감지되지 않은 셀을 추가하고 새로운 셀의 크기를 조정합니다.
    
    Args:
        detected_cells (list): 감지된 셀 정보 리스트
        merged_ocr_results (list): 병합된 OCR 결과 리스트
        image_shape (tuple): 이미지의 shape 정보
    
    Returns:
        list: 모든 셀(감지된 셀 + 새로 추가된 셀) 정보 리스트
    
    사용 상황:
    - OCR 결과 중 감지된 셀에 포함되지 않는 텍스트에 대해 새로운 셀을 생성할 때 사용
    - 표 구조를 완성하기 위해 누락된 셀을 추가할 때 유용
    """
    new_cells = []
    for merged_result, merge_info in merged_ocr_results:
        box, (text, _) = merged_result
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        if not any(is_inside_cell((center_x, center_y), cell) for cell in detected_cells):
            new_cell = {
                'bbox': (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                'score': 0.5,
                'text': text,
                'added': True,
                'merge_info': merge_info
            }
            adjusted_cell = adjust_new_cell_size(new_cell, detected_cells, merged_ocr_results)
            if adjusted_cell is not None:
                new_cells.append(adjusted_cell)
                if "+" in merge_info:
                    logger.info(f"새로운 셀 추가: {merge_info}가 근처에 있어 합쳐졌습니다.")
                else:
                    logger.info(f"새로운 셀 추가: {merge_info}")
            else:
                logger.warning(f"적합한 셀을 찾지 못해 다음 텍스트를 포함하는 셀을 추가하지 않았습니다: {merge_info}")
    
    merged_new_cells = merge_new_cells(new_cells)
    
    logger.info(f"총 {len(detected_cells)}개의 기존 셀, {len(merged_new_cells)}개의 새로운 셀이 생성되었습니다.")
    return detected_cells + merged_new_cells
