
from rdtdet_log import *
from sklearn.cluster import DBSCAN
import numpy as np

def organize_cells_into_grid(cells, ocr_results, eps=10, min_samples=2):
    # 셀 좌표를 수집
    cell_coords = np.array([(cell['bbox'][0], cell['bbox'][1]) for cell in cells])

    # DBSCAN을 사용하여 행과 열 클러스터링
    dbscan_y = DBSCAN(eps=eps, min_samples=min_samples).fit(cell_coords[:, 1].reshape(-1, 1))
    dbscan_x = DBSCAN(eps=eps, min_samples=min_samples).fit(cell_coords[:, 0].reshape(-1, 1))

    # 행과 열 레이블 할당
    row_labels = dbscan_y.labels_
    col_labels = dbscan_x.labels_

    # 그리드 생성
    n_rows = len(set(row_labels)) - (1 if -1 in row_labels else 0)
    n_cols = len(set(col_labels)) - (1 if -1 in col_labels else 0)
    grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    # 셀을 그리드에 배치
    for cell, row, col in zip(cells, row_labels, col_labels):
        if row != -1 and col != -1:
            grid[row][col] = cell

    # OCR 결과를 셀에 매칭
    for ocr in ocr_results:
        ocr_box = ocr[0]
        ocr_text = ocr[1][0]
        matched = False
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell and is_inside_cell(ocr_box, cell['bbox']):
                    if 'text' not in cell or not cell['text']:
                        cell['text'] = ocr_text
                    else:
                        cell['text'] += ' ' + ocr_text
                    matched = True
                    break
            if matched:
                break

    return grid

def is_inside_cell(ocr_box, cell_bbox):
    ocr_center = ((ocr_box[0][0] + ocr_box[2][0]) / 2, (ocr_box[0][1] + ocr_box[2][1]) / 2)
    return (cell_bbox[0] <= ocr_center[0] <= cell_bbox[2] and
            cell_bbox[1] <= ocr_center[1] <= cell_bbox[3])

def merge_ocr_results(ocr_results, y_threshold=20, x_threshold=10):
    # 인식한 글자 합치기
    def _merge_group(group):
        boxes = [line[0] for line in group]
        texts = [line[1][0] for line in group]
        confidences = [line[1][1] for line in group]
        
        merged_box = [
            min(box[0][0] for box in boxes),
            min(box[0][1] for box in boxes),
            max(box[2][0] for box in boxes),
            max(box[2][1] for box in boxes)
        ]
        
        merged_text = '\n'.join(texts)
        avg_confidence = sum(confidences) / len(confidences)
        
        merge_info = " + ".join([f'"{text}"' for text in texts])
        logger.debug(f"병합된 텍스트: {merge_info}, 위치: {merged_box}")
        
        return [merged_box, (merged_text, avg_confidence)], merge_info

    def boxes_overlap(box1, box2):
        return (box1[0][0] < box2[2][0] and box2[0][0] < box1[2][0] and
                box1[0][1] < box2[2][1] and box2[0][1] < box1[2][1])

    merged_results = []
    processed = set()

    for i, (box, (text, conf)) in enumerate(ocr_results):
        if i in processed:
            continue

        current_group = [(box, (text, conf))]
        current_box = box
        processed.add(i)

        while True:
            expanded_box = [
                [current_box[0][0], current_box[0][1]],
                [current_box[1][0], current_box[1][1]],
                [current_box[2][0], current_box[2][1] + y_threshold],
                [current_box[3][0], current_box[3][1] + y_threshold]
            ]

            overlapped = False
            for j, (next_box, (next_text, next_conf)) in enumerate(ocr_results):
                if j in processed:
                    continue

                if boxes_overlap(expanded_box, next_box):
                    if abs(next_box[0][0] - current_box[0][0]) <= x_threshold:
                        current_group.append((next_box, (next_text, next_conf)))
                        current_box = [
                            [min(current_box[0][0], next_box[0][0]), min(current_box[0][1], next_box[0][1])],
                            [max(current_box[1][0], next_box[1][0]), min(current_box[1][1], next_box[1][1])],
                            [max(current_box[2][0], next_box[2][0]), max(current_box[2][1], next_box[2][1])],
                            [min(current_box[3][0], next_box[3][0]), max(current_box[3][1], next_box[3][1])]
                        ]
                        processed.add(j)
                        overlapped = True

            if not overlapped:
                break

        merged_result, merge_info = _merge_group(current_group)
        merged_results.append((merged_result, merge_info))

    logger.info(f"{len(ocr_results)}개의 OCR 결과를 {len(merged_results)}개의 그룹으로 병합했습니다.")
    return merged_results

def cells_overlap(cell1, cell2, overlap_threshold=0.5):
    """
    두 셀이 지정된 임계값 이상으로 겹치는지 확인합니다.
    
    Args:
        cell1 (dict): 첫 번째 셀 정보
        cell2 (dict): 두 번째 셀 정보
        overlap_threshold (float): 겹침 판단을 위한 임계값
    
    Returns:
        bool: 셀 겹침 여부
    
    사용 상황:
    - 새로 추가된 셀이 기존 셀과 겹치는지 확인할 때 사용
    - 셀 병합 여부를 결정할 때 유용
    """
    x1, y1, x2, y2 = cell1['bbox']
    x3, y3, x4, y4 = cell2['bbox']
    
    # 각 셀의 면적 계산
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    
    # 면적이 0인 경우 체크
    if area1 == 0 or area2 == 0:
        logging.warning(f"Zero area detected: cell1={cell1}, cell2={cell2}")
        return False
    
    # 겹치는 영역 계산
    overlap_x = max(0, min(x2, x4) - max(x1, x3))
    overlap_y = max(0, min(y2, y4) - max(y1, y3))
    overlap_area = overlap_x * overlap_y
    
    # 각 셀의 면적 계산
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    
    # 겹침 비율 계산
    overlap_ratio = overlap_area / min(area1, area2)
    
    return overlap_ratio > overlap_threshold
def merge_new_cells(new_cells):
    """
    새로운 셀들끼리만 병합합니다.
    
    Args:
        new_cells (list): 새로 추가된 셀 정보 리스트
    
    Returns:
        list: 병합된 새로운 셀 정보 리스트
    
    사용 상황:
    - OCR 결과로 생성된 새로운 셀들 중 겹치는 셀을 병합할 때 사용
    - 표 구조를 정리하고 중복을 제거하는 데 유용
    """
    merged_cells = []
    for cell in new_cells:
        overlapped = False
        for i, merged_cell in enumerate(merged_cells):
            if cells_overlap(cell, merged_cell):
                # 셀 병합
                x1 = min(cell['bbox'][0], merged_cell['bbox'][0])
                y1 = min(cell['bbox'][1], merged_cell['bbox'][1])
                x2 = max(cell['bbox'][2], merged_cell['bbox'][2])
                y2 = max(cell['bbox'][3], merged_cell['bbox'][3])
                merged_cells[i] = {
                    'bbox': (x1, y1, x2, y2),
                    'score': max(cell['score'], merged_cell['score']),
                    'text': merged_cell['text'] + '\n' + cell['text'],
                    'added': True
                }
                overlapped = True
                break
        if not overlapped:
            merged_cells.append(cell)
    return merged_cells