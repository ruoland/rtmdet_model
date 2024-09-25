import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import logging

# 로거 객체 생성
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 핸들러 생성 및 설정
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# 포맷터 생성 및 핸들러에 추가
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 로거에 핸들러 추가
logger.addHandler(handler)

# 로그 메시지 출력
logger.debug("디버그 메시지")
logger.info("정보 메시지")
logger.warning("경고 메시지")
logger.error("에러 메시지")
logger.critical("심각한 에러 메시지")

# OpenMP 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 모델 초기화
ocr = PaddleOCR(use_angle_cls=False, lang='korean')
cfg = Config.fromfile('mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py')
cfg.model.bbox_head.num_classes = 3
checkpoint_file = r"epoch_19.pth"
rtmdet = init_detector(cfg, checkpoint_file, device='cpu')
def debug_print_ocr_result(ocr_result):
    logger.debug("OCR 결과 구조:")
    for i, line in enumerate(ocr_result):
        logger.debug(f"라인 {i}:")
        logger.debug(f"  타입: {type(line)}")
        logger.debug(f"  내용: {line}")
        if isinstance(line, (list, tuple)) and len(line) == 2:
            box, text_info = line
            logger.debug(f"  박스: {box}")
            logger.debug(f"    타입: {type(box)}")
            logger.debug(f"  텍스트 정보: {text_info}")
            logger.debug(f"    타입: {type(text_info)}")
def process_image(image_path):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"{image_path}에서 이미지를 불러오는데 실패했습니다.")
    
    result = inference_detector(rtmdet, original_img)
    ocr_result = ocr.ocr(original_img, cls=False)
    debug_print_ocr_result(ocr_result[0])
    
    detected_cells = []
    if hasattr(result, 'pred_instances'):
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        
        for bbox, label, score in zip(bboxes, labels, scores):
            if label == 0 and score > 0.7:
                x1, y1, x2, y2 = map(int, bbox)
                cell_info = {
                    'bbox': (x1, y1, x2, y2),
                    'score': score
                }
                detected_cells.append(cell_info)
    
    logger.info(f"{len(detected_cells)}개의 셀과 {len(ocr_result[0])}개의 OCR 결과를 감지했습니다.")
    return original_img, detected_cells, ocr_result[0]

def is_inside_cell(point, cell):
    x, y = point
    x1, y1, x2, y2 = cell['bbox']
    return x1 <= x <= x2 and y1 <= y <= y2

def is_x_aligned(box1, box2, text1, text2, font_size_estimate=20):
    """두 박스의 X 좌표가 유사한지 확인하고, 글자 길이를 고려합니다."""
    x1_center = (box1[0][0] + box1[2][0]) / 2
    x2_center = (box2[0][0] + box2[2][0]) / 2
    x1_width = box1[2][0] - box1[0][0]
    x2_width = box2[2][0] - box2[0][0]
    
    # 글자 길이에 따른 허용 범위 계산
    text_length_diff = abs(len(text1) - len(text2))
    allowed_diff = max(x1_width, x2_width, text_length_diff * font_size_estimate)
    
    # X 중심점 차이가 허용 범위 이내인지 확인
    return abs(x1_center - x2_center) <= allowed_diff / 2

def calculate_text_height(box):
    return box[3][1] - box[0][1]
def is_vertically_aligned(box1, box2, y_threshold):
    """두 박스가 수직으로 정렬되어 있는지 확인합니다."""
    return box2[0][1] - box1[2][1] < y_threshold and \
           abs((box1[0][0] + box1[2][0]) / 2 - (box2[0][0] + box2[2][0]) / 2) < (box1[2][0] - box1[0][0]) / 2

def is_inside_any_cell(point, detected_cells):
    """주어진 점이 감지된 셀 내부에 있는지 확인합니다."""
    return any(is_inside_cell(point, cell) for cell in detected_cells)

def merge_ocr_results(ocr_results, detected_cells, y_threshold=60):
    merged_results = []
    i = 0
    while i < len(ocr_results):
        current_group = [ocr_results[i]]
        j = i + 1
        while j < len(ocr_results):
            prev_box, (prev_text, _) = current_group[-1]
            curr_box, (curr_text, _) = ocr_results[j]
            
            # 현재 박스의 중심점
            curr_center = ((curr_box[0][0] + curr_box[2][0]) / 2, (curr_box[0][1] + curr_box[2][1]) / 2)
            
            if is_vertically_aligned(prev_box, curr_box, y_threshold) and not is_inside_any_cell(curr_center, detected_cells):
                current_group.append(ocr_results[j])
                j += 1
            else:
                break
        
        merged_results.append(merge_group(current_group))
        i = j
    
    logger.info(f"{len(ocr_results)}개의 OCR 결과를 {len(merged_results)}개의 그룹으로 병합했습니다.")
    return merged_results

def merge_group(group):
    """그룹 내의 OCR 결과를 하나로 병합합니다."""
    boxes = [line[0] for line in group]
    texts = [line[1][0] for line in group]
    confidences = [line[1][1] for line in group]
    
    merged_box = [
        min(box[0][0] for box in boxes),  # 왼쪽 상단 x
        min(box[0][1] for box in boxes),  # 왼쪽 상단 y
        max(box[2][0] for box in boxes),  # 오른쪽 하단 x
        max(box[2][1] for box in boxes)   # 오른쪽 하단 y
    ]
    
    merged_text = '\n'.join(texts)  # 줄바꿈으로 텍스트 연결
    avg_confidence = sum(confidences) / len(confidences)
    
    logger.debug(f"병합된 텍스트: '{merged_text}', 위치: {merged_box}")
    return [merged_box, (merged_text, avg_confidence)]

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

def cluster_cell_sizes(cell_sizes, n_clusters=2):
    if not cell_sizes:
        return None
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(cell_sizes)
    logger.info(f"Clustered cell sizes into {n_clusters} groups.")
    return kmeans.cluster_centers_
def cells_overlap(cell1, cell2, overlap_threshold=0.5):
    """두 셀이 지정된 임계값 이상으로 겹치는지 확인합니다."""
    x1, y1, x2, y2 = cell1['bbox']
    x3, y3, x4, y4 = cell2['bbox']
    
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

def merge_overlapping_cells(cells):
    """겹치는 셀들을 병합합니다."""
    merged_cells = []
    for cell in cells:
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
                    'text': merged_cell['text'] + '\n' + cell.get('text', '')
                }
                overlapped = True
                break
        if not overlapped:
            merged_cells.append(cell)
    return merged_cells


def merge_new_cells(new_cells):
    """새로운 셀들끼리만 병합합니다."""
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
def add_missing_cells(detected_cells, merged_ocr_results, image_shape):
    new_cells = []
    
    for line in merged_ocr_results:
        box = line[0]
        text = line[1][0]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        if not any(is_inside_cell((center_x, center_y), cell) for cell in detected_cells):
            new_cell = {
                'bbox': (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                'score': 0.5,
                'text': text,
                'added': True
            }
            adjusted_cell = adjust_new_cell(new_cell, detected_cells, image_shape)
            new_cells.append(adjusted_cell)
    
    # 새로운 셀들끼리 병합
    merged_new_cells = merge_new_cells(new_cells)
    
    logger.info(f"총 {len(detected_cells)}개의 기존 셀, {len(merged_new_cells)}개의 새로운 셀이 생성되었습니다.")
    return detected_cells + merged_new_cells



def draw_results(image, cells, ocr_results):
    """결과를 이미지에 그립니다."""
    result_image = image.copy()
    
    # 셀 그리기
    for cell in cells:
        x1, y1, x2, y2 = map(int, cell['bbox'])
        color = (0, 0, 255) if cell.get('added', False) else (255, 0, 0)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
    
    # OCR 결과 그리기
    font_path = "nanumgothic.ttf"
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    for line in ocr_results:
        box, (text, confidence) = line
        if confidence > 0.5:
            x1, y1 = int(box[0]), int(box[1])
            # OCR 박스 그리기
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=(0, 255, 0), width=2)
            # 여러 줄 텍스트 처리
            y_offset = 0
            for text_line in text.split('\n'):
                draw.text((x1, y1 + y_offset), text_line, font=font, fill=(0, 255, 0, 255))
                y_offset += font_size + 2  # 줄 간격 추가
    
    result_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return result_image

# 메인 실행 부분
if __name__ == "__main__":
    img_path = r"checklist\with_gray_scale_table.png"
    output_dir = "ocr_cell"
    os.makedirs(output_dir, exist_ok=True)

    original_img, detected_cells, ocr_results = process_image(img_path)
    
    logger.info("OCR 결과 병합을 시작합니다.")
    merged_ocr_results = merge_ocr_results(ocr_results, detected_cells)
    
    logger.info("누락된 셀을 추가하고 새로운 셀의 크기를 조정합니다.")
    all_cells = add_missing_cells(detected_cells, merged_ocr_results, original_img.shape)
    
    result_image = draw_results(original_img, all_cells, merged_ocr_results)
    
    output_path = os.path.join(output_dir, "detection_result.png")
    cv2.imwrite(output_path, result_image)
    logger.info(f"감지 결과가 {output_path}에 저장되었습니다.")

    logger.info(f"원본 셀 {len(detected_cells)}개를 감지했습니다.")
    logger.info(f"새로운 셀 {len(all_cells) - len(detected_cells)}개를 추가했습니다.")
    logger.info(f"병합된 OCR 결과 {len(merged_ocr_results)}개를 감지했습니다.")
