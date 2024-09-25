import os
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
import mmcv
import cv2
import numpy as np

# 기본 경로 설정
base_path = r"D:\Projects\OCR-LEARNIGN-PROJECT\OCR-PROJECT_OLD"
dataset_path = os.path.join(base_path, "yolox_table_dataset_simple233-0918")

# 설정 파일 로드 (이 파일의 경로는 실제 위치에 맞게 조정해야 할 수 있습니다)
cfg = Config.fromfile('mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py')

# 제공된 설정으로 cfg 업데이트
cfg.resume = True
cfg.max_epochs = 12
cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=cfg.max_epochs, val_interval=5)
cfg.val_cfg = dict(type='ValLoop')
cfg.test_cfg = dict(type='TestLoop')

cfg.train_dataloader.batch_size = 32
cfg.val_dataloader.batch_size = 32
cfg.test_dataloader = cfg.val_dataloader
cfg.env_cfg = dict(cudnn_benchmark=True)

cfg.data_root = dataset_path
cfg.train_dataloader.dataset.ann_file = os.path.join(dataset_path, 'train_annotations.json')
cfg.train_dataloader.dataset.data_root = os.path.join(dataset_path, 'train')
cfg.val_dataloader.dataset.ann_file = os.path.join(dataset_path, 'val_annotations.json')
cfg.val_dataloader.dataset.data_root = os.path.join(dataset_path, 'val')

cfg.model.bbox_head.num_classes = 3
# 클래스 이름 정의 (예시)
class_names = ['Cell', 'Merged_Cell', 'table']
cfg.model.bbox_head.num_classes = len(class_names)

cfg.work_dir = os.path.join(base_path, 'rtmdet_table_detection')
# 체크포인트 파일 경로
checkpoint_file = r"C:\Users\opron\Downloads\epoch_1.pth"

# 모델 초기화 (CPU 사용)
model = init_detector(cfg, checkpoint_file, device='cpu')

# 이미지 저장 함수
def save_image(image, filename):
    cv2.imwrite(filename, image)
    print(f"Saved image: {filename}")

# 결과 그리기 함수
def draw_detection(image, bbox, label, score, color=(0, 255, 0), thickness=1):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    text = f"{class_names[label]}: {score:.2f}"
    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def preprocess_image(image):
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 적응형 이진화
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # 3채널로 변환
    binary_3channel = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return binary_3channel

def process_and_detect(img_path, model, class_names, score_threshold=0.7):
    # 이미지 읽기
    original_img = cv2.imread(img_path)
    
    # 전처리
    preprocessed_img = preprocess_image(original_img)
    
    # 추론 실행
    result = inference_detector(model, preprocessed_img)
    
    # 결과 처리
    detected_cells = []
    table_info = None
    if hasattr(result, 'pred_instances'):
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
            if score > score_threshold:

                print(f"Detection {i}:")
                print(f"  Class: {class_names[label]}")
                print(f"  Score: {score}")
                print(f"  Bbox: {bbox}")

        for bbox, label, score in zip(bboxes, labels, scores):
            if score > score_threshold:
                class_name = class_names[label]
                x1, y1, x2, y2 = map(int, bbox)
                print(class_name)
                if class_name in ['Cell']:
                    cell_img = original_img[y1:y2, x1:x2]
                    cell_info = {
                        'class': class_name,
                        'bbox': (x1, y1, x2, y2),
                        'score': score,
                        'image': cell_img
                    }
                    
                    detected_cells.append(cell_info)
                
                elif class_name == 'table':
                    table_info = {
                        'bbox': (x1, y1, x2, y2),
                        'score': score
                    }
    detected_cells = sort_cells(detected_cells)

    # 빈 셀 찾기
    all_cells = find_empty_cells(detected_cells, table_info)

    # 새로운 표 이미지 생성
    new_table_img = create_table_image(all_cells, table_info)

    return preprocessed_img, all_cells, table_info, new_table_img


def sort_cells(cells):
    return sorted(cells, key=lambda x: (x['bbox'][1], x['bbox'][0]))

def find_empty_cells(sorted_cells, table_info):
    if not table_info:
        return sorted_cells

    table_x1, table_y1, table_x2, table_y2 = table_info['bbox']
    
    # 그리드 크기 추정
    cell_heights = [cell['bbox'][3] - cell['bbox'][1] for cell in sorted_cells]
    cell_widths = [cell['bbox'][2] - cell['bbox'][0] for cell in sorted_cells]
    avg_cell_height = sum(cell_heights) / len(cell_heights)
    avg_cell_width = sum(cell_widths) / len(cell_widths)

    rows = int((table_y2 - table_y1) / avg_cell_height)
    cols = int((table_x2 - table_x1) / avg_cell_width)

    grid = [[None for _ in range(cols)] for _ in range(rows)]

    for cell in sorted_cells:
        x1, y1, x2, y2 = cell['bbox']
        row = int((y1 - table_y1) / avg_cell_height)
        col = int((x1 - table_x1) / avg_cell_width)
        grid[row][col] = cell

    all_cells = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] is None:
                # 빈 셀 추가
                x1 = int(table_x1 + j * avg_cell_width)
                y1 = int(table_y1 + i * avg_cell_height)
                x2 = int(x1 + avg_cell_width)
                y2 = int(y1 + avg_cell_height)
                empty_cell = {
                    'class': 'Empty_Cell',
                    'bbox': (x1, y1, x2, y2),
                    'score': 0,
                    'image': None
                }
                all_cells.append(empty_cell)
            else:
                all_cells.append(grid[i][j])

    return sort_cells(all_cells)
# 메인 실행 코드
def create_table_image(cells, table_info, padding=5):
    if not table_info:
        return None

    table_x1, table_y1, table_x2, table_y2 = table_info['bbox']
    table_width = table_x2 - table_x1
    table_height = table_y2 - table_y1

    # 흰색 배경의 이미지 생성
    table_img = np.ones((table_height, table_width, 3), dtype=np.uint8) * 255

    # 셀 그리기
    for cell in cells:
        x1, y1, x2, y2 = cell['bbox']
        x1 -= table_x1
        y1 -= table_y1
        x2 -= table_x1
        y2 -= table_y1
        cv2.rectangle(table_img, (x1, y1), (x2, y2), (0, 0, 0), 1)

        # 셀 내용 추가 (예: 셀 번호)
        cell_text = f"Cell {cells.index(cell)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(cell_text, font, font_scale, 1)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(table_img, cell_text, (text_x, text_y), font, font_scale, (0, 0, 0), 1)

    return table_img
def draw_detection_results(image, cells, table_info, class_names):
    result_image = image.copy()
    
    # 테이블 그리기
    if table_info:
        x1, y1, x2, y2 = map(int, table_info['bbox'])
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_image, f"Table: {table_info['score']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 셀 그리기
    for cell in cells:
        x1, y1, x2, y2 = map(int, cell['bbox'])
        class_name = cell['class']
        score = cell.get('score', 0)  # Empty cells might not have a score
        
        if class_name == 'Cell':
            color = (255, 0, 0)  # 빨간색
        elif class_name == 'Merged_Cell':
            color = (0, 0, 255)  # 파란색
        else:  # Empty_Cell
            continue
        
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {score:.2f}" if score > 0 else class_name
        cv2.putText(result_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image

# 메인 실행 코드
if __name__ == "__main__":
    img_path = r"original.png"
    output_dir = "detected_results"
    os.makedirs(output_dir, exist_ok=True)

    # 원본 이미지 읽기
    original_img = cv2.imread(img_path)

    # 전처리 및 감지 실행
    preprocessed_img, all_cells, table_info, new_table_img = process_and_detect(img_path, model, class_names)

    # 결과 출력
    print(f"Detected {len(all_cells)} cells (including empty cells).")
    if table_info:
        print(f"Detected table: {table_info}")

    # 결과를 원본 이미지에 그리기
    result_image = draw_detection_results(original_img, all_cells, table_info, class_names)

    # 결과 이미지 저장
    output_path = os.path.join(output_dir, "detection_result.png")
    cv2.imwrite(output_path, result_image)
    print(f"Detection result saved as {output_path}")

    # 새로운 표 이미지 저장 (기존 코드)
    if new_table_img is not None:
        new_table_path = os.path.join(output_dir, "new_table.png")
        cv2.imwrite(new_table_path, new_table_img)
        print(f"New table image saved as {new_table_path}")

    print(f"Results saved in {output_dir}")
