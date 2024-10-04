import cv2
import os
import numpy as np
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
import json

# 모델 설정
CONFIG_FILE = 'mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
CHECKPOINT_FILE = r"c:\Users\opron\Downloads\epoch_4.pth"
DEVICE = 'cpu'
IMAGE_THRESHOLD = 0.5
RESIZE_SCALE = 2.0  # 1.0은 원본 크기, 2.0은 2배 확대, 0.5는 절반으로 축소
MIN_IMAGE_SIZE = (800, 600)  # 최소 이미지 크기 (너비, 높이)
MAX_IMAGE_SIZE = (2400, 2400)  # 최대 이미지 크기 (너비, 높이)

CLASS_NAMES = ['cell', 'table', 'row', 'column', 'merged_cell', 'overflow_cell', 'merged_overflow_cell']

def initialize_model(config_file=CONFIG_FILE, checkpoint_file=CHECKPOINT_FILE, device=DEVICE):
    cfg = Config.fromfile(config_file)
    cfg.model.bbox_head.num_classes = len(CLASS_NAMES)
    model = init_detector(cfg, checkpoint_file, device=device)
    return model

def resize_image(image, scale=RESIZE_SCALE):
    # 원본 이미지 크기
    original_height, original_width = image.shape[:2]
    
    # 스케일에 따른 새로운 크기 계산
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # 최소 및 최대 크기 제한 적용
    new_width = max(MIN_IMAGE_SIZE[0], min(new_width, MAX_IMAGE_SIZE[0]))
    new_height = max(MIN_IMAGE_SIZE[1], min(new_height, MAX_IMAGE_SIZE[1]))
    
    # 이미지 리사이즈
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image, (new_width / original_width, new_height / original_height)

def detect_objects(model, image, threshold=IMAGE_THRESHOLD):
    result = inference_detector(model, image)
    detected_objects = []
    if hasattr(result, 'pred_instances'):
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        
        for bbox, label, score in zip(bboxes, labels, scores):
            if score > threshold:
                x1, y1, x2, y2 = map(int, bbox)
                object_info = {
                    'bbox': (x1, y1, x2, y2),
                    'score': float(score),
                    'label': int(label),
                    'class_name': CLASS_NAMES[int(label)]
                }
                detected_objects.append(object_info)
    return detected_objects

def expand_bbox(bbox, expand_ratio=1.2):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    new_width = width * expand_ratio
    new_height = height * expand_ratio
    new_x1 = max(0, int(center_x - new_width / 2))
    new_y1 = max(0, int(center_y - new_height / 2))
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)
    return [new_x1, new_y1, new_x2, new_y2]

def is_inside(inner_bbox, outer_bbox):
    return (outer_bbox[0] <= inner_bbox[0] and inner_bbox[2] <= outer_bbox[2] and
            outer_bbox[1] <= inner_bbox[1] and inner_bbox[3] <= outer_bbox[3])

def process_image(image_path, model, do_resize=True):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"{image_path}에서 이미지를 불러오는데 실패했습니다.")
    
    if do_resize:
        resized_img, scale_factors = resize_image(original_img)
    else:
        resized_img = original_img
        scale_factors = (1.0, 1.0)
    
    detected_objects = detect_objects(model, resized_img)
    
    # 원본 크기로 좌표 변환
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        obj['bbox'] = (
            int(x1 / scale_factors[0]),
            int(y1 / scale_factors[1]),
            int(x2 / scale_factors[0]),
            int(y2 / scale_factors[1])
        )
    
    return original_img, detected_objects

def draw_results(image, detected_objects, **kwargs):
    result_image = image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128)]
    
    for obj in detected_objects:
        class_name = obj['class_name']
        if kwargs.get(class_name, False):
            color = colors[obj['label']]
            cv2.rectangle(result_image, (obj['bbox'][0], obj['bbox'][1]),
                          (obj['bbox'][2], obj['bbox'][3]), color, 2)
            cv2.putText(result_image, f"{class_name}: {obj['score']:.2f}", 
                        (obj['bbox'][0], obj['bbox'][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image

# 결과 저장 (옵션)
def save_results(output_dir, result_image, detected_objects):
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'detection_result.jpg'), result_image)
    
    with open(os.path.join(output_dir, 'detection_result.json'), 'w', encoding='utf-8') as f:
        json.dump(detected_objects, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved in {output_dir}")

def main():
    img_path = r"20220304_174813461_61050.jpeg"
    output_dir = r"./output/result.jpeg"
    config_file = 'mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
    rtmdet = initialize_model(config_file, CHECKPOINT_FILE)
    
    # 이미지 처리 및 객체 감지 (리사이즈 수행)
    original_img, detected_objects = process_image(img_path, rtmdet, do_resize=True)
    
    # 결과 저장 함수
    def save_custom_results(selected_classes):
        result_image = draw_results(original_img, detected_objects, **selected_classes)
        save_results(output_dir, result_image, detected_objects)
    
    # 예시 1: 테이블과 행만 표시
    save_custom_results({'table': True, 'row': True})
    
    # 예시 2: 모든 셀 유형 표시
    save_custom_results({'cell': True, 'merged_cell': True, 'overflow_cell': True, 'merged_overflow_cell': True})
    
    # 예시 3: 테이블, 열, 일반 셀만 표시
    save_custom_results({'table': True, 'column': True, 'cell': True})
    
    # 예시 4: 모든 객체 표시
    save_custom_results({class_name: True for class_name in CLASS_NAMES})
    result_image = draw_results(original_img, detected_objects,{class_name: True for class_name in CLASS_NAMES})
    save_results(output_dir, result_image, detected_objects)

if __name__ == "__main__":
    main()
