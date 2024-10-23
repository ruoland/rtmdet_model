from inspect import isgenerator
import cv2
import os
import numpy as np
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
import json

# 모델 설정
CONFIG_FILE = 'mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
CHECKPOINT_FILE = r"C:\Users\opron\Downloads\epoch_17.pth"
DEVICE = 'cpu'
IMAGE_THRESHOLD = 0.7

RESIZE_SCALE = 3.0  # 1.0은 원본 크기, 2.0은 2배 확대, 0.5는 절반으로 축소
MIN_IMAGE_SIZE = (800, 600)  # 최소 이미지 크기 (너비, 높이)
MAX_IMAGE_SIZE = (3200, 3200)  # 최대 이미지 크기 (너비, 높이)

CLASS_NAMES = ['cell', 'table', 'row', 'column', 'merged_cell', 'overflow_cell', 'header_row', 'header_column']

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
def preprocess_image(image):
    # 이미지가 이미 그레이스케일인지 확인
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        gray_image = image
    else:
        # 컬러 이미지인 경우 그레이스케일로 변환
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 이진화 처리
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    
    return gray_image, binary_image
def draw_results(image, detected_objects, class_name=None):
    # 흑백 이미지를 컬러로 변환
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result_image = image.copy()

    
    colors = [
        (255, 0, 0),    # 빨강
        (0, 255, 0),    # 녹색
        (0, 0, 255),    # 파랑
        (255, 255, 0),  # 노랑
        (255, 0, 255),  # 마젠타
        (0, 255, 255),  # 시안
        (255, 128, 0),  # 주황
        (128, 0, 255)   # 보라
    ]
    
    for obj in detected_objects:
        if class_name is None or obj['class_name'] == class_name:
            color = colors[obj['label']]
            cv2.rectangle(result_image, (obj['bbox'][0], obj['bbox'][1]),
                          (obj['bbox'][2], obj['bbox'][3]), color, 2)
            cv2.putText(result_image, f"{obj['class_name']}: {obj['score']:.2f}", 
                        (obj['bbox'][0], obj['bbox'][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image

def save_results(output_dir, image_name, result_image, detected_objects, class_name=None):
    if not detected_objects:
        return  # 감지된 객체가 없으면 저장하지 않음

    os.makedirs(output_dir, exist_ok=True)
    
    if class_name:
        cv2.imwrite(os.path.join(output_dir, f'{image_name}_{class_name}_result.jpg'), result_image)
    else:
        cv2.imwrite(os.path.join(output_dir, f'{image_name}_all_result.jpg'), result_image)

def process_folder(input_folder, output_dir, model):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            process_single_image(img_path, output_dir, model, True)
            
def process_image(image_path, model, do_resize=True, is_gray=True):
    original_img = cv2.imread(image_path)
    result_image = None
    if is_gray:
        result_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_path = image_path[0:15]
        file_name = image_path[16:]
        
        if 'png' in image_path:
            cv2.imwrite(img_path+f'/gray/{file_name}.png', original_img)
        else:
            cv2.imwrite(img_path+f'/gray/{file_name}.jpg', original_img)
    else:
        result_image = cv2.imread(image_path)
    
    if result_image is None:
        raise ValueError(f"{image_path}에서 이미지를 불러오는데 실패했습니다.")
    
    if do_resize:
        resized_img, scale_factors = resize_image(result_image)
    else:
        resized_img = result_image
        scale_factors = (1.0, 1.0)

    if is_gray:
        # 그레이스케일 이미지를 3채널 RGB로 변환
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    
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

def crop_and_save_objects(image, detected_objects, output_dir, image_name):
    # 이미지 이름으로 폴더 생성
    image_folder = os.path.join(output_dir, image_name)
    os.makedirs(image_folder, exist_ok=True)

    for class_name in CLASS_NAMES:
        class_objects = [obj for obj in detected_objects if obj['class_name'] == class_name]
        if class_objects:
            # 클래스별 폴더 생성
            class_dir = os.path.join(image_folder, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i, obj in enumerate(class_objects):
                x1, y1, x2, y2 = obj['bbox']
                cropped_img = image[y1:y2, x1:x2]
                
                # 이미지가 비어있지 않은지 확인
                if cropped_img.size > 0:
                    output_path = os.path.join(class_dir, f"{image_name}_{class_name}_{i}.jpg")
                    cv2.imwrite(output_path, cropped_img)

def process_single_image(img_path, output_dir, model, is_gray):
    original_img, detected_objects = process_image(img_path, model, do_resize=True, is_gray=True)
    image_name = os.path.splitext(os.path.basename(img_path))[0]
    
    if not detected_objects:
        print(f"No objects detected in {image_name}")
        return

    # 모든 객체 표시
    all_result_image = draw_results(original_img, detected_objects)
    save_results(output_dir, image_name, all_result_image, detected_objects)
    
    # 클래스별 결과 저장
    for class_name in CLASS_NAMES:
        class_objects = [obj for obj in detected_objects if obj['class_name'] == class_name]
        if class_objects:
            class_result_image = draw_results(original_img, class_objects, class_name)
            save_results(output_dir, image_name, class_result_image, class_objects, class_name)
    
    # 객체 잘라내어 저장 (이미지별로 폴더에 저장)
    crop_and_save_objects(original_img, detected_objects, output_dir, image_name)

def main():
    config_file = 'mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
    rtmdet = initialize_model(config_file, CHECKPOINT_FILE)
    
    # 폴더 내 모든 이미지 처리
    input_folder = r"./table_testset"
    output_dir = r"./output/result2"
    process_folder(input_folder, output_dir, rtmdet)

if __name__ == "__main__":
    main()
