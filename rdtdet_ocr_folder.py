import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from PIL import Image, ImageDraw, ImageFont
from rdtdet_log import logger
from rdtdet_merge import merge_ocr_results, organize_cells_into_grid
import glob

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# OCR 모델 초기화
ocr = PaddleOCR(use_angle_cls=False, lang='korean')
THRESHOLD = 0.6

# 카테고리 ID 매핑
CATEGORY_MAPPING = {
    0: 'cell',
    1: 'table',
    2: 'row',
    3: 'column'
}

def init_detection_models():
    models = []
    model_configs = [
        ('mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py', r"C:\Users\opron\Downloads\epoch_10.pth")
    ]
    
    for cfg_file, checkpoint_file in model_configs:
        cfg = Config.fromfile(cfg_file)
        cfg.model.bbox_head.num_classes = 7
        model = init_detector(cfg, checkpoint_file, device='cpu')
        models.append(model)
    
    return models

# 이미지 처리 함수
def process_image(image_path, models):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"{image_path}에서 이미지를 불러오는데 실패했습니다.")
    
    results = []
    for model in models:
        result = inference_detector(model, original_img)
        detected_objects = []
        if hasattr(result, 'pred_instances'):
            bboxes = result.pred_instances.bboxes.cpu().numpy()
            labels = result.pred_instances.labels.cpu().numpy()
            scores = result.pred_instances.scores.cpu().numpy()
            
            for bbox, label, score in zip(bboxes, labels, scores):
                if score > THRESHOLD:
                    x1, y1, x2, y2 = map(int, bbox)
                    object_info = {
                        'bbox': (x1, y1, x2, y2),
                        'score': score,
                        'category': CATEGORY_MAPPING[label]
                    }
                    detected_objects.append(object_info)
        results.append(detected_objects)
    
    ocr_result = ocr.ocr(original_img, cls=False)
    
    return original_img, results, ocr_result[0]

# 결과 그리기 함수
def draw_results(image, objects, merged_ocr_results, model_index):
    result_image = image.copy()
    
    color_map = {
        'cell': (0, 255, 0),
        'table': (255, 0, 0),
        'row': (0, 0, 255),
        'column': (255, 255, 0)
    }
    
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        color = color_map.get(obj['category'], (0, 255, 0))
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result_image, f"{obj['category']} {obj['score']:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    font_path = "nanumgothic.ttf"
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    for merged_result, merge_info in merged_ocr_results:
        box, (text, confidence) = merged_result
        if confidence > THRESHOLD:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            
            if "+" in merge_info:
                draw.text((x1, y1 - font_size - 5), merge_info, font=font, fill=(255, 0, 0, 255))
            draw.text((x1, y1), text, font=font, fill=(0, 255, 0, 255))
    
    result_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return result_image

# 메인 실행 부분
if __name__ == "__main__":
    input_folder = "OCR"  # 입력 폴더 경로
    output_dir = "ocr_cell_results"  # 출력 폴더 경로
    os.makedirs(output_dir, exist_ok=True)

    # 모델 초기화
    models = init_detection_models()

    # 입력 폴더의 모든 이미지 파일 처리
    for img_path in glob.glob(os.path.join(input_folder, "*.png")) + glob.glob(os.path.join(input_folder, "*.jpg")):
        logger.info(f"Processing image: {img_path}")
        
        try:
            original_img, model_results, ocr_results = process_image(img_path, models)
            
            for model_index, detected_objects in enumerate(model_results):
                logger.info(f"Model {model_index + 1}: Detected {len(detected_objects)} objects")
                
                merged_ocr_results = merge_ocr_results(ocr_results, [obj for obj in detected_objects if obj['category'] == 'cell'])
                
                result_image = draw_results(original_img, detected_objects, merged_ocr_results, model_index)
                
                output_path = os.path.join(output_dir, f"{os.path.basename(img_path)}_model{model_index + 1}_result.png")
                cv2.imwrite(output_path, result_image)
                logger.info(f"Saved result for model {model_index + 1} to {output_path}")
                
                # 여기에 추가적인 분석이나 로깅을 추가할 수 있습니다
        
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")

    logger.info("All images processed.")
