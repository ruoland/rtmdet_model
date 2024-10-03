from sympy import im
from rdtdet_merge import *
from rdtdet_missing_cells import add_missing_cells
from rdtdet_merge import merge_ocr_results, organize_cells_into_grid
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from PIL import Image, ImageDraw, ImageFont
from rdtdet_log import logger
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 모델 초기화
ocr = PaddleOCR(use_angle_cls=False, lang='korean')
cfg = Config.fromfile('mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py')
cfg.model.bbox_head.num_classes = 2
checkpoint_file = r"c:\Users\opron\Downloads\epoch_13.pth"
rtmdet = init_detector(cfg, checkpoint_file, device='cpu')

def process_image(image_path):
    # OpenCV로 이미지 읽기
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"{image_path}에서 이미지를 불러오는데 실패했습니다.")
    
  
    result = inference_detector(rtmdet, original_img)
    ocr_result = ocr.ocr(original_img, cls=False)
    
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
                    'score': score,
                    'label': label
                }
                detected_cells.append(cell_info)
    
    logger.info(f"{len(detected_cells)}개의 셀과 {len(ocr_result[0])}개의 OCR 결과를 감지했습니다.")
    return original_img, detected_cells, ocr_result[0]


def save_model_results(detected_cells, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for cell in detected_cells:
            f.write(f"bbox: {cell['bbox']}, score: {cell['score']}, label: {cell['label']}\n")
    logger.info(f"모델 인식 결과를 {output_path}에 저장했습니다.")
def draw_results(image, cells, merged_ocr_results):
    result_image = image.copy()
    
    for cell in cells:
        x1, y1, x2, y2 = map(int, cell['bbox'])
        color = (0, 0, 255) if cell.get('added', False) else (255, 0, 0)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
    
    font_path = "nanumgothic.ttf"
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    for merged_result, merge_info in merged_ocr_results:
        box, (text, confidence) = merged_result
        if confidence > 0.5:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            
            # 병합된 텍스트 표시
            if "+" in merge_info:
                draw.text((x1, y1 - font_size - 5), merge_info, font=font, fill=(255, 0, 0, 255))
            draw.text((x1, y1), text, font=font, fill=(0, 255, 0, 255))
    
    result_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return result_image

def draw_model_results(image, detected_cells):
    """모델이 감지한 셀을 이미지에 그립니다."""
    result_image = image.copy()
    for cell in detected_cells:
        x1, y1, x2, y2 = map(int, cell['bbox'])
        score = cell['score']
        
        # 셀 그리기
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 점수와 라벨 표시
        text = f"Score: {score:.2f}"
        cv2.putText(result_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_image
# 메인 실행 부분
if __name__ == "__main__":
    img_path = r"OCR\young\OCR4.png"
    output_dir = "ocr_cell"
    os.makedirs(output_dir, exist_ok=True)

    original_img, detected_cells, ocr_results = process_image(img_path)
    
    # 모델 인식 결과를 이미지에 그리기
    model_result_image = draw_model_results(original_img, detected_cells)
    model_result_path = os.path.join(output_dir, "model_detection_result.png")
    cv2.imwrite(model_result_path, model_result_image)
    logger.info(f"모델 감지 결과 이미지를 {model_result_path}에 저장했습니다.")
    
    logger.info("OCR 결과 병합을 시작합니다.")
    merged_ocr_results = merge_ocr_results(ocr_results, detected_cells)
    logger.debug(f"merged_ocr_results의 첫 번째 항목: {merged_ocr_results[0] if merged_ocr_results else 'None'}")
    # 기존 셀과 새로 추가된 셀을 합칩니다.
    # all_cells 생성 부분 수정
    all_cells = detected_cells.copy()
    for merged_result, _ in merged_ocr_results:
        if isinstance(merged_result, list) and len(merged_result) == 2:
            bbox, (text, confidence) = merged_result
            new_cell = {
                'bbox': bbox,
                'text': text,
                'score': confidence,
                'added': True
            }
            all_cells.append(new_cell)
    else:
        logger.warning(f"예상치 못한 merged_result 구조: {merged_result}")

# 로깅 추가
    logger.info(f"총 셀 개수: {len(all_cells)}")
    # 셀을 그리드로 정리
    ocr_result_list = [(box, (text, conf)) for box, (text, conf) in ocr_results]

    # 셀을 그리드로 정리
    organized_cells = organize_cells_into_grid(all_cells, ocr_result_list)    
    # 정리된 셀 결과 출력
    logger.info("정리된 셀 결과:")
    for row in organized_cells:
        row_text = []
        for cell in row:
            if cell:
                row_text.append(f"{cell.get('text', ''):<20}")
            else:
                row_text.append(" " * 20)
        logger.info(" ".join(row_text))
    
    result_image = draw_results(original_img, all_cells, merged_ocr_results)
    
    output_path = os.path.join(output_dir, "final_detection_result.png")
    cv2.imwrite(output_path, result_image)
    logger.info(f"최종 감지 결과가 {output_path}에 저장되었습니다.")

    logger.info(f"원본 셀 {len(detected_cells)}개를 감지했습니다.")
    logger.info(f"새로운 셀 {len(merged_ocr_results)}개를 추가했습니다.")
    logger.info(f"총 셀 개수: {len(all_cells)}")
    logger.info(f"정리된 그리드: {len(organized_cells)} 행 x {len(organized_cells[0])} 열")
