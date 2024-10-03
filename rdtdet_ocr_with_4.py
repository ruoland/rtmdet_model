import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from rdtdet_log import logger
from model_detection import initialize_model, detect_objects
from rdtdet_merge import merge_ocr_results, organize_cells_into_grid

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 모델 초기화
ocr = PaddleOCR(use_angle_cls=False, lang='korean')
config_file = 'mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
checkpoint_file = r"c:\Users\opron\Downloads\epoch_10.pth"
rtmdet = initialize_model(config_file, checkpoint_file)
IMAGE_THRESHOLD = 0.5

def process_image(image_path):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"{image_path}에서 이미지를 불러오는데 실패했습니다.")
    
    detected_objects = detect_objects(rtmdet, original_img, IMAGE_THRESHOLD)
    ocr_result = ocr.ocr(original_img, cls=False)
    
    logger.info(f"{len(detected_objects)}개의 객체와 {len(ocr_result[0])}개의 OCR 결과를 감지했습니다.")
    return original_img, detected_objects, ocr_result[0]

def draw_results(image, objects, merged_ocr_results, class_filter=None):
    result_image = image.copy()
    colors = [
        (255, 0, 0),     # cell
        (0, 255, 0),     # table
        (0, 0, 255),     # row
        (255, 255, 0),   # column
        (255, 0, 255),   # merged_cell
        (0, 255, 255),   # overflow_cell
        (255, 165, 0)    # merged_overflow_cell
    ]
    class_names = ['cell', 'table', 'row', 'column', 'merged_cell', 'overflow_cell', 'merged_overflow_cell']
    
    for obj in objects:
        if class_filter is not None and obj['label'] != class_filter:
            continue
        x1, y1, x2, y2 = map(int, obj['bbox'])
        color = colors[obj['label']]
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{class_names[obj['label']]}: {obj['score']:.2f}"
        cv2.putText(result_image, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if class_filter is None or class_filter in [0, 4, 5]:  # cell과 merged_cell 및 overflow_cell에 대해서만 OCR 결과 표시
        font_path = "nanumgothic.ttf"
        font_size = 20
        font = ImageFont.truetype(font_path, font_size)
        img_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        for merged_result, merge_info in merged_ocr_results:
            box, (text, confidence) = merged_result
            if confidence > IMAGE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                
                if "+" in merge_info:
                    draw.text((x1, y1 - font_size - 5), merge_info, font=font, fill=(255, 0, 0))
                draw.text((x1, y1), text, font=font, fill=(0, 255, 0))
        
        result_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return result_image
from torch import nn
import torch

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return logits / self.temperature
def display_results(image, objects, merged_ocr_results):
    class_filter = None
    window_name = 'Detection Result'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    scale = 1.0  # 초기 확대/축소 비율 설정

    while True:
        result_image = draw_results(image, objects, merged_ocr_results, class_filter)

        # 현재 비율에 따라 이미지 크기 조정
        resized_image = cv2.resize(result_image, 
                                    (int(result_image.shape[1] * scale), 
                                     int(result_image.shape[0] * scale)))

        # 화면 해상도에 맞춰 창 크기 조정
        cv2.imshow(window_name, resized_image)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('0'):
            class_filter = 0  # cell
        elif key == ord('1'):
            class_filter = 1  # table
        elif key == ord('2'):
            class_filter = 2  # row
        elif key == ord('3'):
            class_filter = 3  # column
        elif key == ord('4'):
            class_filter = 4  # merged_cell
        elif key == ord('5'):
            class_filter = 5  # overflow_cell
        elif key == ord('6'):
            class_filter = 6  # merged_overflow_cell
        elif key == ord('a'):
            class_filter = None  # all classes
        elif key == ord('+'):  # 확대
            scale *= 1.1  # 비율을 10% 증가
        elif key == ord('-'):  # 축소
            scale /= 1.1  # 비율을 10% 감소
    
    cv2.destroyAllWindows()




# 메인 실행 부분
if __name__ == "__main__":
    img_path = r"joeun.jpg"
    output_dir = "ocr_cell"
    os.makedirs(output_dir, exist_ok=True)

    original_img, detected_objects, ocr_results = process_image(img_path)
    
    logger.info("OCR 결과 병합을 시작합니다.")
    detected_cells = [obj for obj in detected_objects if obj['label'] == 0]  # 셀만 선택
    merged_ocr_results = merge_ocr_results(ocr_results, detected_cells)
    
    # 기존 셀과 새로 추가된 셀을 합칩니다.
    all_cells = detected_cells.copy()
    for merged_result in merged_ocr_results:
        if isinstance(merged_result[0], list) and len(merged_result[0]) == 2:
            bbox , (text , confidence) = merged_result[0]
            new_cell = {
                'bbox': bbox,
                'text': text,
                'score': confidence,
                'added': True,
                'label': 0 
            }
            all_cells.append(new_cell)
    
    logger.info(f"총 셀 개수: {len(all_cells)}")
    
    # 셀을 그리드로 정리합니다.
    ocr_result_list = [(box , (text , conf)) for box , (text , conf) in ocr_results]
    organized_cells = organize_cells_into_grid(all_cells , ocr_result_list)    
    
    # 정리된 셀 결과 출력
    logger.info("정리된 셀 결과:")
    for row in organized_cells:
        row_text = []
        row_count = 10
        for cell in row:
            if cell:
                row_text.append(f"{cell.get('text' , ''):<20}")
            elif row_count <= 10:
                row_text.append(" " * 20)
            else:
                break
    # 결과 표시합니다.
    display_results(original_img , detected_objects , merged_ocr_results)
    
    logger.info(f"원본 객체 {len(detected_objects)}개를 감지했습니다.")
    logger.info(f"셀 {len(detected_cells)}개를 감지했습니다.")
    logger.info(f"새로운 셀 {len(merged_ocr_results)}개를 추가했습니다.")
    logger.info(f"총 셀 개수: {len(all_cells)}")
    logger.info(f"정리된 그리드: {len(organized_cells)} 행 x {len(organized_cells[0])} 열")
