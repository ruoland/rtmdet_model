import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
def print_merged_ocr_results(merged_ocr_results):
    print("Merged OCR Results:")
    for i, (result, merge_info) in enumerate(merged_ocr_results):
        box, (text, confidence) = result
        print(f"Result {i+1}:")
        print(f"  Text: {text}")
        print(f"  Confidence: {confidence}")
        print(f"  Bounding Box: {box}")
        print(f"  Merge Info: {merge_info}")
        print()
def draw_results(image, objects, merged_ocr_results, class_filter=None, IMAGE_THRESHOLD=0.5):
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