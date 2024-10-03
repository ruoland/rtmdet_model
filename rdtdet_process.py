import cv2
from model_detection import detect_objects
def process_image(image_path, rtmdet, ocr, threshold, target_size=1000):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"{image_path}에서 이미지를 불러오는데 실패했습니다.")
    
    # 이미지 리사이즈
    height, width = original_img.shape[:2]
    scale = target_size / max(height, width)
    new_size = (int(width * scale), int(height * scale))
    resized_img = cv2.resize(original_img, new_size)
    
    # 표 모델 실행 및 객체 감지
    detected_objects = detect_objects(rtmdet, resized_img, threshold)
    
    # OCR 실행
    ocr_result = ocr.ocr(resized_img, cls=False)
    
    # 원본 크기로 좌표 변환
    for obj in detected_objects:
        obj['bbox'] = [int(coord / scale) for coord in obj['bbox']]
    
    for i in range(len(ocr_result)):
        for j in range(len(ocr_result[i])):
            box = ocr_result[i][j][0]
            ocr_result[i][j] = (
                [[int(x / scale), int(y / scale)] for x, y in box],
                ocr_result[i][j][1]
            )
    
    return original_img, detected_objects, ocr_result[0]
