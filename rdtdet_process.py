import cv2
from model_detection import detect_objects
def process_image(img, rtmdet, ocr, threshold, target_size=3000):
    img = cv2.imread(img)
    if img is None:
        raise ValueError("유효하지 않은 이미지입니다.")
    
    # 이미지 리사이즈
    height, width = img.shape[:2]
    scale = target_size / max(height, width)
    new_size = (int(width * scale), int(height * scale))
    resized_img = cv2.resize(img, new_size)
    
    # 표 모델 실행 및 객체 감지
    detected_objects = detect_objects(rtmdet, resized_img, threshold)
    
    # OCR 실행
    ocr_result = ocr.ocr(resized_img, cls=False)
    
    for line in ocr_result:
        print(f"OCR 결과: {line[1][0]}, 신뢰도 : {line[1][1]}")
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
    
    return img, detected_objects, ocr_result[0]