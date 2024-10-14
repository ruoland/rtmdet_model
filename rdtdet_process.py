import cv2
import multiprocessing as mp
from model_detection import detect_objects
from paddleocr import PaddleOCR

def process_rtmdet(resized_img, rtmdet, threshold, scale, queue):
    detected_objects = detect_objects(rtmdet, resized_img, threshold)
    for obj in detected_objects:
        obj['bbox'] = [int(coord / scale) for coord in obj['bbox']]
    queue.put(detected_objects)
def process_paddleocr(resized_img, scale, queue):
    ocr = PaddleOCR(use_angle_cls=False, lang='korean')
    ocr_result = ocr.ocr(resized_img, cls=False)
    scaled_result = []
    for i in range(len(ocr_result)):
        scaled_line = []
        for item in ocr_result[i]:
            box = item[0]
            scaled_box = [[int(x / scale), int(y / scale)] for x, y in box]
            scaled_line.append((scaled_box, item[1]))
        scaled_result.append(scaled_line)
    queue.put(scaled_result)

def process_image(img_path, rtmdet, threshold, target_size=3000):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("유효하지 않은 이미지입니다.")
    
    # 이미지 리사이즈
    height, width = img.shape[:2]
    scale = target_size / max(height, width)
    new_size = (int(width * scale), int(height * scale))
    resized_img = cv2.resize(img, new_size)
    
    # 멀티프로세싱 큐 생성
    rtmdet_queue = mp.Queue()
    ocr_queue = mp.Queue()
    
    # RTMDet 프로세스 시작
    rtmdet_process = mp.Process(target=process_rtmdet, args=(resized_img, rtmdet, threshold, scale, rtmdet_queue))
    rtmdet_process.start()
    
    # PaddleOCR 프로세스 시작
    ocr_process = mp.Process(target=process_paddleocr, args=(resized_img, scale, ocr_queue))
    ocr_process.start()
    
    # 프로세스 완료 대기
    rtmdet_process.join()
    ocr_process.join()
    
    # 결과 가져오기
    detected_objects = rtmdet_queue.get()
    ocr_result = ocr_queue.get()
    
    # OCR 결과 출력
    for line in ocr_result[0]:
        print(f"OCR 결과: {line[1][0]}, 신뢰도 : {line[1][1]}")
    
    return img, detected_objects, ocr_result[0]