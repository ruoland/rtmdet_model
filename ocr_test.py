import os
import cv2
import numpy as np
import easyocr
import pytesseract
from paddleocr import PaddleOCR
import time
from tabulate import tabulate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

def process_easyocr(image, confidence_threshold=0.5):
    reader = easyocr.Reader(['ko', 'en'])
    results = []
    try:
        easy_result = reader.readtext(image)
        for detection in easy_result:
            box, text, confidence = detection
            if confidence >= confidence_threshold:
                x, y = np.mean(box, axis=0)
                results.append((text, (x, y), confidence))
    except Exception as e:
        print(f"EasyOCR 처리 중 오류 발생: {e}")
    return results

def process_tesseract(image, confidence_threshold=50):
    results = []
    try:
        tess_result = pytesseract.image_to_data(image, lang='kor+eng', output_type=pytesseract.Output.DICT)
        for i in range(len(tess_result['text'])):
            if int(tess_result['conf'][i]) >= confidence_threshold:
                x = tess_result['left'][i] + tess_result['width'][i] / 2
                y = tess_result['top'][i] + tess_result['height'][i] / 2
                text = tess_result['text'][i]
                confidence = int(tess_result['conf'][i]) / 100
                results.append((text, (x, y), confidence))
    except Exception as e:
        print(f"Tesseract 처리 중 오류 발생: {e}")
    return results

def process_paddleocr(image, confidence_threshold=0.5):
    results = []
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='korean')
        paddle_result = ocr.ocr(image, cls=True)
        if paddle_result is not None and len(paddle_result) > 0:
            for line in paddle_result[0]:
                box, (text, confidence) = line
                if confidence >= confidence_threshold:
                    x, y = np.mean(box, axis=0)
                    results.append((text, (x, y), confidence))
    except Exception as e:
        print(f"PaddleOCR 처리 중 오류 발생: {e}")
    return results

def process_column_images(column_dir, use_easyocr=True, use_tesseract=True, use_paddleocr=True):
    results = {}
    times = {}
    
    for filename in sorted(os.listdir(column_dir)):
        if filename.endswith('.png'):
            image_path = os.path.join(column_dir, filename)
            image = cv2.imread(image_path)
            
            if use_easyocr:
                start_time = time.time()
                easy_results = process_easyocr(image)
                times.setdefault("EasyOCR", 0)
                times["EasyOCR"] += time.time() - start_time
                results.setdefault("EasyOCR", []).extend(easy_results)
            
            if use_tesseract:
                start_time = time.time()
                tess_results = process_tesseract(image)
                times.setdefault("Tesseract", 0)
                times["Tesseract"] += time.time() - start_time
                results.setdefault("Tesseract", []).extend(tess_results)
            
            if use_paddleocr:
                start_time = time.time()
                paddle_results = process_paddleocr(image)
                times.setdefault("PaddleOCR", 0)
                times["PaddleOCR"] += time.time() - start_time
                results.setdefault("PaddleOCR", []).extend(paddle_results)
    
    return results, times

def process_and_compare(column_dir, use_easyocr=True, use_tesseract=True, use_paddleocr=True):
    results, times = process_column_images(column_dir, use_easyocr, use_tesseract, use_paddleocr)
    
    print("\n=== 처리 시간 ===")
    for model, process_time in times.items():
        print(f"{model}: {process_time:.2f}초")
    
    # 결과를 모델별로 X,Y 순으로 정렬
    sorted_results = {}
    for model, texts in results.items():
        sorted_results[model] = sorted([(text.strip(), f"({x:.0f}, {y:.0f})", f"{conf:.2f}") for text, (x, y), conf in texts],
                                       key=lambda item: (float(item[1].split(',')[0][1:]), float(item[1].split(',')[1][:-1])))
    
    # 결과를 메모장에 저장
    with open("ocr_comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("=== 처리 시간 ===\n")
        for model, process_time in times.items():
            f.write(f"{model}: {process_time:.2f}초\n")
        f.write("\n")
        
        for model, texts in sorted_results.items():
            f.write(f"\n=== {model} 결과 ===\n")
            f.write(tabulate(texts, headers=["텍스트", "좌표", "신뢰도"], tablefmt="grid"))
            f.write("\n\n")
    
    # 콘솔에 결과 출력
    for model, texts in sorted_results.items():
        print(f"\n=== {model} 결과 ===")
        print(tabulate(texts, headers=["텍스트", "좌표", "신뢰도"], tablefmt="grid"))

# 사용 예시
column_dir = r'ocr_cell\cells'
process_and_compare(column_dir)
