import cv2
import numpy as np
import os

def save_image(image, filename):
    cv2.imwrite(filename, image)
    print(f"Saved image: {filename}")

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_binary(image, threshold=127):
    _, binary = cv2.threshold(to_grayscale(image), threshold, 255, cv2.THRESH_BINARY)
    return binary

def apply_gaussian_blur(image, kernel_size=(5,5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def equalize_histogram(image):
    return cv2.equalizeHist(to_grayscale(image))

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def detect_lines(image):
    lines = cv2.HoughLinesP(image, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    return lines

def draw_lines(image, lines):
    line_image = np.copy(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return line_image
import cv2
import numpy as np
import os

def save_image(image, filename):
    cv2.imwrite(filename, image)
    print(f"Saved image: {filename}")

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_binary(image, threshold=127):
    _, binary = cv2.threshold(to_grayscale(image), threshold, 255, cv2.THRESH_BINARY)
    return binary

def adaptive_threshold(image):
    gray = to_grayscale(image)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def canny_edge(image, low_threshold=100, high_threshold=200):
    gray = to_grayscale(image)
    return cv2.Canny(gray, low_threshold, high_threshold)

def morphology_operation(image, operation='open', kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'open':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        return image

def detect_lines_hough(image):
    gray = to_grayscale(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    result = image.copy()
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return result

def process_image(image_path):
    # 이미지 읽기
    original = cv2.imread(image_path)
    
    # 이미지 크기 확인 및 조정
    height, width = original.shape[:2]
    if height >= 2000 and width >= 2000:
        new_height = int(height * 0.5)
        new_width = int(width * 0.5)
        original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 전처리 적용
    gray = to_grayscale(original)
    binary = to_binary(original)
    adaptive = adaptive_threshold(original)
    edges = canny_edge(original)
    morph_open = morphology_operation(binary, 'open')
    morph_close = morphology_operation(binary, 'close')
    hough_lines = detect_lines_hough(original)
    
    # 결과 저장
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join('output', base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    save_image(original, os.path.join(output_dir, f'{base_name}_original.png'))
    save_image(gray, os.path.join(output_dir, f'{base_name}_gray.png'))
    save_image(binary, os.path.join(output_dir, f'{base_name}_binary.png'))
    save_image(adaptive, os.path.join(output_dir, f'{base_name}_adaptive.png'))
    save_image(edges, os.path.join(output_dir, f'{base_name}_edges.png'))
    save_image(morph_open, os.path.join(output_dir, f'{base_name}_morph_open.png'))
    save_image(morph_close, os.path.join(output_dir, f'{base_name}_morph_close.png'))
    save_image(hough_lines, os.path.join(output_dir, f'{base_name}_hough_lines.png'))

def process_all_images(input_dir):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            process_image(image_path)

# 메인 실행 코드
if __name__ == "__main__":
    input_directory = r"D:\Projects\OCR-PROJECT\ocr"
    process_all_images(input_directory)
    print("All processing steps completed. Check the 'output' folder for results.")
