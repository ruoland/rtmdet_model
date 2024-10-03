from PIL import Image
import numpy as np

def preprocess_image_to_grayscale(image_path, output_path=None):
    # 이미지 열기
    with Image.open(image_path) as img:
        # 흑백으로 변환
        gray_img = img.convert('L')
        
        # NumPy 배열로 변환
        img_array = np.array(gray_img)
        
        # 이진화 (선택적)
        # threshold = 128  # 임계값 설정
        # binary_img_array = (img_array > threshold).astype(np.uint8) * 255
        
        # 다시 PIL Image로 변환
        # processed_img = Image.fromarray(binary_img_array)
        processed_img = Image.fromarray(img_array)
        
        # 결과 저장 또는 반환
        if output_path:
            processed_img.save(output_path)
            print(f"Processed image saved to {output_path}")
        
        return processed_img

# 사용 예시
input_image_path = "original.png"
output_image_path = "grayscale_image.jpg"

grayscale_image = preprocess_image_to_grayscale(input_image_path, output_image_path)
