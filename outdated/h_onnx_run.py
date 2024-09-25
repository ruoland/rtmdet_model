import onnxruntime
import numpy as np
import cv2

# ONNX Runtime 세션 생성
ort_session = onnxruntime.InferenceSession("rtmdet_table_detection_dynamic.onnx")

def preprocess_image(img_path):
    # 이미지 로드 및 전처리
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img_input = np.expand_dims(img_input, axis=0)
    img_input = img_input / 255.0
    return img_input, img.shape[:2]

def postprocess(outputs, orig_shape, input_shape):
    # 출력 후처리
    boxes, scores, labels = outputs

    # 바운딩 박스 크기를 원본 이미지 크기에 맞게 조정
    scale_x = orig_shape[1] / input_shape[3]
    scale_y = orig_shape[0] / input_shape[2]
    
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    return boxes, scores, labels

def draw_boxes(image, boxes, scores, labels, threshold=0.5):
    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"Class: {label}, Score: {score:.2f}"
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# 이미지 경로
img_path = "20220304_174813461_61050.jpeg"

# 이미지 전처리
input_data, original_shape = preprocess_image(img_path)

# 모델 입력 이름 가져오기
input_name = ort_session.get_inputs()[0].name

# 추론 실행
outputs = ort_session.run(None, {input_name: input_data})

# 후처리
boxes, scores, labels = postprocess(outputs, original_shape, input_data.shape)

# 결과 출력
print("Detection Results:")
for box, score, label in zip(boxes, scores, labels):
    if score > 0.5:  # 임계값 설정
        print(f"Class: {label}, Score: {score:.2f}, Box: {box}")

# 결과 시각화
original_image = cv2.imread(img_path)
result_image = draw_boxes(original_image, boxes, scores, labels)

# 결과 이미지 저장
cv2.imwrite("result_image.jpg", result_image)
print("Result image saved as 'result_image.jpg'")

# 클래스 이름 (예시, 실제 클래스 이름으로 수정 필요)
class_names = ['normal_cell', 'merged_cell', 'table']

# 상세 결과 출력
print("\nDetailed Results:")
for box, score, label in zip(boxes, scores, labels):
    if score > 0.5:  # 임계값 설정
        class_name = class_names[int(label)]
        print(f"Class: {class_name}, Score: {score:.2f}, Box: {box}")
