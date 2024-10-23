from transformers import TableTransformerForObjectDetection, AutoFeatureExtractor
import torch
from PIL import Image

# 모델과 특징 추출기 로드
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")

# 이미지 로드
image = Image.open("table_testset\ocr4.jpeg")

# 이미지 처리
inputs = feature_extractor(images=image, return_tensors="pt")

# 모델 추론
outputs = model(**inputs)

# 결과 후처리
target_sizes = torch.tensor([image.size[::-1]])
results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

# 결과 출력
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(f"감지된 {model.config.id2label[label.item()]}, 신뢰도 {round(score.item(), 3)}, 위치 {box}")
