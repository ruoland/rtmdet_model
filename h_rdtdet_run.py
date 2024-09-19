import os
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
import mmcv
import cv2
import numpy as np

# 기본 경로 설정
base_path = r"D:\Projects\OCR-LEARNIGN-PROJECT\OCR-PROJECT_OLD"
dataset_path = os.path.join(base_path, "yolox_table_dataset_simple233-0918")

# 설정 파일 로드 (이 파일의 경로는 실제 위치에 맞게 조정해야 할 수 있습니다)
cfg = Config.fromfile('configs/rtmdet/rtmdet_s_8xb32-300e_coco.py')

# 제공된 설정으로 cfg 업데이트
cfg.resume = True
cfg.max_epochs = 12
cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=cfg.max_epochs, val_interval=5)
cfg.val_cfg = dict(type='ValLoop')
cfg.test_cfg = dict(type='TestLoop')

cfg.train_dataloader.batch_size = 32
cfg.val_dataloader.batch_size = 32
cfg.test_dataloader = cfg.val_dataloader
cfg.env_cfg = dict(cudnn_benchmark=True)

cfg.data_root = dataset_path
cfg.train_dataloader.dataset.ann_file = os.path.join(dataset_path, 'train_annotations.json')
cfg.train_dataloader.dataset.data_root = os.path.join(dataset_path, 'train')
cfg.val_dataloader.dataset.ann_file = os.path.join(dataset_path, 'val_annotations.json')
cfg.val_dataloader.dataset.data_root = os.path.join(dataset_path, 'val')

cfg.model.bbox_head.num_classes = 3
# 클래스 이름 정의 (예시)
class_names = ['Cell', 'Merged_Cell', 'table']
cfg.model.bbox_head.num_classes = len(class_names)

cfg.work_dir = os.path.join(base_path, 'rtmdet_table_detection')
# 체크포인트 파일 경로
checkpoint_file = './epoch_9.pth'

# 모델 초기화 (CPU 사용)
model = init_detector(cfg, checkpoint_file, device='cpu')

# 테스트할 이미지 경로
img_path = r"D:\Projects\OCR-PROJECT\ocr\ocr2.jpg"

# 추론 실행
result = inference_detector(model, img_path)
# 점수 확인 (만약 'scores'가 있다면)
if hasattr(result.pred_instances, 'scores'):
    scores = result.pred_instances.scores.cpu().numpy()
    print("Score distribution:")
    print("Min score:", np.min(scores))
    print("Max score:", np.max(scores))
    print("Mean score:", np.mean(scores))

# 임계값 설정 (필요한 경우)
score_threshold = 0.6 # 매우 낮은 임계값으로 설정

# 결과 필터링 (필요한 경우)
if hasattr(result.pred_instances, 'scores'):
    mask = result.pred_instances.scores > score_threshold
    filtered_result = result.pred_instances[mask]
else:
    filtered_result = result.pred_instances

# 결과 시각화
img = mmcv.imread(img_path)
img = cv2.imread(img_path)

# 결과 그리기
def draw_detection(image, bbox, label, score, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    text = f"{class_names[label]}: {score:.2f}"
    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 결과 처리 및 그리기
if hasattr(result, 'pred_instances'):
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()

    for bbox, label, score in zip(bboxes, labels, scores):
        if score > score_threshold:  # 임계값 설정
            draw_detection(img, bbox, label, score)

# 결과 이미지 저장
cv2.imwrite('detected_result.png', img)

print(f"Detection result saved as 'detected_result.png'")

# 점수 분포 출력 (선택사항)
if len(scores) > 0:
    print("Score distribution:")
    print("Min score:", np.min(scores))
    print("Max score:", np.max(scores))
    print("Mean score:", np.mean(scores))
