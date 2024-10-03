# model_detection.py
import cv2
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector

def initialize_model(config_file, checkpoint_file, device='cpu'):
    cfg = Config.fromfile(config_file)
    cfg.model.bbox_head.num_classes = 7  # 클래스 수
    model = init_detector(cfg, checkpoint_file, device=device)
    return model

def detect_objects(model, image, threshold=0.7):
    result = inference_detector(model, image)
    detected_objects = []
    if hasattr(result, 'pred_instances'):
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        
        for bbox, label, score in zip(bboxes, labels, scores):
            if score > threshold:
                x1, y1, x2, y2 = map(int, bbox)
                object_info = {
                    'bbox': (x1, y1, x2, y2),
                    'score': score,
                    'label': label
                }
                detected_objects.append(object_info)
    return detected_objects
