import torch
from mmengine.config import Config
from mmdet.apis import init_detector
import onnx

# 설정 파일 로드
cfg = Config.fromfile('configs/rtmdet/rtmdet_s_8xb32-300e_coco.py')

# 모델 설정 업데이트
cfg.model.bbox_head.num_classes = 3

# 모델 초기화
model = init_detector(cfg, 'epoch_9.pth', device='cpu')
model.eval()
# ONNX 모델 로드
onnx_model = onnx.load("rtmdet_table_detection_dynamic.onnx")

# 모델 구조 출력
print(onnx.helper.printable_graph(onnx_model.graph))
# 더미 입력 생성 (동적 크기 지원)
dummy_input = torch.randn(1, 3, 640, 640)

# 출력 이름 정의
output_names = ['boxes', 'scores', 'labels']

torch.onnx.export(model,
                  dummy_input,
                  "rtmdet_table_detection_dynamic.onnx",
                  input_names=['input'],
                  output_names=output_names,
                  dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                                'boxes': {0: 'batch_size', 1: 'num_detections'},
                                'scores': {0: 'batch_size', 1: 'num_detections'},
                                'labels': {0: 'batch_size', 1: 'num_detections'}},
                  opset_version=12,
                  do_constant_folding=True)



# ONNX 모델 확인
onnx_model = onnx.load("rtmdet_table_detection_dynamic.onnx")
onnx.checker.check_model(onnx_model)

print("ONNX model with dynamic input shape exported and checked successfully.")
