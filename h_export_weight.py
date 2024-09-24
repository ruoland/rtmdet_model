#체크포인트 파일에서 가중치만 뽑아내는 코드
#이 가중치를 불러오고, 다시 새로운 데이터셋으로 학습 시키자

import torch
import os
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
import mmcv
import cv2
import numpy as np

# 설정 파일 로드 (이 파일의 경로는 실제 위치에 맞게 조정해야 할 수 있습니다)
cfg = Config.fromfile('mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py')

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


cfg.model.bbox_head.num_classes = 2
# 클래스 이름 정의 (예시)
class_names = ['Cell', 'table']
cfg.model.bbox_head.num_classes = len(class_names)

base_path = './'
cfg.work_dir = os.path.join(base_path, 'rtmdet_table_detection')
checkpoint_file = r'C:\Users\opron\Downloads\work_dirs\rtmdet_tiny_custom\epoch_5.pth'

# 모델 초기화 (CPU 사용)
model = init_detector(cfg, checkpoint_file, device='cpu')
torch.save(model.state_dict(), 'epoch_5_weights.pth')
