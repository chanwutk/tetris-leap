
import os
import json
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import warnings

# Use relative paths or absolute paths as discovered
DEFAULT_RETINANET_DIR = '/data/alenachao/data/otif-dataset/retinanet/b3d'
DEFAULT_DETECTRON_CONFIG_DIR = '/data/alenachao/otif/python/detectron2/configs'

class RetinaNetDetector:
    def __init__(self, device='0', model_dir=None, detectron_config_dir=None):
        # Handle various device string formats
        device_str = str(device)

        if device_str.isdigit():
            self.device = f'cuda:{device_str}'
        elif device_str == 'cuda':
            # Plain "cuda" is not recognized by detectron2; use explicit index.
            # CUDA_VISIBLE_DEVICES remaps selected GPU to index 0 inside the process.
            self.device = 'cuda:0'
        elif device_str.startswith('cuda'):
            self.device = device_str
        elif device_str == 'cpu':
            self.device = 'cpu'
        else:
            self.device = 'cuda:0'

        model_dir = model_dir or DEFAULT_RETINANET_DIR
        self.detectron_config_dir = detectron_config_dir or DEFAULT_DETECTRON_CONFIG_DIR
        self.config_file = os.path.join(model_dir, 'config_refined.json')
        self.weights_file = os.path.join(model_dir, 'retinanet_b3d.pth')

        self.predictor = self._init_predictor()
        print(f"RetinaNetDetector initialized on {self.device}")

    def _init_predictor(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        with open(self.config_file) as fp:
            config = json.load(fp)

        cfg = get_cfg()

        # Resolve the base detectron2 config. Try in order:
        # 1. Explicit yaml override stored on self (set via config_yaml constructor arg)
        # 2. detectron2's built-in model_zoo (works without a locally cloned detectron2 repo)
        # 3. Local detectron_config_dir on disk
        base_config_name = config.get('config', '')  # e.g. "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        resolved = False

        if getattr(self, 'config_yaml', None):
            cfg.merge_from_file(self.config_yaml)
            resolved = True

        if not resolved:
            try:
                from detectron2 import model_zoo
                cfg.merge_from_file(model_zoo.get_config_file(base_config_name))
                resolved = True
            except Exception:
                pass

        if not resolved:
            local_path = os.path.join(self.detectron_config_dir, base_config_name)
            assert os.path.exists(local_path), (
                f"Could not find detectron2 config '{base_config_name}' via model_zoo "
                f"or at local path '{local_path}'. "
                f"Pass --detectron_config_dir pointing to your detectron2/configs directory, "
                f"or --detectron_config_yaml with an explicit .yaml path."
            )
            cfg.merge_from_file(local_path)

        # Override weights with actual path
        cfg.MODEL.WEIGHTS = self.weights_file

        # Apply overrides from json
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.get('num_classes', 1)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.get('score_threshold', 0.5)
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config.get('score_threshold', 0.5)
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.get('nms_threshold', 0.5)
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = config.get('nms_threshold', 0.5)
        cfg.TEST.DETECTIONS_PER_IMAGE = config.get('detections_per_image', 100)
        if 'anchor_generator_sizes' in config:
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['anchor_generator_sizes']

        cfg.MODEL.DEVICE = self.device

        return DefaultPredictor(cfg)

    def detect(self, img0):
        # RetinaNet expects BGR image (OpenCV format)
        outputs = self.predictor(img0)
        
        instances = outputs["instances"].to("cpu")
        bboxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        leap_bboxes = []
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            conf = scores[i]
            cls = classes[i]
            
            # Map class 0 (likely car in this 1-class model) to 2 (car in LEAP)
            if cls == 0:
                cls_mapped = 2 # 'car'
            else:
                cls_mapped = int(cls)
                
            leap_bboxes.append([x1, y1, x2, y2, float(conf), cls_mapped])
            
        return img0, leap_bboxes
