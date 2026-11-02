from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
import numpy as np
import torch
from utils.torch_utils import select_device
from models.yolo import Model

class TeacherModel(object):
    def __init__(self, conf_thres=0.5, iou_thres=0.3, imgsz=640, training=False):
        self.model = None
        self.device = None
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.training = training

    def init_model(self, weights, device, nc):
        self.device = select_device(device)

        # Load checkpoint
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        ckpt_model = ckpt['model'].float() if 'model' in ckpt else ckpt.float()

        # Rebuild model with correct classes
        self.model = Model(ckpt_model.yaml, ch=3, nc=nc)

        # Exclude detect layer weights
        state_dict = ckpt_model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if 'model.24.m' not in k}  
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device).float()
        self.model.eval() if not self.training else self.model.train()
        self.stride = int(self.model.model[-1].stride.max())
        self.nc = nc

    def generate_batch_targets(self, imgs, tar_size=[640, 640]):
        targets = []
        with torch.no_grad():
            if self.training:
                preds = self.model(imgs)
            else:
                preds = self.model(imgs)[0]
        if not self.training:

            for img_id in range(imgs.shape[0]):

                pred = preds[img_id:img_id+1]
                pred = non_max_suppression(
                    pred, self.conf_thres, self.iou_thres, distill=True, agnostic=False)

                for det in pred:  # detections per image
                    gn = torch.tensor(tar_size)[[1, 0, 1, 0]]
                    if len(det):
                        # Rescale boxes from img_size to img0 size
                        det[:, :4] = scale_coords(
                            imgs[img_id].unsqueeze(0).shape[2:], det[:, :4], tar_size).round()

                        for value in reversed(det):
                            xyxy, cls_id = value[:4], value[5]
                            logits = value[-self.nc:].logit().tolist()
                            xywh = (xyxy2xywh(torch.tensor(xyxy.cpu()).view(1, 4)
                                              ) / gn).view(-1).tolist()  # normalized xywh
                            line = [img_id, int(cls_id)]
                            line.extend(xywh)
                            line.extend(logits)
                            targets.append(line)

            return torch.tensor(np.array(targets), dtype=torch.float32), None
        else:
            return [], preds


# if __name__ == '__main__':

    # teacher = TeacherModel(conf_thres=0.0001)

    # teacher.init_model('weights/yolov5l.pt', '0', 365)
    # import cv2
    # imgs = torch.tensor([cv2.imread('data/images/bus.jpg')]).to(teacher.device) # PIL
    # targets = teacher.generate_batch_targets(imgs)
    # print(targets)
