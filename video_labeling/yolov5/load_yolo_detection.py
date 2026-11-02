import os
import torch
from yolov5.models.common import DetectMultiBackend


def _is_yolov11(model_name):
    """Check if the model name/path indicates a YOLOv11 model."""
    name_lower = os.path.basename(model_name).lower()
    return 'yolov11' in name_lower or 'yolo11' in name_lower


class UltralyticsModelWrapper:
    """Wrapper around ultralytics YOLO model for compatibility with YOLOv5 code.

    Callers should check model._is_ultralytics to pick the right inference path.
    Use model.predict() for high-level ultralytics inference (NMS included).
    """

    def __init__(self, model, stride, names, pt):
        self.model = model
        self.stride = stride
        self.names = names
        self.pt = pt
        self.jit = False
        self.onnx = False
        self.engine = False
        self._is_ultralytics = True

    def predict(self, img, conf=0.25, iou=0.45, classes=None, max_det=1000, verbose=False):
        """Run inference using ultralytics high-level API."""
        return self.model.predict(img, conf=conf, iou=iou, classes=classes,
                                  max_det=max_det, verbose=verbose)

    def eval(self):
        self.model.model.eval()
        return self

    def warmup(self, imgsz=None, half=True):
        if imgsz is not None:
            device = next(self.model.model.parameters()).device
            dummy = torch.zeros(*imgsz, device=device)
            if half:
                dummy = dummy.half()
            self.model.model(dummy)

    def half(self):
        self.model.model.half()
        return self

    def to(self, device):
        self.model.to(device)
        return self


def load_yolo_detection(model_name,
                        input_size=640,
                        data='yolov5/data/coco128.yaml',
                        dnn=False, half=True,
                        device="cpu"):
    """Load a YOLO detection model.

    Supports both YOLOv5 (via DetectMultiBackend) and YOLOv11 (via ultralytics).
    Model type is auto-detected from the weight filename.
    """
    if _is_yolov11(model_name):
        return _load_yolov11(model_name, input_size, half)
    else:
        return _load_yolov5(model_name, input_size, data, dnn, half, device)


def _load_yolov5(model_name, input_size, data, dnn, half, device):
    """Load YOLOv5 model via DetectMultiBackend (original path)."""
    model = DetectMultiBackend(model_name, device=torch.device("cuda"), data=data, fp16=half)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    # Mark as not ultralytics
    model._is_ultralytics = False
    if isinstance(input_size, int):
        model.warmup(imgsz=(1, 3, input_size, input_size))
    else:
        model.warmup(imgsz=(1, 3, input_size[0], input_size[1]))
    return model, stride, pt


def _load_yolov11(model_name, input_size, half):
    """Load YOLOv11 model via ultralytics API."""
    import ultralytics

    model = ultralytics.YOLO(model_name, verbose=False)
    model.fuse()
    model.to("cuda")

    stride = int(model.model.stride.max()) if hasattr(model.model, 'stride') else 32
    names = model.names if hasattr(model, 'names') else {}
    pt = True

    wrapper = UltralyticsModelWrapper(model, stride, names, pt)

    if isinstance(input_size, int):
        wrapper.warmup(imgsz=(1, 3, input_size, input_size), half=half)
    else:
        wrapper.warmup(imgsz=(1, 3, input_size[0], input_size[1]), half=half)

    return wrapper, stride, pt