import os
import torch
from .common import DetectMultiBackend
from .common_emb import Backbone


def load_yolo_detection(model_name,
                        input_size=640,
                        data_name="coco.yaml",
                        dnn=False, half=True,
                        device="cpu"):
    """Load a YOLO detection model.

    Supports both YOLOv5 (via DetectMultiBackend) and YOLOv11 (via ultralytics).
    Model type is auto-detected from the weight filename.

    To use YOLOv11, pass a weight path containing 'yolov11' or 'yolo11',
    e.g. './weights/yolov11n.pt'.
    To use YOLOv5 (default), pass any other weight path,
    e.g. './video_labeling/yolov5x.pt'.
    """
    if _is_yolov11(model_name):
        return _load_yolov11(model_name, input_size, half)
    else:
        return _load_yolov5(model_name, input_size, data_name, dnn, half, device)


def _is_yolov11(model_name):
    """Check if the model name/path indicates a YOLOv11 model."""
    name_lower = os.path.basename(model_name).lower()
    return 'yolov11' in name_lower or 'yolo11' in name_lower


def _load_yolov5(model_name, input_size, data_name, dnn, half, device):
    """Load YOLOv5 model via DetectMultiBackend (original path)."""
    model = DetectMultiBackend(model_name, device=torch.device("cuda"))
    stride, names, pt, jit, onnx, engine = (
        model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    )
    # Reset cached grids — checkpoints saved after a forward pass store grid
    # and anchor_grid in expanded form which causes shape mismatches in _make_grid.
    # Also ensure 'inplace' attr exists (missing in some older checkpoints).
    for m in model.model.modules():
        if type(m).__name__ == 'Detect':
            if hasattr(m, 'grid'):
                delattr(m, 'grid')
            m.grid = [torch.zeros(1)] * m.nl
            if hasattr(m, 'anchor_grid'):
                delattr(m, 'anchor_grid')
            m.anchor_grid = [torch.zeros(1)] * m.nl
            if not hasattr(m, 'inplace'):
                m.inplace = True
    model.warmup(imgsz=(1, 3, input_size, input_size), half=half)
    # Mark this as not ultralytics
    model._is_ultralytics = False
    return model, stride, pt


class UltralyticsModelWrapper:
    """Wrapper around ultralytics YOLO model to provide a compatible interface.

    The key difference: ultralytics models handle NMS internally and return
    Results objects with .boxes attributes, rather than raw prediction tensors.
    Callers should check model._is_ultralytics to determine which inference
    path to use.
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
        """Run inference using ultralytics high-level API.

        Returns a list of Results objects (one per image in the batch).
        Each result has .boxes with .xyxy, .conf, .cls attributes.
        """
        return self.model.predict(img, conf=conf, iou=iou, classes=classes,
                                  max_det=max_det, verbose=verbose)

    def eval(self):
        self.model.model.eval()
        return self

    def warmup(self, imgsz=None, half=True):
        """Warmup the model."""
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


def _load_yolov11(model_name, input_size, half):
    """Load YOLOv11 model via ultralytics API."""
    import ultralytics

    model = ultralytics.YOLO(model_name, verbose=False)
    model.fuse()
    model.to("cuda")

    # Extract stride and names from the underlying model
    stride = int(model.model.stride.max()) if hasattr(model.model, 'stride') else 32
    names = model.names if hasattr(model, 'names') else {}
    pt = True  # PyTorch model

    wrapper = UltralyticsModelWrapper(model, stride, names, pt)

    # Warmup
    if isinstance(input_size, int):
        wrapper.warmup(imgsz=(1, 3, input_size, input_size), half=half)
    else:
        wrapper.warmup(imgsz=(1, 3, input_size[0], input_size[1]), half=half)

    return wrapper, stride, pt


def load_embedding_detection(cfg):
    model = Backbone(cfg, num_classes=10).cuda()
    model.load_param(cfg.MODEL.EMB_WEIGHT)
    model.eval()
    model.warmup(torch.rand([1, 3, 384, 384]))

    return model