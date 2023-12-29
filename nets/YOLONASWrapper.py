import os
import torch
from ultralytics import NAS


class YOLONASWrapper(object):
    def __init__(self, out="xyxy"):
        self.model = NAS('yolo_nas_m')
        """
        Don't work :(
        # https: // github.com / ultralytics / ultralytics / blob / main / ultralytics / cfg / default.yaml
        # self.model.model.args['mode'] = "predict"
        # self.model.model.args['conf'] = 0.9 # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
        # self.model.model.args['iou'] = 0.5 # 0.7   (float) intersection over union (IoU) threshold for NMS
        # self.model.model.args['max_det'] = 50  # 300  # (int) maximum number of detections per image

        # self.model.model.args['verbose'] =False
        """
        # x = self.model.args

        assert out in ["xyxy", "xywh", "xywhn"]
        self.out = out

    @torch.inference_mode()
    def __call__(self, pil):
        results = self.model.predict(pil,
                                     verbose=False,
                                     save=False,
                                     conf=0.6,
                                     iou=0.5,
                                     max_det=50)
        for res in results:
            r = res.boxes
            # print(results)
            if self.out == "xywh":
                detections = r.xywh
            elif self.out == "xywhn":
                detections = r.xywhn
            else:
                detections = r.xyxy
            detections = detections.cpu()
            rows = torch.empty((0, 5))
            for i in range(len(detections)):
                if r.cls[i] == 0: # and r.conf[i] > 0.9:  # person
                    cls = r.cls[i].unsqueeze(0).cpu()
                    d = detections[i]
                    row = torch.cat((cls, d))
                    rows = torch.cat((rows, row.unsqueeze(0)))

        return rows
