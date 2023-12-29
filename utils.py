import numpy as np
import torch
#from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

from torchmetrics.detection import mean_ap
mean_ap.warn_on_many_detections = False



def collate(x):
    """  [(img1, label1) , (img2, label2) ... ] - > [ img1, img2 , ...] , [label1, label2 , ...] """
    imgs, bboxes = list(zip(*x))
    return imgs, bboxes


def draw(pil, boxes, color="green"):
    draw = ImageDraw.Draw(pil)
    for bb in boxes:
        b = bb.int()
        draw.rectangle((tuple(b[:2]), tuple(b[2:])), outline=color, width=3)
    return None


def calculate_mAP(metric, gt, pred, results_accumulator=None):
    target = []
    for b_list in gt:
        target.append({
            'boxes': b_list,
            'labels': torch.ones([len(b_list)], dtype=torch.int32)
        })
    for p in pred:
        p['labels'] = torch.ones([len(p["boxes"])], dtype=torch.int32)
    metric.update(pred, target)
    if results_accumulator:
        results_accumulator.append(gt, pred)



def save_results(key, imgs, boxes, pred_above_th):
    for j, pil in enumerate(imgs):
        draw(pil, boxes[j], "green")
        if len(pred_above_th[j]["boxes"]) > 0:
            draw(pil, pred_above_th[j]["boxes"], "red")
        pil.save(f'tmp/{key}_{j}.jpg')


class ResultAccumulator(object):
    def __init__(self):
        self.df = pd.DataFrame(columns=['img_id', 'gt', 'x1', 'y1', 'x2', 'y2', 'cls', 'conf',"key"])
        self.frame_num = 0
        self.metric = mean_ap.MeanAveragePrecision()
        self.df_metric = pd.DataFrame()

    def append(self, gt, pred, key):
        assert len(pred) == len(gt)
        for num, p in enumerate(pred):
            for i, _ in enumerate(p["scores"]):
                l = [self.frame_num] + [0] + p["boxes"][i].tolist() + [1] + [p['scores'][i].item()] + [key]
                self.df.loc[len(self.df)] = l

            for b in gt[num]:
                l = [self.frame_num] + [1] + b.tolist() + [1, 1.0, key]
                self.df.loc[len(self.df)] = l
            self.frame_num += 1

    def update(self, gt, pred, key = 0):
        target = []
        for b_list in gt:
            target.append({
                'boxes': b_list,
                'labels': torch.ones([len(b_list)], dtype=torch.int32)
            })
        for p in pred:
            p['labels'] = torch.ones([len(p["boxes"])], dtype=torch.int32)
        self.metric.update(pred, target)
        self.append(gt, pred, key)

    def save_raw(self, path):
        self.df.to_csv(path, sep=';', encoding='utf-8')

    def save_metric(self, tag, path=""):
        mAP_score = self.metric.compute()
        for k in mAP_score:
            mAP_score[k] = mAP_score[k].item()

        mAP_score['dataset'] = tag
        df_dictionary = pd.DataFrame([mAP_score])
        self.df_metric = pd.concat([self.df_metric, df_dictionary], ignore_index=True)

        print(self.df_metric.head())
        self.df_metric.to_csv(f"{path}{tag}.csv", sep=';', encoding='utf-8')

