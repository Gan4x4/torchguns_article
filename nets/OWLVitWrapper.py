import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.ops import nms
from transformers import Owlv2Processor, Owlv2ForObjectDetection


class OWLVitWrapper(torch.nn.Module):
    def __init__(self,
                 texts=["gun", "rifle", "carabine", "shotgun", "pistol", "handgun", "revolver"],
                 device="cpu",
                 threshold=0.001):
        super().__init__()
        self.device = device
        self.threshold = threshold
        with torch.inference_mode():  # help to avoid memory leak
            self.texts = texts
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", )
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", )
            self.model.eval()
            self.model.to(self.device)

    def forward(self, x):
        results = self.predict(x)
        if results is None:
            return 0
        return results.get_prediction()

    def predict(self, batch):
        """
          return raw values for all classes
          576 bbox (xyxy) in total
          num_classes = len(texts)
        """
        with torch.inference_mode():
            batch_size = len(batch)
            if batch_size > 0:
                inputs = self.processor(text=[self.texts * batch_size], images=batch, return_tensors="pt")
                inputs = inputs.to(self.device)

                outputs = self.model(**inputs)

                # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
                # Each element of target_sizes must contain the size (h, w) of each image of the batch
                target_sizes = []
                for im in batch:
                    target_sizes.append(im.size[::-1])
                target_sizes = torch.Tensor(target_sizes).to(self.device)

                # fixing device error
                # outputs.pred_boxes = outputs.pred_boxes.cpu()

                # Convert outputs (bounding boxes and class logits) to COCO API
                # results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)
                results = self.processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=self.threshold
                )
            else:
                return None

            return ResultWrapper(results, self.texts)


class ResultWrapper(object):

    # results model output to batch of images (patches from one image)
    def __init__(self, results, texts):
        self.results = results
        self.texts = texts

    def get_scores_for_weapons(self):
        return self.probs().max(0).values

    def get_scores_for_patches(self):
        # p = self.probs()
        return self.probs().max(1).values

    def probs(self):
        probs = []
        for res_in_dict in self.results:
            scores = []
            indexes = []
            for class_num, name in enumerate(self.texts):
                tmp = res_in_dict['labels']
                idx = (tmp == class_num).nonzero().flatten()  # analog of my_list.index(a)
                if len(res_in_dict['scores'][idx]) > 0:
                    max_score = max(res_in_dict['scores'][idx])
                else:
                    max_score = 0
                    idx_of_max_value = 0
                scores.append(max_score)
            probs.append(scores)
        probs = torch.Tensor(probs).cpu()
        return probs  # batch_size x class_nm

    def get_bbox_with_max_score(self):
        index_in_batch = torch.argmax(self.get_scores_for_patches())
        return self.results[index_in_batch]['boxes']

    def get_prediction(self):
        scores = self.get_scores_for_weapons()
        return max(scores)
        # return sum(scores)

    def get_top(self, n):
        """ return prediction with max score for n-th image in the batch """
        one_result = self.results[n]
        score, idx = torch.max(one_result['scores'], dim=0)
        idx = idx.item()
        bbox = one_result['boxes'][idx]
        class_num = one_result['labels'][idx].item()
        return score.item(), bbox, self.texts[class_num]

    def filter(self, threshold):
        # TODO not work, dimension issue
        callback = lambda x: (x['scores'] > threshold).nonzero().cpu().squeeze(0)
        return self.filter_(callback)

    def filter_(self, callback):
        out = []
        for e in self.results:
            idx = callback(e).cpu()
            d = {}
            for key in e:
                d[key] = e[key].cpu()[idx]
                # d[key] = d[key].squeeze(1),
            out.append(d)
        return ResultWrapper(out, self.texts)

    def nms(self, threshold):
        """Apply NMS algorith"""
        callback = lambda x: nms(x["boxes"], x["scores"], iou_threshold=threshold).cpu()
        return self.filter_(callback)


class OWL2Wrapper(OWLVitWrapper):
    def __init__(self,
                 texts=["gun", "rifle", "carabine", "shotgun", "pistol", "handgun", "revolver"],
                 device="cpu",
                 threshold=0.0001):
        super().__init__()
        self.device = device
        self.threshold = threshold
        with torch.inference_mode():  # help to avoid memory leak
            self.texts = texts
            self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.model.eval()
            self.model.to(self.device)
