from torchvision.transforms import ToPILImage
from utils import collate, save_results, ResultAccumulator
from nets.OWLVitWrapper import OWLVitWrapper, OWL2Wrapper
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchguns.THGPDataset import THGPDataset, HGPDataset
from torchguns.YouTubeGddDataset import YouTubeGddDataset
from torchguns.USRTDataset import USRTDataset
import torch

datasets = {
    'THGP': THGPDataset(root="data", train=False, download=True, transform=ToPILImage()),
    'HGP': HGPDataset(root="data", train=False, download=True, transform=ToPILImage()),
    'YouTube-GDD': YouTubeGddDataset(root="data", train=False, download=True, transform=ToPILImage()),
    'USRT': USRTDataset(root="data", train=None, download=True, transform=ToPILImage())
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# weapon_detector = OWL2Wrapper(threshold=0.0001, device="cuda:0")
weapon_detector = OWLVitWrapper(threshold=0.0001, device=device)

for name, d in datasets.items():
    print(name)
    results_accumulator = ResultAccumulator()
    dataloader = DataLoader(d, batch_size=64, num_workers=0, collate_fn=collate)
    for i, (imgs, boxes) in enumerate(tqdm(dataloader)):
        results = weapon_detector.predict(imgs)
        pred_above_th = results.nms(threshold=0.1).results
        results_accumulator.update(boxes, pred_above_th)

    results_accumulator.save_raw(f"results/{name}_raw.csv")
    results_accumulator.save_metric(name, "results/")
