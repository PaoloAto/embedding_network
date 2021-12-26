import torch
from torchvision import transforms
from tqdm import tqdm
import dataloader_clean
import pytorch_utils as U


ds = dataloader_clean.PifpafDataset(
    image_paths="cache/coco_val/images/*.jpg",
    keypoint_paths="cache/hdd_data_val/features/*.keypoints.pt",
    pif_paths="/mnt/5E18698518695D51/Experiments/caching_val/cache_pifpaf_results/17/*.pt",
    # paf_paths="",
    device="cuda:0"
)

batch_process = dataloader_clean.BatchPreprocess(2, device="cuda:0")


def transform(arg):
    feats, kps, pif = arg
    if feats.dim() == 3:
        feats = feats.unsqueeze(0)
    if pif.dim() == 3:
        pif = pif.unsqueeze(0)
    feats = batch_process(feats, pif)
    return feats.to(dtype=torch.float16), kps.to(dtype=torch.float16)


ds2 = U.data.dmap(ds, transform)
ds3 = U.data.dcache_tensor(ds2, "/mnt/5E18698518695D51/Experiments/caching_val/res_features/{idx}.pt")


print(len(ds3))
for i in tqdm(range(len(ds3))):
    ds3[i]
