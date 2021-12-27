import torch

import dataloader2
import dataloader_clean
import net

import loss
from tqdm import tqdm

import pytorch_utils as U

ds_base = dataloader_clean.PifpafDataset(
    image_paths="cache/coco_train/images/*.jpg",
    keypoint_paths="cache/hdd_data/features/*.keypoints.pt",
    pif_paths="/mnt/5E18698518695D51/Experiments/caching/cache_pifpaf_results/17/*.pt",
    # paf_paths="",
    device="cuda:0",
)

state = dict(model=None)


def transform(arg):
    if state["model"] is None:
        state["model"] = dataloader_clean.BatchPreprocess(2, device="cuda:0")
        print("ResNet is loaded")
    batch_process = state["model"]
    feats, kps, pif = arg
    if feats.dim() == 3:
        feats = feats.unsqueeze(0)
    if pif.dim() == 3:
        pif = pif.unsqueeze(0)
    feats = batch_process(feats, pif)
    return feats.to(dtype=torch.float16), kps.to(dtype=torch.float16)


ds_batch_preprocess = U.data.dmap(ds_base, transform)
ds_cached = U.data.dcache_tensor(ds_batch_preprocess, "/mnt/5E18698518695D51/Experiments/caching/res_features/{idx}.pt")

dl = U.data.DataLoader(ds_cached, batch_size=50, collate_fn=dataloader2.collate_fn)

N = net.CoordNetFirstOnly(597).cuda()
N.load_state_dict(torch.load("models/feature_sim/19.features.pth"))

S = net.SameNet(512).cuda()
S.load_state_dict(torch.load("models/feature_sim/19.classifier.pth"))

TRUE_POSITIVE = 0
FALSE_POSITIVE = 1
TRUE_NEGATIVE = 2
FALSE_NEGATIVE = 3
TOTAL_POSITIVE = 4
TOTAL_NEGATIVE = 5

POSITIVE = True
NEGATIVE = False

stats = 0
for feats, kps in tqdm(dl):
    feats = feats.squeeze_(1).float()
    kps = kps.float()
    embs = N(feats)

    stats += loss.evaluate(embs, kps, S).cpu().numpy()

    print(
        "TruePositiveRate:", (stats[TRUE_POSITIVE] / stats[TOTAL_POSITIVE]),
        ", TrueNegativeRate:", (stats[TRUE_NEGATIVE] / stats[TOTAL_NEGATIVE]),
        ", FalsePositiveRate:", (stats[FALSE_POSITIVE] / stats[TOTAL_POSITIVE]),
        ", FalseNegativeRate:", (stats[FALSE_NEGATIVE] / stats[TOTAL_NEGATIVE]),
        ", Stats:", stats
    )

print(stats)
