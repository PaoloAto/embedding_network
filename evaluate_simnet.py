import torchvision
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

ds_images = U.data.dmap(ds_base, lambda arg: arg[0])

dl_images = U.data.DataLoader(ds_images, batch_size=50)
dl_features = U.data.DataLoader(ds_cached, batch_size=50, collate_fn=dataloader2.collate_fn)

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

CLASS_MATCH = 0
CLASS_MISMATCH = 1
TOTAL_MATCHES = 2

stats = 0
stats_class = 0

for idx, img, (feats, kps) in zip(tqdm(range(len(dl_images))), dl_images, dl_features):
    feats = feats.squeeze_(1).float()
    kps = kps.float()
    embs = N(feats)

    stat, stat_class = loss.evaluate_visualization(embs, kps, S, img)
    stats += stat.cpu().numpy()
    stats_class += stat_class.cpu().numpy()

    precision = stats[TRUE_POSITIVE] / (stats[TRUE_POSITIVE] + stats[FALSE_POSITIVE])
    recall = stats[TRUE_POSITIVE] / (stats[TRUE_POSITIVE] + stats[FALSE_NEGATIVE])
    accuracy = (stats[TRUE_POSITIVE] + stats[TRUE_NEGATIVE]) / (stats[TRUE_POSITIVE] + stats[TRUE_NEGATIVE] + stats[FALSE_NEGATIVE] + stats[FALSE_POSITIVE])
    f1 = 2 * precision * recall / (precision + recall)

    print(
        "TruePositiveRate:", (stats[TRUE_POSITIVE] / stats[TOTAL_POSITIVE]),
        ", TrueNegativeRate:", (stats[TRUE_NEGATIVE] / stats[TOTAL_NEGATIVE]),
        ", Precision:", precision,
        ", Recall:", recall,
        ", Accuracy:", accuracy,
        ", f1:", f1,
        ", Stats:", stats
    )

    print(
        ", ClassMatchRate:", (stats_class[CLASS_MATCH] / stats_class[TOTAL_MATCHES]),
        ", ClassMismatchRate:", (stats_class[CLASS_MISMATCH] / stats_class[TOTAL_MATCHES]),
        ", MatchStats:", stats_class
    )
