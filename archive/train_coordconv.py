from pickle import FALSE
import dataloader
from net import CoordNet, Net, CoordNetFirstOnly
import loss
# import validation

import os
from os import name
import torch.utils.data as data
import numpy as np
import math
import torch

from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from models import Resnet
from glob import glob

from os import makedirs
from os.path import join, basename, splitext, exists
from tqdm import tqdm
import cache_coco_kp

from torchvision import transforms as T
from torchvision.ops import roi_pool

import pytorch_utils as U


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():

    loader1 = T.Compose([
        torch.load,
        T.Resize((81, 100)),
        lambda t: t.cuda()
    ])

    loader = T.Compose([
        torch.load,
        lambda t: t.cuda()
    ])

    run_type = "train"

    if (run_type == "overfit"):
        feature_ds = U.data.glob_files("cache/coco_train/features/*.features.pt", transform=loader)
        keypoint_ds = U.data.glob_files("cache/coco_train/features/*.keypoints.pt", transform=loader)
        name_ds = U.data.glob_files("cache/coco_train/features/*.features.pt")
        ds = U.data.dzip(feature_ds, keypoint_ds, name_ds)
    elif (run_type == "train"):
        feature_ds = U.data.glob_files("cache/hdd_data/features/*.features.pt", transform=loader)
        keypoint_ds = U.data.glob_files("cache/hdd_data/features/*.keypoints.pt", transform=loader)
        name_ds = U.data.glob_files("cache/hdd_data/features/*.features.pt")
        ds = U.data.dzip(feature_ds, keypoint_ds, name_ds)
    else:
        feature_ds = U.data.glob_files("/mnt/5E18698518695D51/Experiments/caching_val/features/*.features.pt", transform=loader)
        keypoint_ds = U.data.glob_files("/mnt/5E18698518695D51/Experiments/caching_val/features/*.keypoints.pt", transform=loader)
        name_ds = U.data.glob_files("/mnt/5E18698518695D51/Experiments/caching_val/features/*.features.pt")
        ds = U.data.dzip(feature_ds, keypoint_ds, name_ds)

    inv_channel = 0

    # N = CoordNet().cuda()
    N = CoordNetFirstOnly().cuda()
    optim = torch.optim.Adam(N.parameters(), lr=0.0001)

    epoch = 500

    for e in range(epoch):
        print("Epoch", e)
        record_losses = []

        check = 0

        # feat = features, kp = keypoints, fn = 'cache/coco_train/features/{img}.features.pt'
        for feat, kp, fn in tqdm(data.DataLoader(ds, batch_size=1, num_workers=0)):
            if (feat.shape[1] == 88):
                z = N(feat)
                # losses = dataloader.loss_fn(z, kp, e)
                losses = loss.loss_fn(z, kp)

                if (math.isinf(losses) == True or math.isnan(losses) == True):
                    check += 1
                    # temp_list.append(fn)
                else:
                    # All gradient computation
                    optim.zero_grad()
                    losses.backward()
                    optim.step()

                    record_losses.append(losses.item())
            else:
                inv_channel += 1

        print("Check: ", check)
        print("Loss:", sum(record_losses)/len(record_losses))
        writer.add_scalar('cdist/loss', sum(record_losses)/len(record_losses), e)
        torch.save(N.state_dict(), f"models/coordconv_firstLayer/{e:02}.pth")
        # if (e % 5 == 0):
        #     validation.val()


if __name__ == '__main__':
    main()
