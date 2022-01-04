import dataloader2
import pytorch_utils as U
import net
import loss

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

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def main():

    ds = dataloader2.LoadCachedFeatureDataset("cache/hdd_data/features/*.features.pt", "cache/hdd_data/features/*.keypoints.pt", device="cuda:0")
    dl = U.data.DataLoader(ds, batch_size=100, collate_fn=dataloader2.collate_fn)

    N = net.CoordNetFirstOnly().cuda()
    # N = net.Net().cuda()

    optim = torch.optim.Adam(N.parameters(), lr=1e-4)

    epoch = 1000

    for e in range(epoch):

        print("Epoch", e)
        record_losses = []

        for feats, kps in tqdm(dl):
            embs = N(feats)
            l = loss.loss_fn_batch(embs, kps)

            # All gradient computation
            optim.zero_grad()
            l.backward()
            optim.step()

            record_losses.append(l.item())

        print("Loss:", sum(record_losses)/len(record_losses))
        writer.add_scalar('cdist/loss', sum(record_losses)/len(record_losses), e)
        torch.save(N.state_dict(), f"models/batch_coordconv(100)/{e:02}.pth")


if __name__ == '__main__':
    main()
