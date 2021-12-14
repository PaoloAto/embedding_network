import dataloader
from net import Net
import loss

import os
from os import name
import torch.utils.data as data
import numpy as np
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

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main ():
    loader = T.Compose([
        torch.load,
        lambda t: t.cuda()
    ])

    feature_ds = dataloader.GlobDataset("cache/coco_train/features/*.features.pt", transform=loader)
    keypoint_ds = dataloader.GlobDataset("cache/coco_train/features/*.keypoints.pt", transform=loader)
    name_ds = dataloader.GlobDataset("cache/coco_train/features/*.features.pt")
    ds = dataloader.ZipDataset(feature_ds, keypoint_ds, name_ds)

    inv_channel = 0

    N = Net().cuda()
    optim = torch.optim.Adam(N.parameters())

    epoch = 101

    for e in range(epoch):
        print("Epoch", e)
        record_losses = []

        # feat = features, kp = keypoints, fn = 'cache/coco_train/features/{img}.features.pt'
        for feat, kp, fn in tqdm(data.DataLoader(ds, batch_size=1)):
            
            if (feat.shape[1] == 88):
                z = N(feat)
                losses = dataloader.loss_fn(z, kp, e)

                # All gradient computation
                optim.zero_grad()
                losses.backward()
                optim.step()

                record_losses.append(losses.item())
            else:
                inv_channel += 1

        # print("Number of Images not with 88 channels in training: ", inv_channel)
        print("Loss:", sum(record_losses)/len(record_losses))
        writer.add_scalar('cdist/loss', sum(record_losses)/len(record_losses), e)
        torch.save(N.state_dict(), f"models/{e:02}.pth")
    
if __name__ == '__main__':
    main()