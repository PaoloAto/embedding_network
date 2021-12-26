import pytorch_utils as U
import dataloader_clean

from tqdm import tqdm
import loss

import torch

import models

import net

import dataloader2
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def main():


    ds = dataloader_clean.ResnetKeypointsDataset(
        image_paths="cache/coco_train/images/*.jpg",
        keypoint_paths="cache/hdd_data/features/*.keypoints.pt",
        device="cuda:0"
    )
    # ds = U.data.dcache_tensor(ds, ".cache/ResnetKeypointsDataset/{idx}.pt")

    dl = U.data.DataLoader(ds, batch_size=3, collate_fn=dataloader2.collate_fn)

    N = net.CoordNetFirstOnly(512).cuda()
    S = net.SameNet(512).cuda()
    # N = net.Net(597).cuda()
    # N = net.CoordNet(597).cuda()

    optim = torch.optim.Adam(list(N.parameters()) + list(S.parameters()), lr=1e-4)

    epoch = 1000

    for e in range(epoch):

        print("Epoch", e)
        record_losses = []

        for feats, kps in tqdm(dl):
            feats = feats.squeeze_(1).float()
            kps = kps.float()
            embs = N(feats)

            l = loss.loss_fn_batch_sim(embs, kps, S)

            # All gradient computation
            optim.zero_grad()
            l.backward()
            optim.step()

            record_losses.append(l.item())

        print("Loss:", sum(record_losses)/len(record_losses))
        writer.add_scalar('cdist/loss', sum(record_losses)/len(record_losses), e)
        torch.save(N.state_dict(), f"models/resnet_input_only/{e:02}.features.pth")
        torch.save(S.state_dict(), f"models/resnet_input_only/{e:02}.classifier.pth")


if __name__ == '__main__':
    main()
