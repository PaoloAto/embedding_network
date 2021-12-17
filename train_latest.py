import pytorch_utils as U
import dataloader_clean

from tqdm import tqdm
import loss

import torch

import net

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def main():

    ds = dataloader_clean.PifpafDataset(
        image_paths="cache/coco_train/images/*.jpg",
        keypoint_paths="cache/hdd_data/features/*.keypoints.pt",
        pif_paths="/mnt/5E18698518695D51/Experiments/caching/cache_pifpaf_results/17/*.pt",
        # paf_paths="",
        device="cuda:0"
    )

    dl = U.data.DataLoader(ds, batch_size=3, collate_fn=dataloader_clean.collate_fn)

    N = net.CoordNetFirstOnly(597).cuda()
    # N = net.Net().cuda()

    optim = torch.optim.Adam(N.parameters(), lr=1e-4)

    batch_process = dataloader_clean.BatchPreprocess(2)

    epoch = 1000

    for e in range(epoch):

        print("Epoch", e)
        record_losses = []

        for feats, kps, pif in tqdm(dl):
            feats = batch_process(feats, pif)
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
