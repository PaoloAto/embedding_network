import pytorch_utils as U
import dataloader_clean

from tqdm import tqdm
import loss

import torch

import net

import dataloader2
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def main():

    # ds = dataloader_clean.PifpafDataset(
    #     image_paths="cache/coco_train/images/*.jpg",
    #     keypoint_paths="/mnt/5E18698518695D51/Experiments/caching/res_features/*.pt",
    #     pif_paths="/mnt/5E18698518695D51/Experiments/caching/cache_pifpaf_results/17/*.pt",
    #     # paf_paths="",
    #     device="cuda:0"
    # )

    ds_base = dataloader_clean.PifpafDataset(
        image_paths="cache/coco_train/images/*.jpg",
        keypoint_paths="cache/hdd_data/features/*.keypoints.pt",
        pif_paths="/mnt/5E18698518695D51/Experiments/caching/cache_pifpaf_results/17/*.pt",
        # paf_paths="",
        device="cuda:0"
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
    # N = net.Net(597).cuda()
    # N = net.CoordNet(597).cuda()
    S = net.SameNet(512).cuda()

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
        torch.save(N.state_dict(), f"models/feature_sim_fullCoord/{e:02}.features.pth")
        torch.save(S.state_dict(), f"models/feature_sim_fullCoord/{e:02}.classifier.pth")


if __name__ == '__main__':
    main()
