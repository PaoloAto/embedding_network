import torch
import os.path as P

import numpy as np
from sklearn.manifold import TSNE
import dataloader
import net

from torchvision.ops import roi_pool
import torchvision

from matplotlib import pyplot as plt

with open("text_files/equals12.txt") as f:
    files = f.readlines()


def transform(p: str, device="cuda:0"):
    p = p.strip()
    features = f"cache/hdd_data/features/{p}.features.pt"
    keypoints = f"cache/hdd_data/features/{p}.keypoints.pt"
    return torch.load(features).to(device=device), torch.load(keypoints).to(device=device)


ds = dataloader.ValueDataset(files[40:], transform=transform)


def doview(emb: torch.Tensor, kp: torch.Tensor, sim_out: str, view_out: str):

    boxes = roi_pool(emb, [kp[:, -4:].float()], output_size=2, spatial_scale=0.125)
    num_boxes, *dims = boxes.size()
    boxes = boxes.view(num_boxes, -1)

    objid = kp[:, 0]
    segments = []

    bw = []

    clr = 0.0

    for id in torch.unique(objid):
        segment = boxes[objid == id, :]
        segments.append(segment)
        bw.append([clr] * len(segment))
        clr = np.abs(clr - 1)

    segments = torch.cat(segments, dim=0)
    print(segments.shape)

    similarity = torch.cdist(segments, segments).squeeze(0)
    line = torch.tensor([item for sublist in bw for item in sublist], device=similarity.device)

    pics = torch.cat([line.unsqueeze(0), similarity], dim=0)
    torchvision.utils.save_image(pics, sim_out)

    # TSNE Visualization
    objid = objid.cpu().numpy()
    boxes = boxes.cpu().numpy()

    rep2d = TSNE(n_components=2, learning_rate='auto', n_iter=1000, init='random').fit_transform(boxes)

    for id in np.unique(objid):
        xy = rep2d[objid == id]
        plt.scatter(xy[:, 0], xy[:, 1])

    plt.savefig(view_out)

    print(rep2d.shape)


with torch.no_grad():
    N = net.Net().cuda()
    N.load_state_dict(torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/models/run(130imgs_1bs_500ep)/499.pth"))

    NO = net.Net().cuda()
    NO.load_state_dict(torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/models/full(normConv)/12.pth"))

    CN = net.CoordNet().cuda()
    CN.load_state_dict(torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/models/coordconv_kpu/00.pth"))

    NOCN = net.CoordNet().cuda()
    NOCN.load_state_dict(torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/models/coordconv(no_kpu)/00.pth"))

    CNKP = net.CoordNetFirstOnly().cuda()
    CNKP.load_state_dict(torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/models/coordconv_firstLayer/04.pth"))

    BCN8 = net.CoordNetFirstOnly().cuda()
    BCN8.load_state_dict(torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/models/batch_coordconv(8)/00.pth"))

    BCN8NOKP = net.CoordNetFirstOnly().cuda()
    BCN8NOKP.load_state_dict(torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/models/batch_coordconv(8)/00.pth"))

    BCN100 = net.CoordNetFirstOnly().cuda()
    BCN100.load_state_dict(torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/models/batch_coordconv(8)/00.pth"))

    for ft, kp in dataloader.DataLoader(ds, batch_size=1, shuffle=False):
        emb: torch.Tensor = N(ft)
        kp: torch.Tensor = kp.squeeze(0)
        doview(emb, kp, "visualize/out_base_sim.png", "visualize/out_base_view.png")

        emb: torch.Tensor = NO(ft)
        kp: torch.Tensor = kp.squeeze(0)
        doview(emb, kp, "visualize/out_base_overfit_sim.png", "visualize/out_base_overfit_view.png")

        emb: torch.Tensor = CN(ft)
        kp: torch.Tensor = kp.squeeze(0)
        doview(emb, kp, "visualize/out_coord_sim.png", "visualize/out_coord_view.png")

        emb: torch.Tensor = NOCN(ft)
        kp: torch.Tensor = kp.squeeze(0)
        doview(emb, kp, "visualize/out_coord_nokp_sim.png", "visualize/out_coord_nokp_view.png")

        emb: torch.Tensor = CNKP(ft)
        kp: torch.Tensor = kp.squeeze(0)
        doview(emb, kp, "visualize/out_coord_kp_1stLay_sim.png", "visualize/out_coord_kp_1stLay_view.png")

        emb: torch.Tensor = BCN8(ft)
        kp: torch.Tensor = kp.squeeze(0)
        doview(emb, kp, "visualize/out_coord_kp_batch8_sim.png", "visualize/out_coord_kp_batch8_view.png")

        emb: torch.Tensor = BCN8NOKP(ft)
        kp: torch.Tensor = kp.squeeze(0)
        doview(emb, kp, "visualize/out_coord_kp_batch8_NOKP_sim.png", "visualize/out_coord_kp_batch8_NOKP_view.png")

        emb: torch.Tensor = BCN100(ft)
        kp: torch.Tensor = kp.squeeze(0)
        doview(emb, kp, "visualize/out_coord_kp_batch100_sim.png", "visualize/out_coord_kp_batch100_view.png")

        exit()
