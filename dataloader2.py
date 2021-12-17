from os import name
import torch.utils.data as data
from torch.utils.data import DataLoader
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
from net import Net

import pytorch_utils as U


# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

class ImageWithHeatmapDataset(data.Dataset):

    def __init__(self, image_paths, heatmap_dir, resnet_pretrained="weights/resnet50.pth", use_resnet_features=False, resnet_block_idx=-1, heatmap_size=True, device="cuda:0") -> None:

        self.anns = cache_coco_kp.cache_train_data()

        # /home/hestia/Documents/Experiments/Test/embedding_network/cached_images/coco_train/
        self.images = glob(image_paths)
        self.images.sort()

        # /home/hestia/Documents/Experiments/Test/embedding_network/cache_pifpaf_results/field_*/
        self.heatmap_dir = heatmap_dir

        self.device = device

        if use_resnet_features:
            self.model = Resnet(output_block_idx=resnet_block_idx)
            self.model.load_state_dict(torch.load(resnet_pretrained))
            self.model.eval()
            self.model.to(device=device)
        else:
            self.model = (lambda x: x)

        self.heatmap_size = heatmap_size

    def __len__(self):
        return len(self.images)

    def _keypoint_preprocess(self, image_path) -> torch.Tensor:
        # keypoints.size() == total_keypoints_in_image x 6 (#object, #keypoint_type, x1, y1, x2, y2)
        from os.path import basename, splitext
        name = basename(image_path)
        name, ext = splitext(name)
        if name not in self.anns:
            self.anns[name] = torch.zeros((0, 6))
        ordered_anns = self.anns[name]

        return ordered_anns

    @torch.no_grad()
    def __getitem__(self, index):
        image_path = self.images[index]
        features = self._image_preprocess(image_path)
        heatmap = self._heatmap_preprocess(image_path)
        keypoints = self._keypoint_preprocess(image_path)

        return self._join(features, heatmap), keypoints

    def _join(self, features, heatmap):
        if self.heatmap_size:
            C, H, W = heatmap.size()  # Force to heatmap size
            features = F.interpolate(features.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(0)  # set image and heatmap to same size
        else:
            C, H, W = features.size()  # Force to feature size
            heatmap = F.interpolate(heatmap.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(0)  # set image and heatmap to same size

        return torch.cat([features, heatmap], dim=0)

    def _image_preprocess(self, image_path):
        t = TF.to_tensor(Image.open(image_path))
        t = self.model(t.to(device=self.device).unsqueeze(0)).squeeze(0)
        return t  # Features already

    def _heatmap_preprocess(self, image_path):
        filepath = basename(image_path)
        name, ext = splitext(filepath)
        feature_path = join(self.heatmap_dir, f"{name}.pt")

        t = torch.load(feature_path).to(device=self.device)
        ONE, SEVENTEEN, FIVE, H, W = t.size()
        assert ONE == 1
        assert SEVENTEEN == 17
        assert FIVE == 5

        return t.view(-1, H, W)


def cache_feature(out_dir, image_paths, heatmap_dir, resnet_pretrained="weights/resnet50.pth", block_idx=-1, device="cpu"):
    ds = ImageWithHeatmapDataset(image_paths, heatmap_dir, resnet_pretrained, resnet_block_idx=block_idx, device=device)
    makedirs(out_dir, exist_ok=True)

    wrong_channels = 0
    # file1 = open("b&w_channels_overfit.txt","a")
    file1 = open("text_files/b&w_channels_overfit.txt", "a")
    max_x = 0
    max_y = 100
    x_vals = 0
    y_vals = 0

    for i, image_path in enumerate(tqdm(ds.images)):
        filepath = basename(image_path)
        name, ext = splitext(filepath)
        outpath = join(out_dir, f"{name}.pt")

        # if exists(outpath):
        #     continue

        feature, keypoints = ds[i]

        if keypoints.size(0):
            print("Feat & KP", feature.size(), keypoints.size())
            torch.save(feature.cpu(), join(out_dir, f"{name}.features.pt"))
            torch.save(keypoints.cpu(), join(out_dir, f"{name}.keypoints.pt"))
            # x_vals += feature.shape[1]
            # y_vals += feature.shape[2]

            # if (feature.shape[1] > max_x):
            #     max_x = feature.shape[1]

            # if (feature.shape[2] > max_y):
            #     max_y = feature.shape[2]

            if (feature.size()[0] != 88):
                wrong_channels += 1
                file1.write(f"{name}.jpg => Feature Shape: {feature.shape} \n")
        else:
            print("No kp: ", name)

    # average_x = x_vals/(i+1)
    # average_y = y_vals/(i+1)

    # file1.write(f" Max X: {max_x}, Max Y: {max_y}, Average X: {average_x}, Average Y: {average_y} \n")
    print("Number of Images not with 88 channels: ", wrong_channels)
    file1.close()


class LoadCachedFeatureDataset(U.data.Dataset):

    def __init__(self, feature_paths, keypoint_paths, size=(80, 100), device="cpu"):
        features = U.data.glob_files(feature_paths, torch.load)
        keypoints = U.data.glob_files(keypoint_paths, torch.load)

        if isinstance(size, int):
            size = (size, size)
        self.size = size

        self.device = device

        self.ds = U.data.dzip(features, keypoints, zip_transform=self.process_data)

    def process_data(self, feature: torch.Tensor, keypoint: torch.Tensor):
        from torchvision.transforms.functional import resize

        feature = feature.to(device=self.device)
        keypoint = keypoint.to(device=self.device).float()

        c, h, w = feature.size()
        k, SIX = keypoint.size()

        if c == 86:
            dup = feature[0:1, :, :]
            feature = torch.cat([dup, dup, feature], dim=0)

        nh, nw = self.size
        feature = resize(feature, size=(nh, nw))

        rh = nh / h
        rw = nw / w

        keypoint[:, 2:4] *= rw
        keypoint[:, 4:6] *= rh

        return feature, keypoint

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]


def collate_fn(data):
    features = []
    keypoints = []
    for i, row in enumerate(data):
        feature, keypoint = row
        features.append(feature)

        num_keys, SIX = keypoint.size()
        index = torch.ones(num_keys, 1, device=keypoint.device) * i
        keypoint = torch.cat([index, keypoint], dim=1)

        keypoints.append(keypoint)

    features = torch.stack(features, dim=0)
    keypoints = torch.cat(keypoints, dim=0)

    return features, keypoints


def loss_fn(embeddings, keypoints, epoch):
    loss = 0
    assert embeddings.size(0) == 1
    keypoints = keypoints.squeeze(0)

    # ROI Pooling => Embeddings from net: [B, C, H, W], GT KP: [Obj_Instance, kp_type, x1, x2, y1, y2]
    boxes = roi_pool(embeddings, [keypoints[:, -4:].float()], output_size=2, spatial_scale=0.125)
    all_obj_idxs = keypoints[:, 0]
    for obj_idx in torch.unique(all_obj_idxs):  # per kp in obj
        #
        positive = boxes[all_obj_idxs == obj_idx]
        num_boxes, *dims = positive.size()
        if(epoch == 100):
            breakpoint()
        positive = positive.view(1, num_boxes, -1)

        compute_p = torch.cdist(positive, positive)
        loss += compute_p.mean()
        if(epoch == 100):
            breakpoint()

        negative = boxes[all_obj_idxs != obj_idx]

        if negative.size(0) != 0:
            num_boxes, *dims = negative.size()
            if(epoch == 100):
                breakpoint()
            negative = negative.view(1, num_boxes, -1)

            compute_n = torch.cdist(positive, negative)
            loss += 1 / compute_n.mean()
            if(epoch == 100):
                breakpoint()

    return loss


if __name__ == "__main__":
    # cache_feature("cache/coco_train/features", "cache/coco_train/overfit_images/*.jpg", "cache_pifpaf_results/17/", block_idx=3, device="cuda:0")
    # Feature & Keypoint path: cache/coco_train/features
    # Image path: cache/coco_train/overfit_images/*.jpg
    # Cached path: cache_pifpaf_results/17/

    # Train Set
    cache_feature("/mnt/5E18698518695D51/Experiments/caching/features/", "cache/coco_train/images/*.jpg",
                  "/mnt/5E18698518695D51/Experiments/caching/cache_pifpaf_results/17", block_idx=3, device="cuda:0")

    # Val Set
    # cache_feature("/mnt/5E18698518695D51/Experiments/caching_val/features/", "cache/coco_val/images/*.jpg", "/mnt/5E18698518695D51/Experiments/caching_val/cache_pifpaf_results/17", block_idx=3, device="cuda:0")
