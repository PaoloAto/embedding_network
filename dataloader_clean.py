import torch
from torch._C import device
import pytorch_utils as U

import torchvision.transforms.functional as TF


class ResnetKeypointsDataset(U.data.Dataset):
    def __init__(
        self,
        image_paths,
        keypoint_paths,
        size=(640, 800),
        resnet_block_output=3,
        resnet_preprocessing=True,
        device="cuda:0",
    ):

        self.images = U.data.glob_files(image_paths, transform=self.load_image_features)
        self.keypoints = U.data.glob_files(
            keypoint_paths, transform=self.load_cached_keypoints
        )
        self.ds = U.data.dzip(
            self.images, self.keypoints, zip_transform=self.combine_all
        )

        self.resnet = None

        self.size = size
        self.device = device
        self.resnet_block_output = resnet_block_output
        self.resnet_preprocessing = resnet_preprocessing

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]

    def combine_all(self, img: torch.Tensor, kps: torch.Tensor):
        img = TF.resize(img, size=self.size)

        img_nh, img_nw = self.size
        img_h, img_w = img.shape[-2:]

        rh = img_nh / img_h
        rw = img_nw / img_w

        kps[:, 2:4] *= rw
        kps[:, 4:6] *= rh

        img = img.to(device=self.device)
        kps = kps.to(device=self.device)

        return img, kps

    def load_cached_keypoints(self, path):
        return torch.load(path).float()

    def load_image_features(self, path):
        from PIL import Image

        img = Image.open(path)
        img = TF.to_tensor(img).to(device=self.device)
        if img.size(0) == 1:
            return img.repeat(3, 1, 1)

        if self.resnet is None:
            from models import Resnet

            self.resnet = Resnet(
                output_block_idx=self.resnet_block_output,
                preprocessing=self.resnet_preprocessing,
            ).to(device=self.device)
        return self.resnet(img.unsqueeze(0)).squeeze(0)


class PifpafDataset(U.data.Dataset):
    def __init__(
        self,
        image_paths,
        keypoint_paths,
        pif_paths=None,
        paf_paths=None,
        pif_size=(80, 100),
        device="cuda:0",
    ):

        self.images = U.data.glob_files(image_paths, transform=self.load_image_features)
        self.keypoints = U.data.glob_files(keypoint_paths, transform=self.load_cached_keypoints)
        self.pifs = U.data.glob_files(pif_paths, transform=self.load_cached_pifpaf)

        if paf_paths is not None:
            self.pafs = U.data.glob_files(paf_paths, transform=self.load_cached_pifpaf)
            self.ds = U.data.dzip(
                self.images,
                self.keypoints,
                self.pifs,
                self.pafs,
                zip_transform=self.combine_all,
            )
        else:
            self.ds = U.data.dzip(self.images, self.keypoints, self.pifs, zip_transform=self.combine_all)

        if isinstance(pif_size, int):
            pif_size = (pif_size, pif_size)
        self.pif_size = pif_size

        self.device = device

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]

    def combine_all(self, img: torch.Tensor, kps: torch.Tensor, pif: torch.Tensor, *args):
        paf: torch.Tensor = args[0] if len(args) > 0 else None

        pif_h, pif_w = pif.shape[-2:]
        pif_nh, pif_nw = self.pif_size

        img = TF.resize(img, size=(pif_nh * 8, pif_nw * 8))
        pif = TF.resize(pif, size=(pif_nh, pif_nw))

        rh = pif_nh / pif_h
        rw = pif_nw / pif_w

        kps[:, 2:4] *= rw
        kps[:, 4:6] *= rh

        img = img.to(device=self.device)
        kps = kps.to(device=self.device)
        pif = pif.to(device=self.device)

        if paf is None:
            return img, kps, pif

        paf = TF.resize(paf, size=(pif_nh, pif_nw))
        paf = paf.to(device=self.device)
        return img, kps, pif, paf

    def load_cached_keypoints(self, path):
        return torch.load(path).float()

    def load_cached_pifpaf(self, path):
        t: torch.Tensor = torch.load(path)
        H, W = t.shape[-2:]
        return t.view(-1, H, W)

    def load_image_features(self, path):
        from PIL import Image

        img = Image.open(path)
        img = TF.to_tensor(img).to(device=self.device)
        if img.size(0) == 1:
            return img.repeat(3, 1, 1)
        return img


class BatchPreprocess:
    def __init__(self, resnet_block_output=3, preprocessing=True, device="cuda:0"):
        self.resnet_block_output = resnet_block_output
        self.preprocessing = preprocessing
        self.device = device

        self.feature_extractor = None

    def __call__(self, img: torch.Tensor, pif: torch.Tensor, paf: torch.Tensor = None):
        if self.feature_extractor is None:
            from models import Resnet

            self.feature_extractor = Resnet(
                output_block_idx=self.resnet_block_output,
                preprocessing=self.preprocessing,
            ).to(device=self.device)

        out = []

        img = self.feature_extractor(img)

        from torch.nn.functional import interpolate

        pif_h, pif_w = pif.shape[-2:]

        img = interpolate(img, size=(pif_h, pif_w), mode="bilinear")
        out.append(img)
        out.append(pif)

        if paf is not None:
            paf = interpolate(paf, size=(pif_h, pif_w), mode="bilinear")
            out.append(paf)

        return torch.cat(out, dim=1)


def collate_fn(data):
    images = []
    keypoints = []
    pifs = []
    pafs = []

    for i, row in enumerate(data):
        image, keypoint, pif, *args = row
        paf = args[0] if len(args) > 0 else None

        images.append(image)

        num_keys, SIX = keypoint.size()
        index = torch.ones(num_keys, 1, device=keypoint.device) * i
        keypoint = torch.cat([index, keypoint], dim=1)

        keypoints.append(keypoint)

        pifs.append(pif)
        if paf is not None:
            pafs.append(pafs)

    images = torch.stack(images, dim=0)
    keypoints = torch.cat(keypoints, dim=0)
    pifs = torch.stack(pifs, dim=0)

    if len(pafs) == 0:
        return images, keypoints, pifs

    pafs = torch.stack(pafs, dim=0)
    return images, keypoints, pifs, pafs
