from posixpath import join
import torch.utils.data as data
import numpy as np
import torch

from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class ImageWithHeatmapDataset(data.Dataset):
    
    def __init__(self, image_paths, heatmap_dir, device="cuda:0") -> None:
        from glob import glob

        self.images = glob(image_paths)
        self.images.sort()

        self.heatmap_dir = heatmap_dir

        self.device = device


        from models import Resnet
        self.model = Resnet()
        self.model.load_state_dict(torch.load("resnet50.pth"))
        self.model.eval()
        self.model.to(device=device)

        # Assume images and heatmaps have correct correspondences
        assert len(self.images) == len(self.heatmaps)

    def __len__(self):
        return len(self.images)

    def get_heatmap_paths(self, image_path):
        from os.path import join

        paths = []
        for i in range(17):
            # Data Format: Heatmap dir + /field_{i}/'file_name'.npy
            p = join(self.heatmap_dir, f"field_{i}/")

            # TODO Fix p => fix dirs, input - img paths, output - heatmap 

            paths.append(p)
        return paths


    def __getitem__(self, index):
        image_path = self.images[index]
        heatmap_path = self.heatmaps[index]
        features = self._image_preprocess(image_path)
        heatmap = self._heatmap_preprocess(heatmap_path)
        return self._join(features, heatmap)

    def _join(self, image, heatmap):
        C, H, W = heatmap.size()
        image = F.interpolate(image.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(0) #set image and heatmap to same size
        return torch.cat([image, heatmap], dim=0)

    
    def _image_preprocess(self, image_path):
        x = TF.to_tensor(Image.open(image_path)).unsqueeze(0)
        y = self.model(x)
        return y # Features already

    def _heatmap_preprocess(self, heatmap_paths):
        array = []
        for p in heatmap_paths:
            t = np.load(p)
            array.append(t)
        return torch.tensor(array, device=self.device)
            
def cache_feature(out_dir, image_paths, heatmap_dir, device="cpu"):
    ds = ImageWithHeatmapDataset(image_paths, heatmap_dir, device)
    for i, x in enumerate(ds):
        outpath = join(out_dir, f"{i+i:05}.pt")
        torch.save(x.cpu(), outpath)
