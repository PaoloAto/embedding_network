from os import name
from posixpath import join
import torch.utils.data as data
import numpy as np
import torch

from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from models import Resnet
from glob import glob

class ImageWithHeatmapDataset(data.Dataset):
    
    def __init__(self, image_paths, heatmap_dir, resnet_pretrained="weights/resnet50.pth", output_block_idx=-1, device="cuda:0") -> None:
    
        #/home/hestia/Documents/Experiments/Test/embedding_network/cached_images/coco_train/
        self.images = glob(image_paths)
        self.images.sort()

        #/home/hestia/Documents/Experiments/Test/embedding_network/cache_pifpaf_results/field_*/
        self.heatmap_dir = heatmap_dir

        self.device = device

        self.model = Resnet(output_block_idx=output_block_idx)
        self.model.load_state_dict(torch.load(resnet_pretrained))
        self.model.eval()
        self.model.to(device=device)


    def __len__(self):
        return len(self.images)


    @torch.no_grad()
    def __getitem__(self, index):
        image_path = self.images[index]
        features = self._image_preprocess(image_path)
        heatmap = self._heatmap_preprocess(image_path)
        return self._join(features, heatmap)

    def _join(self, features, heatmap):
        print("Features", features.size(), "Heatmap", heatmap.size())

        C, H, W = heatmap.size()
        features = F.interpolate(features.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(0) #set image and heatmap to same size
        return torch.cat([features, heatmap], dim=0)

    
    def _image_preprocess(self, image_path):
        x = TF.to_tensor(Image.open(image_path)).unsqueeze(0)
        y = self.model(x.to(device=self.device)).squeeze(0)
        return y # Features already

    def _heatmap_preprocess(self, image_path):
        from os.path import join, basename, splitext 
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
    from os import makedirs
    from os.path import join, basename, splitext, exists
    from tqdm import tqdm

    ds = ImageWithHeatmapDataset(image_paths, heatmap_dir, resnet_pretrained, output_block_idx=block_idx, device=device)
    makedirs(out_dir, exist_ok=True)

    for i, image_path in enumerate(tqdm(ds.images)):
        filepath = basename(image_path)
        name, ext = splitext(filepath)
        outpath = join(out_dir, f"{name}.pt")

        # if exists(outpath):
        #     continue

        feature = ds[i]
        torch.save(feature.cpu(), outpath)


if __name__ == "__main__":
    cache_feature("cache/coco_train/features", "cache/coco_train/images/*.jpg", "cache/coco_train/pifpaf/17/", block_idx=3, device="cuda:0")