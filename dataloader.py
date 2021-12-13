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

class ImageWithHeatmapDataset(data.Dataset):
    
    def __init__(self, image_paths, heatmap_dir, resnet_pretrained="weights/resnet50.pth", use_resnet_features=False, resnet_block_idx=-1, heatmap_size=True, device="cuda:0") -> None:

        self.anns = cache_coco_kp.cache_train_data()
    
        #/home/hestia/Documents/Experiments/Test/embedding_network/cached_images/coco_train/
        self.images = glob(image_paths)
        self.images.sort()

        #/home/hestia/Documents/Experiments/Test/embedding_network/cache_pifpaf_results/field_*/
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
            C, H, W = heatmap.size() # Force to heatmap size
            features = F.interpolate(features.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(0) #set image and heatmap to same size
        else:
            C, H, W = features.size() # Force to feature size
            heatmap = F.interpolate(heatmap.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(0) #set image and heatmap to same size

        return torch.cat([features, heatmap], dim=0)
    
    def _image_preprocess(self, image_path):
        t = TF.to_tensor(Image.open(image_path))
        t = self.model(t.to(device=self.device).unsqueeze(0)).squeeze(0)
        return t # Features already

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
    from os.path import join
    ds = ImageWithHeatmapDataset(image_paths, heatmap_dir, resnet_pretrained, resnet_block_idx=block_idx, device=device)
    makedirs(out_dir, exist_ok=True)

    for i, image_path in enumerate(tqdm(ds.images)):
        filepath = basename(image_path)
        name, ext = splitext(filepath)
        outpath = join(out_dir, f"{name}.pt")

        # if exists(outpath):
        #     continue

        feature, keypoints = ds[i]
        if keypoints.size(0):
            print("KP", feature.size(), keypoints.size())
            torch.save(feature.cpu(), join(out_dir, f"{name}.features.pt"))
            torch.save(keypoints.cpu(), join(out_dir, f"{name}.keypoints.pt"))
        else:
            print("No kp: ", name)


class GlobDataset(data.Dataset):

    def __init__(self, *paths, transform=None):
        from glob import glob

        values = []
        for p in paths:
            items = glob(p)
            for i in items:
                values.append(i)
        
        values.sort()

        self.values = values
        self.transform = transform or (lambda x: x)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.transform(self.values[index])

class ZipDataset(data.Dataset):

    def __init__(self, *ds):
        self.ds = ds

    def __len__(self):
        return min(map(len, self.ds))

    def getitem0(self, index):
        for d in self.ds:
            yield d[index]

    def __getitem__(self, index):
        return tuple(self.getitem0(index))


from torchvision.ops import roi_pool

def loss_fn(embeddings, keypoints):
    loss = 0

    assert embeddings.size(0) == 1
    keypoints = keypoints.squeeze(0)

    # ROI Pooling
    boxes = roi_pool(embeddings, [keypoints[:,-4:].float()], output_size=2, spatial_scale=0.125)# TODO find scale
    all_obj_idxs = keypoints[:,0]
    for obj_idx in torch.unique(all_obj_idxs):

        positive = boxes[all_obj_idxs == obj_idx]
        num_boxes, *dims = positive.size()
        positive = positive.view(1, num_boxes, -1)
        loss += torch.cdist(positive, positive).mean()


        negative = boxes[all_obj_idxs != obj_idx]
        if negative.size(0) != 0:
            num_boxes, *dims = negative.size()
            negative = negative.view(1, num_boxes, -1)
            loss += 1 / torch.cdist(positive, negative).mean()


    return loss



if __name__ == "__main__":
    from torchvision import transforms as T
    # cache_feature("cache/coco_train/features", "cache/coco_train/images/*.jpg", "cache_pifpaf_results/17/", block_idx=3, device="cuda:0")
    # print("-"*120)

    loader = T.Compose([
        torch.load,
        lambda t: t.cuda()
    ])
    feature_ds = GlobDataset("cache/coco_train/features/*.features.pt", transform=loader)
    keypoint_ds = GlobDataset("cache/coco_train/features/*.keypoints.pt", transform=loader)
    name_ds = GlobDataset("cache/coco_train/features/*.features.pt")
    ds = ZipDataset(feature_ds, keypoint_ds, name_ds)

    from net import Net

    
    N = Net().cuda()
    optim = torch.optim.Adam(N.parameters())

    from tqdm import tqdm



    for e in range(100):
        print("Epoch", e)
        record_losses = []

        for x, y, n in tqdm(data.DataLoader(ds, batch_size=1)):

            z = N(x)
            l = loss_fn(z, y)

            # All gradient computation
            optim.zero_grad()
            l.backward()
            optim.step()

            record_losses.append(l.item())

        print("Loss:", sum(record_losses)/len(record_losses))
        torch.save(N.state_dict(), f"models/{e:02}.pth")

