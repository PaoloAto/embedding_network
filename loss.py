import torch.utils.data as data
import numpy as np
import torch

import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.ops import roi_pool


def loss_fn(embeddings, keypoints, keypoint_uniqueness_loss=True, eps=1e-7):
    loss = 0
    assert embeddings.size(0) == 1
    keypoints = keypoints.squeeze(0)

    # ROI Pooling => Embeddings from net: [B, C, H, W], GT KP: [Obj_Instance, kp_type, x1, x2, y1, y2]
    boxes = roi_pool(embeddings, [keypoints[:, -4:].float()], output_size=2, spatial_scale=0.125)
    all_obj_idxs = keypoints[:, 0]
    for obj_idx in torch.unique(all_obj_idxs):  # kp_types of per obj_instance in an image

        positive = boxes[all_obj_idxs == obj_idx]  # Getting all annotated kp from the current obj_idx
        num_boxes, *dims = positive.size()

        positive = positive.view(1, num_boxes, -1)

        compute_p = torch.cdist(positive, positive)
        loss += compute_p.mean()

        negative = boxes[all_obj_idxs != obj_idx]  # Getting all kp from the other obj_idx different from the current (can be none if no other annotated kp person instance in image)

        if negative.size(0) != 0:
            num_boxes, *dims = negative.size()

            negative = negative.view(1, num_boxes, -1)

            compute_n = torch.cdist(positive, negative)
            loss += 1 / (compute_n.mean() + eps)

    if keypoint_uniqueness_loss:
        all_kp_idxs = keypoints[:, 1]
        for kp_idx in torch.unique(all_kp_idxs):  # Find all unique keypoints id
            same_kps = boxes[all_kp_idxs == kp_idx]
            if same_kps.size(0) > 0:
                num_boxes, *dims = same_kps.size()
                same_kps = same_kps.view(1, num_boxes, -1)
                compute_same_kps = torch.cdist(same_kps, same_kps)
                loss += 1 / (compute_same_kps.mean() + eps)  # If they are assigned to the same keypoints map them far apart

    return loss
