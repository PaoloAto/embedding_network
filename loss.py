import torch.utils.data as data
import numpy as np
import torch

import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.ops import roi_pool


def loss_fn(embeddings, keypoints, keypoint_uniqueness_loss=True):  # , eps=1e-7
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
            loss += 1 / (compute_n.mean())  # + eps

    if keypoint_uniqueness_loss:
        all_kp_idxs = keypoints[:, 1]
        for kp_idx in torch.unique(all_kp_idxs):  # Find all unique keypoints id
            same_kps = boxes[all_kp_idxs == kp_idx]
            if same_kps.size(0) > 0:
                num_boxes, *dims = same_kps.size()
                same_kps = same_kps.view(1, num_boxes, -1)
                compute_same_kps = torch.cdist(same_kps, same_kps)
                loss += 1 / (compute_same_kps.mean())  # If they are assigned to the same keypoints map them far apart     #  + eps

    return loss


def loss_fn_batch(embeddings, keypoints, keypoint_uniqueness_loss=True):  # , eps=1e-7
    loss = 0

    B, C, H, W = embeddings.size()
    K, SEVEN = keypoints.size()
    assert SEVEN == 7

    # ROI Pooling => Embeddings from net: [B, C, H, W], GT KP: [Obj_Instance, kp_type, x1, x2, y1, y2]
    boxes_all = roi_pool(embeddings, keypoints[:, [0, 3, 4, 5, 6]], output_size=2, spatial_scale=0.125)
    num_boxes_all, *dims = boxes_all.size()
    boxes_all = boxes_all.view(num_boxes_all, -1)

    all_batch_idxs = keypoints[:, 0]

    for batch_idx in torch.unique(all_batch_idxs):

        keypoint_subset = keypoints[all_batch_idxs == batch_idx, 1:]
        boxes = boxes_all[all_batch_idxs == batch_idx, :]

        all_obj_idxs = keypoint_subset[:, 0]
        for obj_idx in torch.unique(all_obj_idxs):  # kp_types of per obj_instance in an image

            positive = boxes[all_obj_idxs == obj_idx, :]  # Getting all annotated kp from the current obj_idx
            num_boxes, *dims = positive.size()

            positive = positive.view(1, num_boxes, -1)

            compute_p = torch.cdist(positive, positive)
            loss += compute_p.mean()

            negative = boxes[all_obj_idxs != obj_idx]  # Getting all kp from the other obj_idx different from the current (can be none if no other annotated kp person instance in image)

            if negative.size(0) != 0:
                num_boxes, *dims = negative.size()

                negative = negative.view(1, num_boxes, -1)

                compute_n = torch.cdist(positive, negative)

                score = compute_n.mean()
                if score > 0:
                    loss += 1 / score  # + eps

    # cross batch comparisson
    if keypoint_uniqueness_loss:
        all_kp_idxs = keypoints[:, 2]
        for kp_idx in torch.unique(all_kp_idxs):  # Find all unique keypoints id
            same_kps = boxes_all[all_kp_idxs == kp_idx, :]
            if same_kps.size(0) > 0:
                num_boxes, *dims = same_kps.size()
                same_kps = same_kps.view(1, num_boxes, -1)
                compute_same_kps = torch.cdist(same_kps, same_kps)

                score = compute_same_kps.mean()
                if score > 0:
                    loss += 1 / score  # If they are assigned to the same keypoints map them far apart     #  + eps

    return loss

def pairwise_random(positive: torch.Tensor, negative: torch.Tensor, count=100):
    positive_idxs = torch.randint(positive.size(0), size=(count,), device=positive.device)
    negative_idxs = torch.randint(negative.size(0), size=(count,), device=negative.device)
    return positive[positive_idxs], negative[negative_idxs]


def loss_fn_batch_sim(embeddings, keypoints, sim_fn=None, keypoint_uniqueness_loss=True, pairwise_random_count=100):  # , eps=1e-7
    loss = 0

    B, C, H, W = embeddings.size()
    K, SEVEN = keypoints.size()
    assert SEVEN == 7

    # ROI Pooling => Embeddings from net: [B, C, H, W], GT KP: [Obj_Instance, kp_type, x1, x2, y1, y2]
    boxes_all = roi_pool(embeddings, keypoints[:, [0, 3, 4, 5, 6]], output_size=2, spatial_scale=0.125)
    num_boxes_all, *dims = boxes_all.size()
    boxes_all = boxes_all.view(num_boxes_all, -1)

    all_batch_idxs = keypoints[:, 0]

    loss_metric = 0

    for batch_idx in torch.unique(all_batch_idxs):

        keypoint_subset = keypoints[all_batch_idxs == batch_idx, 1:]
        boxes = boxes_all[all_batch_idxs == batch_idx, :]

        all_obj_idxs = keypoint_subset[:, 0]
        for obj_idx in torch.unique(all_obj_idxs):  # kp_types of per obj_instance in an image

            positive = boxes[all_obj_idxs == obj_idx, :]  # Getting all annotated kp from the current obj_idx
            num_boxes, *dims = positive.size()

            positive = positive.view(1, num_boxes, -1)

            compute_p = torch.cdist(positive, positive)
            loss_metric += compute_p.mean()

            negative = boxes[all_obj_idxs != obj_idx, :]  # Getting all kp from the other obj_idx different from the current (can be none if no other annotated kp person instance in image)

            if negative.size(0) != 0:
                num_boxes, *dims = negative.size()

                negative = negative.view(1, num_boxes, -1)

                compute_n = torch.cdist(positive, negative)

                score = compute_n.mean()
                if score > 0:
                    loss_metric += 1 / score  # + eps

    loss += loss_metric / B

    if sim_fn is not None:
        loss_sim = 0
        loss_sim_denom = 0

        for batch_idx in torch.unique(all_batch_idxs):

            keypoint_subset = keypoints[all_batch_idxs == batch_idx, 1:]
            boxes = boxes_all[all_batch_idxs == batch_idx, :]

            all_obj_idxs = keypoint_subset[:, 0]
            for obj_idx in torch.unique(all_obj_idxs):  # kp_types of per obj_instance in an image

                positive = boxes[all_obj_idxs == obj_idx, :]  # Getting all annotated kp from the current obj_idx
                if positive.size(0) > 0:

                    a, b = pairwise_random(positive, positive, pairwise_random_count)
                    pred = sim_fn(a,b)
                    labels = torch.ones((pairwise_random_count, 1), device=pred.device)

                    loss_sim += F.binary_cross_entropy_with_logits(pred, labels)
                    loss_sim_denom += 1

                    negative = boxes[all_obj_idxs != obj_idx, :]
                    if negative.size(0) > 0:

                        a, b = pairwise_random(positive, negative, pairwise_random_count)
                        pred = sim_fn(a,b)
                        labels = torch.zeros((pairwise_random_count, 1), device=pred.device)

                        loss_sim += F.binary_cross_entropy_with_logits(pred, labels)
                        loss_sim_denom += 1


        loss += loss_sim / B / loss_sim_denom

    # cross batch comparisson
    if keypoint_uniqueness_loss:
        loss_kp_unique = 0

        all_kp_idxs = keypoints[:, 2]
        for kp_idx in torch.unique(all_kp_idxs):  # Find all unique keypoints id
            same_kps = boxes_all[all_kp_idxs == kp_idx, :]
            if same_kps.size(0) > 0:
                num_boxes, *dims = same_kps.size()
                same_kps = same_kps.view(1, num_boxes, -1)
                compute_same_kps = torch.cdist(same_kps, same_kps)

                score = compute_same_kps.mean()
                if score > 0:
                    loss_kp_unique += 1 / score  # If they are assigned to the same keypoints map them far apart     #  + eps
        
        loss += loss_kp_unique / B

    return loss
