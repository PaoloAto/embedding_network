import argparse
import pytorch_utils as U

from pytorch_utils.cfg import arg


class config:

    do_cache = arg(action="store_true")
    do_eval = arg(action="store_true")

    images = arg(type=str, default="cache/coco_train/images/*.jpg")
    keypoints = arg(type=str, default="cache/hdd_data/features/*.keypoints.pt")
    pif_cache = arg(type=str, default="/mnt/5E18698518695D51/Experiments/caching/cache_pifpaf_results/17/*.pt")
    dataset_cache = arg(type=str, default="/mnt/5E18698518695D51/Experiments/caching/res_features/{idx}.pt")


if config.do_cache:
    print("Do CACHE")

    exit()

if config.do_eval:
    print("Do Eval")

    exit()
