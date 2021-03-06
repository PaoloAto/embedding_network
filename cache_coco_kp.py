from pycocotools.coco import COCO
import numpy as np
import pylab
import PIL
from PIL import Image
import os
import glob

import torch
from torch._C import TracingState


def kp_to_box(x, y):
    if(x == 0 and y == 0):
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
    else:
        if x <= 4 and y <= 4:
            x1 = 0
            x2 = x + 4
            y1 = 0
            y2 = y + 4
        elif x <= 4 and y > 4:
            x1 = 0
            x2 = x + 4
            y1 = y - 4
            y2 = y + 4
        elif x > 4 and y <= 4:
            x1 = x - 4
            x2 = x + 4
            y1 = 0
            y2 = y + 4
        else:
            x1 = x - 4
            x2 = x + 4
            y1 = y - 4
            y2 = y + 4

    return x1, x2, y1, y2


def cache_train_data():

    path = "train"

    if (path == "train"):
        dataType = 'train2017'
    else:
        dataType = 'val2017'

    dataDir = '/home/hestia/Documents/Experiments/Test/Implement/openpifpaf/openpifpaf/data-mscoco'
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)

    filterClasses = ['person']

    catIds = coco.getCatIds(catNms=filterClasses)

    imgIds = coco.getImgIds(catIds=catIds)

    print("Number of images containing ['Person'] class:", len(imgIds))

    ims = []
    # kp = []
    image_count = len(imgIds)
    annotated_count = 0
    anns = []

    for i in range(image_count):
        # Save Image
        ims.append(coco.loadImgs(imgIds[i])[0])
        # ims.append(coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]) # Random loading

        im_name = ims[i]['file_name']
        id = ims[i]['id']
        image = Image.open(f'{dataDir}/images/{dataType}/{im_name}')

        image.save(f"/home/hestia/Documents/Experiments/Test/Center/CenterNet/data/coco/train2017/{im_name}")

        # # Save Needed Annotation Data
        # annotations = coco.loadAnns(coco.getAnnIds([id]))
        # kp = []
        # for j in range(len(annotations)):
        #     for k in range(17):
        #         if (annotations[j]['keypoints'][3*k] == 0 and annotations[j]['keypoints'][3*k+1] == 0):
        #             continue
        #         else:
        #             temp = []
        #             x1, x2, y1, y2 = kp_to_box(annotations[j]['keypoints'][3*k], annotations[j]['keypoints'][3*k+1])
        #             temp.extend((f'{id}.jpg', j, k, x1, x2, y1, y2))
        #             kp.append(temp)
        # # Indexing: List[perImage][anns_perImage][image_name, object_instance #, key_point #, x1, x2, y1, y2]
        # # kp[:, :, kp[2] == 1, :, :, :, :]
        # if(len(kp) == 0):
        #     continue
        # else:
        #     anns.append(kp)
        #     # image.save(f"/home/hestia/Documents/Experiments/Test/embedding_network/cache/coco_val/images/{id}.jpg")
        #     annotated_count += 1

    print("Number of images that have annotated keypoints: ", annotated_count)

    db = {}

    for group in anns:
        for a in group:
            name, *data = a
            name = name.replace(".jpg", "")
            if name not in db:
                db[name] = []
            db[name].append(data)

    out = {}
    for name, value in db.items():
        out[name] = torch.tensor(value)

    # Run Pifpaf Predict on the train data to obtain the PIF and the HR heatmap
    # os.system("python3 -m openpifpaf.predict /home/hestia/Documents/Experiments/Test/embedding_network/cache/coco_train/overfit_images/*.jpg  --debug-indices cif:0 cifhr:0  --checkpoint=resnet50  --q")

    return out


def main():
    cache_train_data()


if __name__ == '__main__':
    main()
