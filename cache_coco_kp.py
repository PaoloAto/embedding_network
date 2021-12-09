from pycocotools.coco import COCO
import numpy as np
import pylab
from PIL import Image 
import PIL 
import os
import openpifpaf

def kp_to_box (x,y):
    if(x == 0 and y == 0):
        x1 = 0 
        x2 = 0
        y1 = 0
        y2 = 0
    else:
        if x <= 4:
            x1 = 0
            x2 = x + 4
        elif y <= 4:
            y1 = 0
            y2 = y + 4
        else:
            x1 = x - 4
            x2 = x + 4
            y1 = y - 4
            y2 = y + 4

    return x1,x2,y1,y2

def cache_train_data ():
    dataDir='/home/hestia/Documents/Experiments/Test/Implement/openpifpaf/openpifpaf/data-mscoco'
    dataType='train2017'
    annFile='{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)

    coco=COCO(annFile)

    filterClasses = ['person']

    catIds = coco.getCatIds(catNms=filterClasses)

    imgIds = coco.getImgIds(catIds=catIds)

    print("Number of images containing ['Person'] class:", len(imgIds))

    ims = []
    anns = []
    image_count = 2

    for i in range (image_count):
        #Save Image
        ims.append(coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0])
        im_name = ims[i]['file_name']
        id = ims[i]['id']
        image = Image.open(f'{dataDir}/images/{dataType}/{im_name}') 
        image.save(f"/home/hestia/Documents/Experiments/Test/embedding_network/cache/coco_train/images/{id}.jpg")

        #Save Needed Annotation Data
        annotations = coco.loadAnns(coco.getAnnIds([id]))

        for j in range (len(annotations)):
            for k in range (17):
                temp = []
                x1,x2,y1,y2 = kp_to_box(annotations[j]['keypoints'][3*k], annotations[j]['keypoints'][3*k+1])
                # [image_id, object_instance #, key_point #, x1, x2, y1, y2]
                temp.extend((id,j,k,x1,x2,y1,y2))
                anns.append(temp)
        breakpoint()

    breakpoint()

    #Run Pifpaf Predict on the train data to obtain the PIF and the HR heatmap
    # os.system("pytho  openpifpaf.predict /home/hestia/Documents/Experiments/Test/embedding_network/cached_images/coco_train/*.jpg  --debug-indices cif:0 cifhr:0  --checkpoint=resnet50")

    return anns


def main ():
   annotations = cache_train_data () 
   print(annotations)

if __name__ == '__main__':
    main()