from pycocotools.coco import COCO
import numpy as np
import pylab
from PIL import Image 
import PIL 
import os
import openpifpaf

def cache_train_data ():
    dataDir='/home/hestia/Documents/Experiments/Test/Implement/openpifpaf/openpifpaf/data-mscoco'
    dataType='train2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    coco=COCO(annFile)

    filterClasses = ['person']

    catIds = coco.getCatIds(catNms=filterClasses)

    imgIds = coco.getImgIds(catIds=catIds)

    # print("Number of images containing all the  classes:", len(imgIds))

    ims = []
    anns = []
    image_count = 10

    for i in range (image_count):
        #Save Image
        ims.append(coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0])
        im_name = ims[i]['file_name']
        id = ims[i]['id']
        image = Image.open(f'{dataDir}/images/{dataType}/{im_name}') 
        image.save(f"/home/hestia/Documents/Experiments/Test/new/cached_images/coco_train/{id}.jpg")

        #Save Needed Annotation Data
        anns.append(coco.loadAnns(coco.getAnnIds([id])))

    #Run Pifpaf Predict on the train data to obtain the PIF and the HR heatmap
    os.system("python3 -m openpifpaf.predict /home/hestia/Documents/Experiments/Test/new/cached_images/coco_train/*.jpg  --debug-indices cif:0 cifhr:0  --checkpoint=resnet50")


def main ():
   cache_train_data () 

if __name__ == '__main__':
    main()
    
    

    