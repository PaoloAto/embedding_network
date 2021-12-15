from glob import glob
from shutil import copy
from os.path import join, basename, dirname, abspath
from os import makedirs
import os

def iterate(path, size=100):
    files = glob(path)

    print("Total", len(files))

    for start in range(0, len(files), size):
        end = start + size
        end = min(len(files), end)

        yield files[start:end]


temp_dir = "temp_val"
done_imgs = 0

# Batching images and copying to temp folder
# for group_idx, imagepaths in enumerate(iterate('/home/hestia/Documents/Experiments/Test/embedding_network/cache/coco_val/images/*.jpg')):
#     for imagepath in imagepaths:
#         name = basename(imagepath)
#         outpath = join(temp_dir, str(group_idx), name)
#         makedirs(dirname(outpath), exist_ok=True)

#         copy(imagepath, outpath)



    # cmd = "python3 -m openpifpaf.predict "+ img  +" --debug-indices cif:0   --checkpoint=resnet50"
    # # cmd = "python3 -m openpifpaf.predict "+ img  +" --debug-indices cif:0 cifhr:0  --checkpoint=resnet50  --q"
    # os.system(cmd)


for subdir in glob(join(temp_dir, "*")):
    targets = join(abspath(subdir), "*.jpg")
    cmd = f"python3 -m openpifpaf.predict {targets} --debug-indices cif:0   --checkpoint=resnet50 --q"
    # cmd = "python3 -m openpifpaf.predict "+ img  +" --debug-indices cif:0 --checkpoint=resnet50  --q"
    os.system(cmd)
    done_imgs += 100
    print("Images Done: ", done_imgs)

    # python3 .batch_pifpaf.py