import os
import os.path as p

path = '/home/hestia/Documents/Experiments/Test/Center/CenterNet/data/coco/train2017'
files = os.listdir(path)

for index, file in enumerate(files):
    #print (file)
    res, _ = file.split('.')
    os.rename(p.join(path, file), p.join(path, f"{int(res):012d}.jpg"))
