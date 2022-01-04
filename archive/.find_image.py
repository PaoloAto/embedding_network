from glob import glob
from os.path import basename

import torch

from matplotlib import pyplot as plt
from tqdm import tqdm

counts = []

testlist = []


files = glob("cache/hdd_data/features/*.keypoints.pt")

for f in tqdm(files):

    t = torch.load(f)
    objs = torch.unique(t[:, 0])

    counts.append(len(objs))

    if len(objs) == 12:
        f = basename(f).replace(".keypoints.pt", "")
        testlist.append(f + "\n")


plt.hist(counts)
plt.savefig("test.png")


with open("text_files/equals12.txt", "w+") as f:
    f.writelines(testlist)
