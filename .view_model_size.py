import torch


x = torch.load("cache/coco_train/features/53800.pt")
print(x.size())
print("Ok")