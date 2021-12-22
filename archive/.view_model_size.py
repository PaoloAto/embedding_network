import torch


x = torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/cache/hdd_data/features/100312.features.pt")
y = torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/cache/hdd_data/features/100312.keypoints.pt")

x2 = torch.load("/mnt/5E18698518695D51/Experiments/caching/res_features/0.pt")

breakpoint()
