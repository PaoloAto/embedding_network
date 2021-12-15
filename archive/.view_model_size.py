import torch


x = torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/cache/hdd_data/features/100312.features.pt")
y = torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/cache/hdd_data/features/100312.keypoints.pt")

val1 = torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/cache/hdd_data/features/36.features.pt")
val2 = torch.load("/home/hestia/Documents/Experiments/Test/embedding_network/cache/hdd_data/features/36.keypoints.pt")

breakpoint()