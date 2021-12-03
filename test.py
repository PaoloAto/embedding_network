path = "/home/hestia/Documents/Experiments/Test/Implement/openpifpaf/openpifpaf/cache_wat/17/old_000000081988.jpg.pt"

import torch

x = torch.load(path)
print(x.shape)