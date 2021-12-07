from typing import OrderedDict
import torch
import models

pif = models.PIF()
pif.conv.load_state_dict(torch.load("weights/pif.pth"))

resnet = models.Resnet()
resnet.load_state_dict(torch.load("weights/resnet50.pth"))

print(resnet)

input = torch.randn(1, 3, 480, 640)
out = resnet.forward(input)
print(out.shape)