from typing import OrderedDict
import torch
import models

pif = models.PIF()
pif.conv.load_state_dict(torch.load("wat.pth"))


resnet = models.Resnet()
resnet.load_state_dict(torch.load("resnet_wat.pth"))

print("OK")