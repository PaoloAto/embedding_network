from typing import OrderedDict
import torch
import models

pif = models.PIF()
pif.conv.load_state_dict(torch.load("pif.pth"))


resnet = models.Resnet()
resnet.load_state_dict(torch.load("resnet50.pth"))

print("OK")