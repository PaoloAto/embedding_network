import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        #Idea: Conv2D -> BN -> Relu (Not yet fully sure on the in & out channels, kernel 3, stride & padding 1)
        self.C1 = nn.Sequential(
                                nn.Conv2d(in_channels , out_channels, 3, 1, 1),
                                nn.BatchNorm2d(num_features),
                                nn.ReLU(inplace=True)
                                )
        self.C2 = nn.Sequential(
                                nn.Conv2d(in_channels , out_channels, 3, 1, 1),
                                nn.BatchNorm2d(num_features),
                                nn.ReLU(inplace=True)
                                )
        self.C3 = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                nn.BatchNorm2d(num_features),
                                nn.ReLU(inplace=True)
                                )

    def forward(self, x):
        c1_out = self.C1(x)
        c2_out = self.C2(c1_out)
        x = self.C3(c2_out)
        return x

def main ():
    net = Net()
    print(net)

if __name__ == '__main__':
    main()