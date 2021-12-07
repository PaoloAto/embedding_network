import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        #Idea: 4 Layers of: Conv2D -> BN -> Relu (Not yet fully sure on the in & out channels, kernel 3, stride & padding 1)
        self.C1 = nn.Sequential(
                                nn.Conv2d(in_channels , out_channels, 3, 1, 1),
                                nn.BatchNorm2d(num_features),
                                # nn.ReLU(inplace=True)
                                )
        self.C2 = nn.Sequential(
                                nn.Conv2d(in_channels , out_channels, 3, 1, 1),
                                nn.BatchNorm2d(num_features),
                                # nn.ReLU(inplace=True)
                                )
        self.C3 = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                nn.BatchNorm2d(num_features),
                                # nn.ReLU(inplace=True)
                                )
        self.C4 = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                nn.BatchNorm2d(num_features),
                                nn.ReLU(inplace=True)
                                )
    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        return x

def main ():
    net = Net()
    print(net)

if __name__ == '__main__':
    main()