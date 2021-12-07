import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 


        channels = [1109, 512, 256, 128]

        layers = []

        in_channels = channels[:-1]
        out_channels = channels[1:]

        relus = [True] * len(in_channels)
        relus[-1] = False

        for in_channel, out_channel, has_relu in zip(in_channels, out_channels, relus):
            layers.append(nn.Conv2d(in_channel , out_channel, 3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channel))
            if has_relu:
                layers.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def main ():
    net = Net()
    print(net)

if __name__ == '__main__':

    x = torch.load("cache/coco_train/features/53800.pt").cuda().unsqueeze(0)
    print(x.size())

    net = Net().cuda()

    y = net(x)
    B, C, H, W = y.size()
    y_flat = y.view(B, C, H*W).permute(0, 2, 1)

    z = torch.cdist(y_flat, y_flat)


    vanilla_similarity_map = None

    def from_same_object(a, b):
        return True

    def get_coords_in_feature_space(k):
        return 0, 0

    def to_flat_idx(x, y, feature_size):
        B, C, H, W = feature_size
        return x * H + y


    keypoints = []
    for a in keypoints:
        for b in keypoints:
            if from_same_object(a, b): #,keypoint_meta_data
                a_x, a_y = get_coords_in_feature_space(a)
                b_x, b_y = get_coords_in_feature_space(b)
                # vanilla_similarity_map[to_flat_idx(a_x, a_y, y.size()), to_flat_idx(b_x, b_y, y.size())] = 0

                # score = similarity(z[a_x, a_y], z[b_x, b_y])

    F.l1_loss(z, vanilla_similarity_map)

    import torchvision
    
    torchvision.utils.save_image(z.cpu(), "test.png")