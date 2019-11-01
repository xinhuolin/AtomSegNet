# unet.py
# 

from __future__ import division

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    # class UNet(torch.jit.ScriptModule):
    def __init__(self, colordim=1):
        super(UNet, self).__init__()
        self.conv1_1 = nn.Conv2d(colordim, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.upconv4 = nn.Conv2d(256, 128, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.bn4_out = nn.BatchNorm2d(256)

        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv7 = nn.Conv2d(128, 64, 1)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.bn7_2 = nn.BatchNorm2d(128)
        self.bn7_out = nn.BatchNorm2d(128)

        self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv9_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(64)
        self.bn9_2 = nn.BatchNorm2d(64)
        self.conv9_3 = nn.Conv2d(64, colordim, 1)
        self.bn9_3 = nn.BatchNorm2d(colordim)
        self.bn9 = nn.BatchNorm2d(colordim)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        #self._initialize_weights()

        # self.input_layer = nn.Sequential(self.conv1_1, self.bn1_1, nn.ReLU(),self.conv1_2, self.bn1_2,  nn.ReLU())
        #
        # self.down1 = nn.Sequential(self.conv2_1, self.bn2_1, nn.ReLU(), self.conv2_2, self.bn2_2, nn.ReLU())
        #
        # self.down2 = nn.Sequential(self.conv4_1, self.bn4_1, nn.ReLU(), self.conv4_2, self.bn4_2, nn.ReLU())
        #
        # self.up1 = nn.Sequential(self.upconv4, self.bn4)
        #
        # self.up2 = nn.Sequential(self.bn4_out, self.conv7_1,self.bn7_1 , nn.ReLU(), self.conv7_2, self.bn7_2, nn.ReLU())
        #
        # self.output = nn.Sequential(self.conv4_1, self.bn4_1, nn.ReLU(), self.conv4_2, self.bn4_2, nn.ReLU())


    def forward(self, x1):
        x1 = F.relu(self.bn1_2(self.conv1_2(F.relu(self.bn1_1(self.conv1_1(x1))))))
        x2 = F.relu(self.bn2_2(self.conv2_2(F.relu(self.bn2_1(self.conv2_1(self.maxpool(x1)))))))
        xup = F.relu(self.bn4_2(self.conv4_2(F.relu(self.bn4_1(self.conv4_1(self.maxpool(x2)))))))
        xup = self.bn4(self.upconv4(self.upsample(xup)))
        xup = self.bn4_out(torch.cat((x2, xup), 1))
        xup = F.relu(self.bn7_2(self.conv7_2(F.relu(self.bn7_1(self.conv7_1(xup))))))

        xup = self.bn7(self.upconv7(self.upsample(xup)))
        xup = self.bn7_out(torch.cat((x1, xup), 1))

        xup = F.relu(self.conv9_3(F.relu(self.bn9_2(self.conv9_2(F.relu(self.bn9_1(self.conv9_1(xup))))))))

        return torch.sigmoid(self.bn9(xup))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    unet = UNet()
    weights = torch.load("../model_weights/denoise.pth")
    for m in unet.named_parameters():
        print(m)