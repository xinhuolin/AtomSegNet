# coding=utf-8
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, InstanceNorm2d, LeakyReLU, Linear, Sigmoid, Upsample, \
    Tanh, Dropout, ReLU, MaxPool2d, AvgPool2d
import torch
from mypackage.model.shared.basic import getNormLayer, getActiveLayer


class NLayer_D(Module):
    def __init__(self, input_nc=1, output_nc=1, depth=64, use_sigmoid=True, use_liner=True, norm_type="batch",
                 active_type="ReLU"):
        super(NLayer_D, self).__init__()
        self.norm = getNormLayer(norm_type)
        self.active = getActiveLayer(active_type)
        self.use_sigmoid = use_sigmoid
        self.use_liner = use_liner
        # 256 x 256
        self.layer1 = Sequential(Conv2d(input_nc + output_nc, depth, kernel_size=7, stride=1, padding=3),
                                 LeakyReLU(0.2))
        # 128 x 128
        self.layer2 = Sequential(Conv2d(depth, depth * 2, kernel_size=3, stride=1, padding=1),
                                 self.norm(depth * 2, affine=True),
                                 LeakyReLU(0.2), MaxPool2d(2, 2))
        # 64 x 64
        self.layer3 = Sequential(Conv2d(depth * 2, depth * 4, kernel_size=3, stride=1, padding=1),
                                 self.norm(depth * 4, affine=True),
                                 LeakyReLU(0.2), MaxPool2d(2, 2))
        # 32 x 32
        self.layer4 = Sequential(Conv2d(depth * 4, depth * 8, kernel_size=3, stride=1, padding=1),
                                 self.norm(depth * 8, affine=True),
                                 LeakyReLU(0.2), AvgPool2d(2, 2))
        # 16 x 16
        self.layer5 = Sequential(Conv2d(depth * 8, output_nc, kernel_size=7, stride=1, padding=3))
        # 16 x 16 ,1
        self.liner = Linear(256, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x, g):
        out = torch.cat((x, g), 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        if self.use_liner:
            out = out.view(out.size(0), -1)

        if self.use_sigmoid:
            out = self.sigmoid(out)

        return out


class Unet_G(Module):
    def __init__(self, input_nc=1, output_nc=1, depth=32, norm_type="batch", active_type="LeakyReLU"):
        super(Unet_G, self).__init__()
        self.norm = getNormLayer(norm_type)
        self.active = getActiveLayer(active_type)
        self.depth = depth
        self.d0 = Sequential(Conv2d(input_nc, depth, 5, 1, 2), getActiveLayer(active_type))  # 256, 256, 16
        self.d1 = Sequential(Conv2d(depth * 1, depth * 2, 3, 1, 1), self.norm(depth * 2, affine=True),
                             getActiveLayer(active_type),
                             MaxPool2d(2, 2))  # 128, 128, 32
        self.d2 = Sequential(Conv2d(depth * 2, depth * 2, 3, 1, 1), self.norm(depth * 2, affine=True),
                             getActiveLayer(active_type),
                             MaxPool2d(2, 2))  # 64, 64, 32
        self.d3 = Sequential(Conv2d(depth * 2, depth * 4, 3, 1,1), self.norm(depth * 4, affine=True),
                             getActiveLayer(active_type),
                             MaxPool2d(2, 2))  # 32, 32, 64
        self.d4 = Sequential(Conv2d(depth * 4, depth * 4, 3, 1,1), self.norm(depth * 4, affine=True),
                             getActiveLayer(active_type),
                             MaxPool2d(2, 2))  # 16, 16, 64
        self.d5 = Sequential(Conv2d(depth * 4, depth * 8, 3, 1,1), self.norm(depth * 8, affine=True),
                             getActiveLayer(active_type),
                             MaxPool2d(2, 2))  # 8, 8, 128
        self.d6 = Sequential(Conv2d(depth * 8, depth * 8, 3, 1, 1), self.norm(depth * 8, affine=True),
                             getActiveLayer(active_type),
                             AvgPool2d(2, 2))  # 4, 4, 128
        self.d7 = Sequential(Conv2d(depth * 8, depth * 16, 3, 1, 1), self.norm(depth * 16, affine=True),
                             getActiveLayer(active_type),
                             AvgPool2d(2, 2))  # 2, 2, 256
        self.d8 = Sequential(Conv2d(depth * 16, depth * 16, 3, 1, 1), self.norm(depth * 16, affine=True),
                             getActiveLayer(active_type))  # 2, 2, 256

        self.u0 = Sequential(Upsample(scale_factor=2),
                             Conv2d(depth * 16, depth * 16, 3, 1, 1), self.norm(depth * 16, affine=True),
                             getActiveLayer(active_type))
        self.u1 = Sequential(getActiveLayer(active_type),
                Upsample(scale_factor=2),
                Conv2d(depth * 8, depth * 8, 3, 1, 2, dilation=2),
                self.norm(depth * 8, affine=True),
                getActiveLayer(active_type))
        self.u2 = Sequential(getActiveLayer(active_type),
                Upsample(scale_factor=2),
                             Conv2d(depth * 4, depth * 4, 3, 1, 2, dilation=2), self.norm(depth * 4, affine=True),
                             getActiveLayer(active_type))
        self.u3 = Sequential(getActiveLayer(active_type),
                Upsample(scale_factor=2),
                             Conv2d(depth * 4, depth * 4, 3, 1, 2, dilation=2), self.norm(depth * 4, affine=True),
                             getActiveLayer(active_type))
        self.u4 = Sequential(getActiveLayer(active_type),
                Upsample(scale_factor=2),
                             Conv2d(depth * 2, depth * 2, 3, 1, 2, dilation=2), self.norm(depth * 2, affine=True),
                             getActiveLayer(active_type))
        self.u5 = Sequential(getActiveLayer(active_type),
                Upsample(scale_factor=2),
                             Conv2d(depth * 2, depth * 2, 3, 1, 2, dilation=2), self.norm(depth * 2, affine=True),
                             getActiveLayer(active_type))
        self.u6 = Sequential(getActiveLayer(active_type),
                Upsample(scale_factor=2),
                             Conv2d(depth * 1, depth * 1, 5, 1, 2), self.norm(depth * 1, affine=True),
                             getActiveLayer(active_type))

        self.u7 = Sequential(Conv2d(depth * 1, output_nc, 7, 1, 3), Tanh())

        self.leaky_relu = self.active
        self.conv_32_8 = Conv2d(depth * 16 * 2, depth * 8, 3, 1, 1)
        self.conv_16_8 = Conv2d(depth * 8 * 2, depth * 8, 3, 1, 1)
        self.conv_16_4 = Conv2d(depth * 8 * 2, depth * 4, 3, 1, 1)
        self.conv_8_4 = Conv2d(depth * 4 * 2, depth * 4, 3, 1, 1)
        self.conv_8_2 = Conv2d(depth * 4 * 2, depth * 2, 3, 1, 1)
        self.conv_4_2 = Conv2d(depth * 2 * 2, depth * 2, 3, 1, 1)
        self.conv_4_1 = Conv2d(depth * 2 * 2, depth * 1, 3, 1, 1)
        self.conv_2_1 = Conv2d(depth * 1 * 2, depth * 1, 3, 1, 1)

    def forward(self, input):
        d0 = self.d0(input)  # 1,256,256
        d1 = self.d1(d0)  # 2,128,128
        d2 = self.d2(d1)  # 2,64,64
        d3 = self.d3(d2)  # 4,32,32
        d4 = self.d4(d3)  # 4,16,16
        d5 = self.d5(d4)  # 8,8,8
        d6 = self.d6(d5)  # 8,4,4
        d7 = self.d7(d6)  # 16,2,2
        d8 = self.d8(d7)  # 16,1,1

        # depth * 16, depth * 16                  depth * 8
        # d8 = self.u0(d8)
        u0 = torch.cat((d8, d7), 1)  # 32, 2, 2
        u0 = self.leaky_relu(self.conv_32_8(u0))  # 8, 2 ,2

        # depth * 8, depth * 8                    depth * 8
        u0 = self.u1(u0)
        u1 = torch.cat((u0, d6), 1)
        u1 = self.leaky_relu(self.conv_16_8(u1))

        # depth * 8, depth * 8                    depth * 4
        u1 = self.u1(u1)
        u2 = torch.cat((u1, d5), 1)
        u2 = self.leaky_relu(self.conv_16_4(u2))

        # depth * 4 , depth * 4                   depth * 2
        u2 = self.u2(u2)
        u3 = torch.cat((u2, d4), 1)
        u3 = self.leaky_relu(self.conv_8_4(u3))

        # depth * 4 , depth * 4                   depth * 2
        u3 = self.u3(u3)
        u4 = torch.cat((u3, d3), 1)
        u4 = self.leaky_relu(self.conv_8_2(u4))

        # depth * 2 , depth * 2                   depth * 1
        u4 = self.u4(u4)
        u5 = torch.cat((u4, d2), 1)
        u5 = self.leaky_relu(self.conv_4_2(u5))

        # depth * 2 , depth * 2                   depth * 1
        u5 = self.u5(u5)
        u6 = torch.cat((u5, d1), 1)
        u6 = self.leaky_relu(self.conv_4_1(u6))

        # depth * 1 , depth * 1                   depth * 1
        u6 = self.u6(u6)
        u7 = torch.cat((u6, d0), 1)
        u7 = self.leaky_relu(self.conv_2_1(u7))
        output = self.u7(u7)

        return output
