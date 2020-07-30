# coding=utf-8
from .shared.basic import *
import torch.nn.functional as F
import math


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, convlayers = 6, reduction = 0.5, bottleneck=False):
        super(DenseNet, self).__init__()

        # nDenseBlocks = (convlayers - 4) // 3
        # if bottleneck:
        #     nDenseBlocks //= 2
        nDenseBlocks = convlayers

        nChannels = [2 * growthRate]
        for i in range(4):
            nChannels.append(nChannels[i] + nDenseBlocks * growthRate)

        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, padding=1,
                               bias=False)

        self.down_dense1 = self._make_dense(nChannels[0], growthRate, nDenseBlocks, bottleneck) #2g, 2g + g

        self.down_trans1 = Transition(nChannels[1],  int(math.floor(nChannels[1] * reduction))) #2g, (2g + g)/d  down

        self.down_dense2 = self._make_dense(nChannels[1], growthRate, nDenseBlocks, bottleneck) #3g, 3g + g

        self.down_trans2 = Transition(nChannels[2],  int(math.floor(nChannels[2] * reduction))) #3g, (3g + g)/d  down
        # ------------------------------------------------------------------------------------------
        self.bottleneck = self._make_dense(nChannels[3], growthRate, nDenseBlocks, bottleneck) #4g, 4g + g
        # ------------------------------------------------------------------------------------------
        self.up_trans1 = Transition(nChannels[2],  int(math.floor(nChannels[2] * reduction))) #3g, (3g + g)/d  up

        self.up_dense2 = self._make_dense(nChannels[1], growthRate, nDenseBlocks, bottleneck) #3g, 3g + g

        self.up_trans2 = Transition(nChannels[2],  int(math.floor(nChannels[2] * reduction))) #3g, (3g + g)/d  up

        self.up_dense3 = self._make_dense(nChannels[3], growthRate, nDenseBlocks, bottleneck) #4g, 4g + g


    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)

        return out


def denseNet(growthRate=12, depth=100):
    denseNet = DenseNet(growthRate=growthRate, convlayers=depth, reduction=0.5,
                        bottleneck=False)
    return denseNet


class Unetppslim(nn.Module):
    def __init__(self, depth=32, blocks=6, input_nc=1, output_nc=1, norm_type="batch",
                 active_type="LeakyReLU"):
        super(Unetppslim, self).__init__()
        self.depth = depth
        blocks = blocks
        max_depth = depth * (2 ** (blocks - 1))
        num_hidden_blocks = blocks - 1
        dilation = 1

        growth_rate = 6
        self.input = NormalBlock(input_nc, depth, (3, 1, 1), active_type=active_type,
                                 norm_type=norm_type, use_spectralnorm=False)
        # down sample block 2-6, (0,1,2,3,4)
        self.downsample = []
        self.downsample_block = self._make_dense(2 * growth_rate, growth_rate)
        for i in range(num_hidden_blocks):
            if i == num_hidden_blocks - 1:
                pool_type = "Avg"
            else:
                pool_type = "Max"
            self.add_module("downsample_block_" + str(i + 1),
                            DownsampleBlock(self._depth(i), self._depth(i + 1),
                                            (3, 1, 1),
                                            pool_type=pool_type,
                                            active_type=active_type,
                                            norm_type=norm_type,
                                            dilation=dilation))
            self.downsample += ["downsample_block_" + str(i + 1)]

        # bottle neck block
        self.bottleneck = NormalBlock(max_depth, max_depth, (3, 1, 1), active_type=active_type,
                                      norm_type=norm_type)

        # up sample block 2-6, (4,3,2,1,0)
        self.upsample = []
        for i in range(num_hidden_blocks - 1, -1, -1):
            self.add_module("upsample_block_" + str(i + 1),
                            UpsampleBlock(self._depth(i + 2), self._depth(i),
                                          (3, 1, 1),
                                          active_type=active_type,
                                          norm_type=norm_type,
                                          dilation=dilation))
            self.upsample += ["upsample_block_" + str(i + 1)]

        # output bloack
        self.output = NormalBlock(depth, output_nc, (3, 1, 1), active_type="Tanh", norm_type=None)

    def _depth(self, i):
        return self.depth * (2 ** i)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, input):
        d_result = []
        _input = self.input(input)
        d_result.append(_input)
        # down sample block 2-6, (0,1,2,3,4)
        for name in self.downsample:
            _input = self._modules[name](_input)
            # _output = self.dens(_input)
            # d_result.append(_input)
            d_result.append(_input)

        _input = self.bottleneck(_input)

        # up sample block 2-6, (4,3,2,1,0)
        for name in self.upsample:
            prev_input = d_result.pop()
            _input = self._modules[name](prev_input, _input)
        # output
        out = self.output(_input)

        return out


class NLD(nn.Module):
    def __init__(self, depth=64, input_nc=1, output_nc=1, norm_type="batch"):
        super(NLD, self).__init__()
        self.norm = getNormLayer(norm_type)
        # self.active = getActiveLayer(active_type)
        # self.use_sigmoid = use_sigmoid
        # self.use_liner = use_liner

        # 256 x 256
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc + output_nc, depth, kernel_size=7, stride=1, padding=3),
                                    self.norm(depth),
                                    nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 128 x 128
        self.layer2 = nn.Sequential(nn.Conv2d(depth, depth * 2, kernel_size=3, stride=1, padding=2, dilation=2),
                                    self.norm(depth * 2),
                                    nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 64 x 64
        self.layer3 = nn.Sequential(nn.Conv2d(depth * 2, depth * 4, kernel_size=3, stride=1, padding=2, dilation=2),
                                    self.norm(depth * 4),
                                    nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 32 x 32
        self.layer4 = nn.Sequential(nn.Conv2d(depth * 4, depth * 8, kernel_size=3, stride=1, padding=2, dilation=2),
                                    self.norm(depth * 8),
                                    nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 16 x 16
        self.layer5 = nn.Sequential(nn.Conv2d(depth * 8, depth * 8, kernel_size=3, stride=1, padding=2, dilation=2),
                                    self.norm(depth * 8),
                                    nn.LeakyReLU(0.2), nn.AvgPool2d(2, 2))
        # 8 x 8
        self.layer6 = nn.Sequential(nn.Conv2d(depth * 8, output_nc, kernel_size=8, stride=1, padding=0))

    def forward(self, x, g):
        out = torch.cat((x, g), 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        return out
