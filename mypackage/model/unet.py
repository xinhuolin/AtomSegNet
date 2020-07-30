# coding=utf-8
from .shared.basic import *
from functools import partial
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, depth=32, blocks=6, input_nc=1, output_nc=1, dilation=1,
                 norm_type="batch", active_type="LeakyReLU"):
        super(Unet, self).__init__()
        self.depth = depth
        blocks = blocks
        max_depth = depth * (2 ** (blocks - 1))
        num_hidden_blocks = blocks - 1
        dilation = dilation
        self.input = NormalBlock(input_nc, depth, (3, 1, 1), active_type=active_type,
                                 norm_type=norm_type, use_spectralnorm=False)
        # down sample block 2-6, (0,1,2,3,4)
        self.downsample = []
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

    def forward(self, input):
        d_result = []
        _input = self.input(input)
        d_result.append(_input)
        # down sample block 2-6, (0,1,2,3,4)
        for name in self.downsample:
            _input = self._modules[name](_input)
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


class NLD_SpecNrom(nn.Module):
    def __init__(self, depth=64, input_nc=1, output_nc=1):
        super(NLD_SpecNrom, self).__init__()
        self.up = partial(F.interpolate, scale_factor=2)
        # 512 x 512
        self.layer0 = nn.Sequential(
            SpectralNorm(nn.Conv2d(input_nc+output_nc, depth//2, kernel_size=7, stride=1, padding=3)),
            nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 256 x 256
        self.layer1 = nn.Sequential(
                SpectralNorm(nn.Conv2d(input_nc+output_nc, depth, kernel_size=3, stride=1, padding=2, dilation=2)),
                nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 128 x 128
        self.layer2 = nn.Sequential(
                SpectralNorm(nn.Conv2d(depth, depth * 2, kernel_size=3, stride=1, padding=2, dilation=2)),
                nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 64 x 64
        self.layer3 = nn.Sequential(
                SpectralNorm(nn.Conv2d(depth * 2, depth * 4, kernel_size=3, stride=1, padding=2, dilation=2)),
                nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 32 x 32
        self.layer4 = nn.Sequential(
                SpectralNorm(nn.Conv2d(depth * 4, depth * 8, kernel_size=3, stride=1, padding=2, dilation=2)),
                nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 16 x 16
        self.layer5 = nn.Sequential(
                SpectralNorm(nn.Conv2d(depth * 8, depth * 8, kernel_size=3, stride=1, padding=2, dilation=2)),
                nn.LeakyReLU(0.2), nn.AvgPool2d(2, 2))
        # 8 x 8
        self.layer6 = nn.Sequential(nn.Conv2d(depth * 8, output_nc, kernel_size=8, stride=1, padding=0))

    def forward(self, x, g):
        g_up = self.up(g)
        out = torch.cat((x, g_up), 1)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        return out
