import torch.nn as nn
import torch
from mypackage.model.shared.basic import SpectralNorm


def conv(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    if stride == 1:
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)
    else:
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=stride,
                         padding=1, bias=False)
    return conv


def deconv(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    if stride == 1:
        conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)
    else:
        conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride,
                                  padding=1, bias=False)
    return conv


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, drop_rate=0.2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.lrelu = nn.LeakyReLU(0.1)
        self.conv2 = conv(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride
        self.drop2d = nn.Dropout2d(drop_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop2d(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2d(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.lrelu(out)

        return out


class UpBasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, upsample=None, drop_rate=0.2):
        super(UpBasicBlock, self).__init__()
        self.deconv1 = deconv(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.lrelu = nn.LeakyReLU(0.1)
        self.deconv2 = deconv(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.upsample = upsample
        self.stride = stride
        self.drop2d = nn.Dropout2d(drop_rate)

    def forward(self, x, pre):
        x = torch.cat((x, pre), 1)
        residual = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.drop2d(out)
        out = self.lrelu(out)

        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.drop2d(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.lrelu(out)

        return out


class UpLayer(nn.Module):
    def __init__(self, UpBasicBlock, BasicBlocks):
        super(UpLayer, self).__init__()
        self.UpBasicBlock = UpBasicBlock
        self.BasicBlocks = BasicBlocks

    def forward(self, x, pre):
        out = self.UpBasicBlock(x, pre)
        for block in self.BasicBlocks:
            out = block(out)
        return out


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return nn.functional.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class ResUNet(nn.Module):

    def __init__(self, layers=(2, 2, 2, 2), depth=64):

        super(ResUNet, self).__init__()
        # 1,256,256
        self.init_layer = nn.Sequential(nn.Conv2d(1, depth, kernel_size=3, stride=1, padding=1,
                                                  bias=False),
                                        nn.BatchNorm2d(depth),
                                        nn.LeakyReLU(0.1)
                                        )
        # depth,256,256
        self.down_layer1 = self._make_down_layer(depth * 1, depth * 2, layers[0], stride=2,
                                                 drop_rate=0.2)
        # 2depth,128,128
        self.down_layer2 = self._make_down_layer(depth * 2, depth * 4, layers[1], stride=2,
                                                 drop_rate=0.2)
        # 4depth,64,64
        self.down_layer3 = self._make_down_layer(depth * 4, depth * 8, layers[2], stride=2,
                                                 drop_rate=0.2)
        # 8depth,32,32
        self.bottleneck = self._make_down_layer(depth * 8, depth * 8, layers[3], stride=1, drop_rate=0.2)
        # 8depth,32,32 + 8depth,32,32   =>  4depth,64,64
        self.up_layer1 = self._make_up_layer(depth * 8 + depth * 8, depth * 4, layers[0], stride=2,
                                             drop_rate=0.2)
        # 4depth,64,64 + 4depth,64,64   =>  2depth,128,128
        self.up_layer2 = self._make_up_layer(depth * 4 + depth * 4, depth * 2, layers[1], stride=2,
                                             drop_rate=0.2)
        # 2depth,128,128 + 2depth,128,128   =>  depth,256,256
        self.up_layer3 = self._make_up_layer(depth * 2 + depth * 2, depth * 1, layers[2], stride=2,
                                             drop_rate=0.2)
        # depth,256,256
        self.out_layer = nn.Sequential(nn.Conv2d(depth * 1, 1, kernel_size=3, stride=1, padding=1,
                                                 bias=True), nn.Tanh())

    def _make_down_layer(self, in_planes, out_planes, num_blocks, stride=1, drop_rate=0.):
        downsample = None

        if stride != 1:
            downsample = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_planes),
                    )

        layers = []
        layers.append(
                BasicBlock(in_planes, out_planes, stride, downsample=downsample, drop_rate=drop_rate))

        for i in range(1, num_blocks):
            layers.append(
                    BasicBlock(out_planes, out_planes, stride=1, drop_rate=drop_rate))

        return nn.Sequential(*layers)

    def _make_up_layer(self, in_planes, out_planes, num_blocks, stride=1, drop_rate=0.):
        upsample = None

        if stride != 1:
            upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_planes, out_planes,
                              kernel_size=1, stride=1, bias=False),
                    # nn.ConvTranspose2d(in_planes, out_planes,
                    #                    kernel_size=4, stride=stride, padding=1, bias=False),

                    nn.BatchNorm2d(out_planes),
                    )

        basicblocks = []
        upblock = UpBasicBlock(in_planes, out_planes, stride, upsample=upsample, drop_rate=drop_rate)

        for i in range(1, num_blocks):
            basicblocks.append(BasicBlock(out_planes, out_planes, stride=1, drop_rate=drop_rate))

        return UpLayer(upblock, basicblocks)

    def forward(self, x):
        x = self.init_layer(x)
        down_layer1 = self.down_layer1(x)
        down_layer2 = self.down_layer2(down_layer1)
        down_layer3 = self.down_layer3(down_layer2)
        bottleneck = self.bottleneck(down_layer3)
        up_layer1 = self.up_layer1(bottleneck, down_layer3)
        up_layer2 = self.up_layer2(up_layer1, down_layer2)
        up_layer3 = self.up_layer3(up_layer2, down_layer1)
        out = self.out_layer(up_layer3)
        return out


class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=4, stride=2, padding=1, use_sp=True, n_power_iterations=1):
        super(ConvLayer, self).__init__()
        if use_sp:
            self.Conv2d = SpectralNorm(nn.Conv2d(c_in, c_out, kernel_size, stride, padding),
                                       power_iterations=n_power_iterations)
        else:
            self.Conv2d = nn.Conv2d(c_in, c_out, kernel_size, stride, padding)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, input):
        out = self.Conv2d(input)
        out = self.lrelu(out)
        return out


class NLayer_D(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, depth=64):
        super(NLayer_D, self).__init__()
        # 256 x 256
        self.layer1 = ConvLayer(input_nc + output_nc, depth * 1, kernel_size=7, stride=1, padding=3, use_sp=False)
        # 256 x 256
        self.layer2 = ConvLayer(depth, depth * 2, kernel_size=4, stride=2, padding=1)
        # 128 x 128
        self.layer3 = ConvLayer(depth * 2, depth * 4, kernel_size=4, stride=2, padding=1)
        # 64 x 64
        self.layer4 = ConvLayer(depth * 4, depth * 4, kernel_size=4, stride=2, padding=1)
        # 32 x 32
        ## 16 x 16
        self.layer6 = nn.Conv2d(depth * 4, output_nc, kernel_size=3, stride=1, padding=1)
        # 16 x 16 ,1

    def forward(self, x, g):
        out = torch.cat((x, g), 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.layer5(out)
        out = self.layer6(out)
        return out


def resunet18(depth=64):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResUNet([2, 2, 2, 2], depth=depth)
    return model


def nlayer(depth=64):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = NLayer_D(depth=depth)
    return model


if __name__ == '__main__':
    from jdit.model import Model

    model = resunet18(32)
    m = Model(model)
    print(m.num_params)
    # d = nlayer(8)
    # x = torch.randn(1, 1, 256, 256)
    # y = model(x)
