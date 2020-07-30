import math
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision

from torch.nn.utils import spectral_norm
####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % pad_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [%s]' % mode
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
                  dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


####################
# Useful blocks
####################


class ResNetBlock(nn.Module):
    """
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    """

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
                 bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
                           norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
                           norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                 norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc + gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc + 2 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc + 3 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc + 4 * gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                 norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
                                          norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
                                          norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
                                          norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                       pad_type='zero', norm_type=None, act_type='relu'):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias,
                      pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                 pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias,
                      pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)


class RRDB_Net(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                 mode='CNA', res_scale=1, upsample_mode='upconv'):
        super(RRDB_Net, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                          norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = sequential(fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)), \
                                *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


####################
# Discriminator
####################


# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
                           mode=mode)
        conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 64, 64
        conv2 = conv_block(base_nf, base_nf * 2, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv3 = conv_block(base_nf * 2, base_nf * 2, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 32, 128
        conv4 = conv_block(base_nf * 2, base_nf * 4, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv5 = conv_block(base_nf * 4, base_nf * 4, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 16, 256
        conv6 = conv_block(base_nf * 4, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv7 = conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 8, 512
        conv8 = conv_block(base_nf * 8, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv9 = conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 4, 512
        self.features = sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, \
                                   conv9)

        # classifier
        self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
                           mode=mode)
        conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 48, 64
        conv2 = conv_block(base_nf, base_nf * 2, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv3 = conv_block(base_nf * 2, base_nf * 2, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 24, 128
        conv4 = conv_block(base_nf * 2, base_nf * 4, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv5 = conv_block(base_nf * 4, base_nf * 4, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 12, 256
        conv6 = conv_block(base_nf * 4, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv7 = conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 6, 512
        conv8 = conv_block(base_nf * 8, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv9 = conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 3, 512
        self.features = sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, \
                                   conv9)

        # classifier
        self.classifier = nn.Sequential(
                nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_nc, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
                           mode=mode)
        conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 96, 64
        conv2 = conv_block(base_nf, base_nf * 2, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv3 = conv_block(base_nf * 2, base_nf * 2, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 48, 128
        conv4 = conv_block(base_nf * 2, base_nf * 4, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv5 = conv_block(base_nf * 4, base_nf * 4, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 24, 256
        conv6 = conv_block(base_nf * 4, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv7 = conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 12, 512
        conv8 = conv_block(base_nf * 8, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        conv9 = conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type, \
                           act_type=act_type, mode=mode)
        # 6, 512
        conv10 = conv_block(base_nf * 8, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type, \
                            act_type=act_type, mode=mode)
        conv11 = conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type, \
                            act_type=act_type, mode=mode)
        # 3, 512
        self.features = sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, \
                                   conv9, conv10, conv11)

        # classifier
        self.classifier = nn.Sequential(
                nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


####################
# Perceptual Network
####################


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


# Assume input range is [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, \
                 device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
                torch.load('../experiments/pretrained_models/VGG16minc_53.pth'), strict=True)
        self.features.eval()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output

def define_F(device, use_bn=False):
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF