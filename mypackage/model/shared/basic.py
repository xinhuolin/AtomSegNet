# coding=utf-8

import torch
from torch import nn
from torch.nn import Parameter


class DownsampleBlock(nn.Module):
    def __init__(self, c_in, c_out, ksp=(3, 1, 1), pool_type="Max", active_type="ReLU",
                 norm_type=None, groups=1, dilation=1, use_spectralnorm=False):
        super(DownsampleBlock, self).__init__()
        kernel_size, stride, padding = ksp
        if stride != 1:
            assert kernel_size - 2 * padding == stride, "(k,s,p) is (%d,%d,%d) can not downsample!" % (
                kernel_size, stride, padding)
            self.poolLayer = None
        else:
            assert pool_type is not None, "You should use stride=2 or pooling layer to downsample!"
            self.poolLayer = getPoolLayer(pool_type)
        self.convLayer_1 = ConvLayer(c_in, c_out, kernel_size, stride, padding, active_type, norm_type, dilation,
                                     groups,
                                     use_spectralnorm=use_spectralnorm)

    def forward(self, input):
        out = self.convLayer_1(input)
        if self.poolLayer is not None:
            out = self.poolLayer(out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, c_in, c_out, ksp=(3, 1, 1), active_type="ReLU", norm_type=None, groups=1,
                 dilation=1,
                 use_spectralnorm=False):
        super(UpsampleBlock, self).__init__()
        kernel_size, stride, padding = ksp
        if stride != 1:
            assert kernel_size - 2 * padding == stride, "(k,s,p) is (%d,%d,%d) can not upsample!" % (
                kernel_size, stride, padding)
            self.upSampleLayer = None
            self.convLayer = DeconvLayer(c_in, c_out, kernel_size, stride, padding, active_type, norm_type, dilation,
                                         groups,
                                         use_spectralnorm=use_spectralnorm)
        else:
            assert stride == 1 and kernel_size == 2 * padding + 1
            self.upSampleLayer = nn.functional.interpolate
            self.convLayer = ConvLayer(c_in, c_out, kernel_size, stride, padding, active_type, norm_type, dilation,
                                       groups,
                                       use_spectralnorm=use_spectralnorm)

    def forward(self, prev_input, now_input):
        out = torch.cat((prev_input, now_input), 1)
        if self.upSampleLayer:
            out = self.upSampleLayer(out,scale_factor=2)
        out = self.convLayer(out)
        return out


class NormalBlock(nn.Module):
    def __init__(self, c_in, c_out, ksp=(3, 1, 1), active_type="ReLU", norm_type="batch", groups=1,
                 dilation=1,
                 use_spectralnorm=False):
        super(NormalBlock, self).__init__()
        kernel_size, stride, padding = ksp
        assert stride + 2 * padding - kernel_size == 0, \
            "kernel_size%s, stride%s, padding%s is not a 'same' group! s+2p-k ==0" % (
                kernel_size, stride, padding)

        self.convLayer_1 = ConvLayer(c_in, c_out, kernel_size, stride, padding, active_type, norm_type,
                                     dilation=dilation, groups=groups, use_spectralnorm=use_spectralnorm)

    def forward(self, input):
        out = input
        out = self.convLayer_1(out)
        return out


class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, active_type="ReLU", norm_type=None,
                 dilation=1,
                 groups=1,
                 use_spectralnorm=False):
        super(ConvLayer, self).__init__()
        norm = getNormLayer(norm_type)
        act = getActiveLayer(active_type)
        bias = norm_type is None
        self.module_list = nn.ModuleList([])

        if use_spectralnorm:
            self.module_list += [
                SpectralNorm(
                        nn.Conv2d(c_in, c_out, kernel_size, stride, padding, groups=groups, dilation=dilation,
                                  bias=bias))]
        else:
            self.module_list += [
                nn.Conv2d(c_in, c_out, kernel_size, stride, padding, groups=groups, dilation=dilation, bias=bias)]
        if norm:
            self.module_list += [norm(c_out)]
        self.module_list += [act]

    def forward(self, input):
        out = input
        for layer in self.module_list:
            out = layer(out)
        return out


class DeconvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=4, stride=2, padding=1, active_type="ReLU", norm_type=None,
                 dilation=1,
                 groups=1,
                 use_spectralnorm=False):
        super(DeconvLayer, self).__init__()
        norm = getNormLayer(norm_type)
        act = getActiveLayer(active_type)
        bias = norm_type is None
        self.module_list = nn.ModuleList([])
        if use_spectralnorm:
            self.module_list += [
                SpectralNorm(
                        nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding, dilation=dilation, groups=groups,
                                           bias=bias))]
        else:
            self.module_list += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding, dilation=dilation, groups=groups,
                                   bias=bias)]
        if norm:
            self.module_list += [norm(c_out, affine=True)]
        self.module_list += [act]

    def forward(self, input):
        out = input
        for layer in self.module_list:
            out = layer(out)
        return out


# ___________________for unet________________

def getNormLayer(norm_type):
    norm_layer = nn.BatchNorm2d
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif norm_type == 'layer':
        norm_layer = nn.LayerNorm
    elif norm_type == 'group':
        norm_layer = nn.GroupNorm
    elif norm_type is None:
        norm_layer = None
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def getPoolLayer(pool_type):
    pool_layer = nn.MaxPool2d(2, 2)
    if pool_type == 'Max':
        pool_layer = nn.MaxPool2d(2, 2)
    elif pool_type == 'Avg':
        pool_layer = nn.AvgPool2d(2, 2)
    elif pool_type is None:
        pool_layer = None
    else:
        print('pool layer [%s] is not found' % pool_layer)
    return pool_layer


def getActiveLayer(active_type):
    active_layer = nn.ReLU()
    if active_type == 'ReLU':
        active_layer = nn.ReLU()
    elif active_type == 'LeakyReLU':
        active_layer = nn.LeakyReLU(0.1)
    elif active_type == 'Tanh':
        active_layer = nn.Tanh()
    elif active_type == 'Sigmoid':
        active_layer = nn.Sigmoid()
    elif active_type is None:
        active_layer = None
    else:
        print('active layer [%s] is not found' % active_layer)
    return active_layer


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self.l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
