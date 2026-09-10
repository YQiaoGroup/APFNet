# -*- coding: utf-8 -*-
"""
Created on 2023/05/21 21:47

@Author : zhengyx :)

参考CSDN: https://blog.csdn.net/hbw136/article/details/125071858
GitHub代码: https://github.com/bytedance/RLFN
"""

from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F


def _make_pair(value):
    '''改成3倍kernel size'''
    if isinstance(value, int):
        value = (value,) * 3
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    修改为三维图像适应的卷积层
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2), int((kernel_size[2] - 1) / 2))
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they are passed.
    
    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


'''
由于nn.PixelShuffle(upscale_factor)只支持二维图像, 而我们需要使用三维的PixelShuffle
所以需要自己写一个三维的PixelShuffle
参考代码: https://github.com/assassint2017/PixelShuffle3D/blob/master/PixelShuffle3D.py
'''
class PixelShuffle3D(nn.Module):
    """
    三维PixelShuffle模块
    """
    def __init__(self, upscale_factor):
        """
        :param upscale_factor: tensor的放大倍数
        """
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        batch_size, channels, in_depth, in_height, in_width = inputs.size()
        channels //= self.upscale_factor ** 3
        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor
        input_view = inputs.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
            in_depth, in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    修改为三维图像适应的卷积层
    此处的pixelshuffle操作把原来的nn.PixelShuffle(upscale_factor)替换成了自己写的PixelShuffle3D
    """
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 3), kernel_size)
    pixel_shuffle = PixelShuffle3D(upscale_factor)  # pixelshuffle操作本质上就是图像上采样, 达到放大效果 
    # 输入(N, C*upscale_factor^3, H, W, D)
    # 输出(N, C, H*upscale_factor, W*upscale_factor, D*upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    修改为三维图像适应的卷积层, esa_channels=16
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)  # 1*1conv
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)  # strided conv
        self.conv3 = conv(f, f, kernel_size=3, padding=1)  # Conv Groups
        self.conv4 = conv(f, n_feats, kernel_size=1)  # 1*1conv
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool3d(c1, kernel_size=7, stride=3)  # strided max-pooling
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3), x.size(4)), mode='trilinear', align_corners=False)  # 上采样, (x.size(2), x.size(3))是指定输出尺寸
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RLFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    修改为三维图像适应的卷积层
    """

    def __init__(self, in_channels, mid_channels=None, out_channels=None, esa_channels=16):
        super(RLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)  # 1*1卷积
        self.esa = ESA(esa_channels, out_channels, nn.Conv3d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.esa(self.c5(out))

        return out





