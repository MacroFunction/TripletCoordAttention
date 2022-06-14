import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class TripletCoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(TripletCoordAtt, self).__init__()
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_c = nn.AdaptiveAvgPool2d(1)

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, 1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, 1, kernel_size=1, stride=1, padding=0)
        self.conv_c = nn.Conv2d(mip, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_w = self.pool_w(x)
        x_h = self.pool_h(x).permute(0, 1, 3, 2)
        x_c = self.pool_c(x).permute(0, 3, 2, 1)

        y = torch.cat([x_w, x_h, x_c], dim=3)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_w, x_h, x_c = torch.split(y, [w, h, c], dim=3)

        a_w = self.conv_w(x_w).sigmoid()
        a_h = self.conv_h(x_h).sigmoid().permute(0, 1, 3, 2)
        a_c = self.conv_h(x_c).sigmoid().permute(0, 3, 2, 1)

        out = identity * a_w * a_h * a_c

        return out
