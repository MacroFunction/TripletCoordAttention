import torch
import torch.nn as nn
from torchsummary import summary

from torchstat import stat

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


class CSPOSA(nn.Module):
    def __init__(self, inp):
        super(CSPOSA, self).__init__()
        mip = inp / 2
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        a,b = torch.split(x,[32,32],dim=1)
        b = self.conv2(b)
        c = self.conv3(b)
        d = torch.cat([b,c],dim=1)
        d = self.conv4(d)
        x = torch.cat([x,d], dim=1)


        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CSPOSA(inp=64)#.to(device)

# input = torch.randn(1, 64, 56, 56)
# input=input.to(device)
# y = model(input)
# summary(model, (64, 56, 56))  # 输出网络结构
stat(model, (64, 56, 56))

import torch
import torch.nn as nn
from torchsummary import summary


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


class DRC(nn.Module):
    def __init__(self, inp):
        super(DRC, self).__init__()
        mip = inp / 2
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)


        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        a,b = torch.split(x,[32,32],dim=1)
        b = self.conv2(b)
        c = self.conv3(b)
        d  =  b + c

        d = self.conv4(d)
        x = torch.cat([x,d], dim=1)


        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DRC(inp=64)#.to(device)

# input = torch.randn(1, 64, 56, 56)
# input=input.to(device)
# y = model(input)
# summary(model, (64, 56, 56))  # 输出网络结构
stat(model, (64, 56, 56))