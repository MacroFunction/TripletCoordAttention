import csv
import paddle
import paddle.nn as nn
import importlib
import paddle.nn.functional as F


class CA(nn.Layer):
    def __init__(self, in_ch, reduction=32):
        super(CA, self).__init__()
    #
    #     self.pool_h = nn.AdaptiveAvgPool2D((None, 1))
    #     self.pool_w = nn.AdaptiveAvgPool2D((1, None))
    #
    #     mip = max(8, in_ch // reduction)
    #
    #     self.conv1 = nn.Conv2D(in_ch, mip, kernel_size=1, stride=1, padding=0)
    #     self.bn1 = nn.BatchNorm2D(mip)
    #     self.act = nn.Hardswish()
    #
    #     self.conv_h = nn.Conv2D(mip, in_ch, kernel_size=1, stride=1, padding=0)
    #     self.conv_w = nn.Conv2D(mip, in_ch, kernel_size=1, stride=1, padding=0)
    #
    # def forward(self, x):
    #     identity = x
    #
    #     n, c, h, w = x.shape
    #     x_h = self.pool_h(x)
    #     x_w = self.pool_w(x).transpose([0, 1, 3, 2])
    #
    #     y = paddle.concat([x_h, x_w], axis=2)
    #     y = self.conv1(y)
    #     y = self.bn1(y)
    #     y = self.act(y)
    #
    #     x_h, x_w = paddle.split(y, [h, w], axis=2)
    #     x_w = x_w.transpose([0, 1, 3, 2])
    #
    #     x_h = F.sigmoid(self.conv_h(x_h))
    #     x_w = F.sigmoid(self.conv_w(x_w))
    #
    #     out = identity * x_w * x_h
    #
    #     return out
    #     reducation_c = make_divisible(in_c * reduction_ratio, 4)
        mip = max(8, in_ch // reduction)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_ch, mip, kernel_size=1),
            nn.ReLU(),
            nn.Conv2D(mip, in_ch, kernel_size=1),
            nn.Hardsigmoid()
        )

        self.pool_h = nn.AdaptiveAvgPool2D((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2D((1, None))
        self.conv1 = nn.Conv2D(in_ch, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(mip)
        self.act = nn.ReLU()
        self.conv_h = nn.Conv2D(mip, 1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2D(mip, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.shape

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose([0, 1, 3, 2])

        y = paddle.concat([x_h, x_w], axis=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = paddle.split(y, [h, w], axis=2)
        x_w = x_w.transpose([0, 1, 3, 2])

        x_h = F.sigmoid(self.conv_h(x_h))
        x_w = F.sigmoid(self.conv_w(x_w))

        return x * self.block(x) * x_w * x_h
# # validation
# if __name__ == '__main__':
#     ca = CA(512)                     # in_channel
#     x = paddle.randn([64,512,14,14]) # (batchsize, channel, H, W)
#     y = ca(x)
#     y.shape
import paddle
import paddle.nn as nn


class BottleneckBlock(nn.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2D(
            width,
            width,
            3,
            padding=dilation,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias_attr=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2D(
            width, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.ca = CA(planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.ca(out)  # add CA
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):

    def __init__(self, block, depth, num_classes=1000, with_pool=True):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.num_classes = num_classes
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, 1, 64,
                  previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)

        return x


def ca_resnet50(**kwargs):
    return ResNet(BottleneckBlock, 50, **kwargs)
# 利用高阶 API 查看模型
ca_res50 = ca_resnet50(num_classes=10)
paddle.Model(ca_res50).summary((1,3,224,224))

import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10
print(paddle.device.get_device())

paddle.device.set_device('gpu:0')

# 数据准备
transform = T.Compose([
    T.Resize(size=(224,224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],data_format='HWC'),
    T.ToTensor()
])

train_dataset = Cifar10(mode='train', transform=transform)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=True, num_workers=4)



f_loss = open('resnet_50_loss1.csv', 'w')
f_acc = open('resnet_50_acc1.csv', 'w')
fca_loss = open('resnet_tca_loss1.csv', 'w')
fca_acc = open('resnet_tca_acc1.csv', 'w')


# 模型准备
ca_res50 = ca_resnet50(num_classes=10)
ca_res50.train()
# 训练准备
epoch_num = 10
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=ca_res50.parameters())
loss_fn = paddle.nn.CrossEntropyLoss()

ca_res50_loss = []
ca_res50_acc = []

for epoch in range(epoch_num):
    for batch_id, data in enumerate(train_loader):
        inputs = data[0]
        labels = data[1].unsqueeze(1)
        predicts = ca_res50(inputs)

        loss = loss_fn(predicts, labels)
        acc = paddle.metric.accuracy(predicts, labels)
        loss.backward()

        if batch_id % 100 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))

        if batch_id % 20 == 0:
            ca_res50_loss.append(loss.numpy())
            ca_res50_acc.append(acc.numpy())

        optim.step()
        optim.clear_grad()
csv_write_fca_loss = csv.writer(fca_loss)
csv_write_fca_acc = csv.writer(fca_acc)
csv_write_fca_loss.writerows(ca_res50_loss)
csv_write_fca_acc.writerows(ca_res50_acc)



# 模型准备
res50 = paddle.vision.models.resnet50(num_classes=10)
res50.train()
# # 训练准备
epoch_num = 10
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=res50.parameters())
loss_fn = paddle.nn.CrossEntropyLoss()
res50_loss = []
res50_acc = []

for epoch in range(epoch_num):
    for batch_id, data in enumerate(train_loader):
        inputs = data[0]
        labels = data[1].unsqueeze(1)
        predicts = res50(inputs)

        loss = loss_fn(predicts, labels)
        acc = paddle.metric.accuracy(predicts, labels)
        loss.backward()

        if batch_id % 100 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))

        if batch_id % 20 == 0:

            res50_loss.append(loss.numpy())
            res50_acc.append(acc.numpy())

        optim.step()
        optim.clear_grad()


import matplotlib.pyplot as plt

plt.figure(figsize=(18, 12))
plt.subplot(211)

plt.xlabel('iter')
plt.ylabel('loss')
plt.title('train loss')

x=range(len(ca_res50_loss))
plt.plot(x,res50_loss,color='b',label='ResNet50')
plt.plot(x,ca_res50_loss,color='r',label='ResNet50 + CA')

plt.legend()
plt.grid()

plt.subplot(212)
plt.xlabel('iter')
plt.ylabel('acc')
plt.title('train acc')

x=range(len(ca_res50_acc))
plt.plot(x, res50_acc, color='b',label='ResNet50')
plt.plot(x, ca_res50_acc, color='r',label='ResNet50 + CA')

plt.legend()
plt.grid()

plt.show()


csv_write_f_loss = csv.writer(f_loss)
csv_write_f_acc = csv.writer(f_acc)

csv_write_f_loss.writerows(res50_loss)
csv_write_f_acc.writerows(res50_acc)

plt.savefig('img.png')
