import csv

import paddle.device
import paddle.nn as nn
import paddle.nn.functional as F

import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10

paddle.device.set_device('gpu:0')

# 数据准备
transform = T.Compose([
    T.Resize(size=(224,224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],data_format='HWC'),
    T.ToTensor()
])

train_dataset = Cifar10(mode='train', transform=transform)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=True, num_workers=4)

#模型准备
net = paddle.vision.models.mobilenet_v2(num_classes=10)
print(net)