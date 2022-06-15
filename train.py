import os
import time
import argparse
import logging
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ghost_tca import ghostnet
from sklearn.metrics import accuracy_score



def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu(), y_pred_cls.cpu())

def train(valdir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(160),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    model = ghostnet(num_classes=args.num_classes, width=args.width, dropout=args.dropout)
    model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.loss_func = nn.CrossEntropyLoss()
    model.metric_func = accuracy
    model.metric_name = "accuracy"
    model_weight_path = "./models/state_dict_73.98.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location=device)

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)
    model.to(device)
    epochs = 64
    log_step_freq = 100

    for epoch in range(1, epochs + 1):
        # 1，训练循环-------------------------------------------------
        print(epoch)
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        for step, (features, labels) in enumerate(loader, 1):
            # 梯度清零
            model.optimizer.zero_grad()

            # 正向传播求损失
            predictions = model(features.to(device))
            loss = model.loss_func(predictions, labels.to(device))
            metric = model.metric_func(predictions, labels.to(device))

            # 反向传播求梯度
            loss.backward()
            model.optimizer.step()

            # 打印batch级别日志
            loss_sum += loss.item()
            metric_sum += metric.item()
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, " + model.metric_name + ": %.3f") %
                      (step, loss_sum / step, metric_sum / step))
    torch.save(model.state_dict(), './models//model.pt')


def valid_step(model, features, labels):
    # 预测模式，dropout层不发生作用
    model.eval()

    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)

    return loss.item(), metric.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
    parser.add_argument('--data', metavar='DIR', default='D:/ImageNet/data/ImageNet2012',
                        help='path to dataset')
    parser.add_argument('--output_dir', metavar='DIR', default='./models/',
                        help='path to output files')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number classes in dataset')
    parser.add_argument('--width', type=float, default=1.0,
                        help='Width ratio (default: 1.0)')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='PCT',
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--num-gpu', type=int, default=1,
                        help='Number of GPUS to use')
    args = parser.parse_args()

    valdir = os.path.join(args.data, 'ILSVRC2012_img_train')
    train(valdir)
