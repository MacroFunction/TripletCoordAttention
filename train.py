import os
import time
import argparse
import logging
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ghost_tca2 import ghostnet
from tqdm import tqdm
from sklearn.metrics import accuracy_score



def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_true.cpu(), y_pred_cls.cpu())

def train(dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    traindir = os.path.join(dir, 'ILSVRC2012_img_train')
    valdir = os.path.join(dir, 'ILSVRC2012_img_val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = ghostnet(num_classes=args.num_classes, width=args.width, dropout=args.dropout)

    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "./models/state_dict_73.98.pth"
    pre_model_weight_path = "./models/model72.266.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location=device)

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)


    for name, value in model.named_parameters():
        if (name in missing_keys):
            value.requires_grad = True
    model.load_state_dict(torch.load(pre_model_weight_path))
    model.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, momentum=0.9, weight_decay=4e-5)
    # model.optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

    scheduler = CosineAnnealingWarmRestarts(model.optimizer, T_0=5)

    model.loss_func = nn.CrossEntropyLoss()
    model.metric_func = accuracy
    model.metric_name = "accuracy"
    model.to(device)
    epochs = 100
    log_step_freq = 100
    # print(model)
    train_steps = len(train_loader)
    val_num = len(validate_loader)

    best_acc = 0.0
    metric_sum = 0.0
    running_loss = 0.0

    writer = SummaryWriter()
    sum_step = 1
    # writer.add_graph(model, input_to_model=None, verbose=False)
    for epoch in range(1, epochs + 1):

        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_step = 1
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[%d/%d] val_accuracy: %.3f " % \
                               (epoch, epochs, acc / val_step)
                val_step = val_step + 1
        val_accurate = acc / val_num
        print('[epoch %d]  val_accuracy: %.3f' %
              (epoch, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            save_path = './models/model' + str(best_acc) + '.pth'
            torch.save(model.state_dict(), save_path)

        # 1，训练循环-------------------------------------------------
        model.train()
        train_bar = tqdm(train_loader)
        for step, (features, labels) in enumerate(train_bar, 1):
            # 梯度清零
            model.optimizer.zero_grad()

            # 正向传播求损失
            predictions = model(features.to(device))
            loss = model.loss_func(predictions, labels.to(device))
            metric = model.metric_func(predictions, labels.to(device))

            # 反向传播求梯度
            loss.backward()
            model.optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            metric_sum += metric.item()
            writer.add_scalar('loss', running_loss / sum_step, global_step=sum_step)
            writer.add_scalar('acc', metric_sum / sum_step, global_step=sum_step)
            train_bar.desc = ("train epoch[%d/%d] loss:%.3f " + model.metric_name + ":%.3f") % \
                             (epoch, epochs, running_loss / sum_step, metric_sum / sum_step)
            sum_step = sum_step + 1


    #
    # torch.save(model.state_dict(), './models//model.pt')


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
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--num-classes', type=int,
                        default=1000, help='Number classes in dataset')
    parser.add_argument('--width', type=float, default=1.0,
                        help='Width ratio (default: 1.0)')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='PCT',
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--num-gpu', type=int, default=1,
                        help='Number of GPUS to use')
    args = parser.parse_args()


    train(args.data)
