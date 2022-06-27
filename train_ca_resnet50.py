import os
import argparse
from tensorboardX import SummaryWriter

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

# from ghost_tca2 import ghostnet
from resnet_ca import resnet50


# 优化器
import torch.optim as optim



def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_true.cpu(), y_pred_cls.cpu())

def train(dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ###############数据加载与预处理
    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # 训练集
    trainset = torchvision.datasets.CIFAR10(root='./data/',
                                   train=True,
                                   download=False,
                                   transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=48,
                                          shuffle=True,
                                          num_workers=4)
    # 测试集
    testset = torchvision.datasets.CIFAR10(root='./data/',
                                  train=False,
                                  download=False,
                                  transform=transform)

    testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = resnet50(10)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.Adam(model.parameters(), lr=0.001)
    model.loss_func = nn.CrossEntropyLoss()
    model.metric_func = accuracy
    model.metric_name = "accuracy"
    model.to(device)
    epochs = 100
    train_steps = len(trainloader)
    val_num = len(testloader)

    best_acc = 0.0
    metric_sum = 0.0
    running_loss = 0.0

    writer = SummaryWriter()
    sum_step = 1
    # writer.add_graph(model, input_to_model=None, verbose=False)
    for epoch in range(1, epochs + 1):


        # 1，训练循环-------------------------------------------------
        model.train()
        train_bar = tqdm(trainloader)
        for step, (features, labels) in enumerate(train_bar, 1):
            # 梯度清零
            # model.optimizer.zero_grad()

            # 正向传播求损失
            predictions = model(features.to(device))
            loss = model.loss_func(predictions, labels.to(device))
            metric = model.metric_func(predictions, labels.to(device))

            # 反向传播求梯度
            loss.backward()
            # model.optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            metric_sum += metric.item()
            writer.add_scalar('loss', running_loss / sum_step, global_step=sum_step)
            writer.add_scalar('acc', metric_sum / sum_step, global_step=sum_step)
            train_bar.desc = ("train epoch[%d/%d] loss:%.3f " + model.metric_name + ":%.3f") % \
                             (epoch, epochs, running_loss / sum_step, metric_sum / sum_step)
            sum_step = sum_step + 1
        #
        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_step = 1
        with torch.no_grad():
            val_bar = tqdm(testloader)
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
                        default=10, help='Number classes in dataset')
    parser.add_argument('--width', type=float, default=1.0,
                        help='Width ratio (default: 1.0)')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='PCT',
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--num-gpu', type=int, default=1,
                        help='Number of GPUS to use')
    args = parser.parse_args()


    train(args.data)
