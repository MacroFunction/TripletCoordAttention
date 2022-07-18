from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn as nn
import time
import os
# from tensorboardX import SummaryWriter
from LabelSmoothing import LSR
from ghost_tca2 import ghostnet


def train(net, device, epochs, learning_rate,
          weight_decay, dir):
    # odel_weight_path = "./models/state_dict_73.98.pth"
    # model_weight_path = "./mobilenet_v2-b0353104.pth"
    # assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    # pre_weights = torch.load(model_weight_path, map_location=device)
    #
    # pre_dict = {k: v for k, v in pre_weights.items() if
    #             k in net.state_dict() and net.state_dict()[k].numel() == v.numel()}
    # missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    #
    # # for name, value in net.named_parameters():
    # #     if (name in missing_keys):
    # #         value.requires_grad = True
    # # model.load_state_dict(torch.load("mobilenet_v2-b0353104.pth"))
    # net.load_state_dict(pre_dict, strict=False)
    # # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # schedular = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # # if loss do not change for 5 epochs, change lr*0.1
    # # schedular_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=5, verbose=True, eps=1e-5)
    # # schedular = GradualWarmupScheduler(optimizer, multiplier=3.0, total_epoch=5, after_scheduler=schedular_r)
    # #schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=5)

    # loss_function = LSR()
    model_weight_path = "./models/state_dict_73.98.pth"
    # pre_weights = torch.load(model_weight_path, map_location=device)
    # net.load_state_dict(pre_weights, strict=False)
    #
    # pre_dict = {k: v for k, v in pre_weights.items() if
    #             k in net.state_dict() and net.state_dict()[k].numel() == v.numel()}
    # missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    #
    # for name, value in net.named_parameters():
    #     if (name in missing_keys):
    #         value.requires_grad = True
    net.load_state_dict(torch.load("./models/best_73.604_78.13272113867005.pth"))
    # # net.load_state_dict(pre_dict, strict=False)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, momentum=0.9,
    #                       weight_decay=4e-5)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9,
                          weight_decay=4e-5)

    # if loss do not change for 5 epochs, change lr*0.1
    schedular_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3,
                                                             verbose=True, eps=1e-5)
    schedular = GradualWarmupScheduler(optimizer, multiplier=5.0, total_epoch=5, after_scheduler=schedular_r)
    initepoch = 1
    batch_size = 128
    loss = LSR()
    best_test_acc = 0
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers

    # writer = SummaryWriter()
    torch.backends.cudnn.benchmark = True
    traindir = os.path.join(dir, 'train')
    valdir = os.path.join(dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True)
    # for epoch in range(initepoch, epochs):  # loop over the dataset multiple times
    #     print('epoch %d,  lr: %.7f'
    #           % (epoch + 1, optimizer.param_groups[0]['lr']))
    #     print(schedular.last_epoch)
    #     schedular.step(metrics=1)
    global_step = 1
    for epoch in range(initepoch, epochs):  # loop over the dataset multiple times
        sum_step = 1
        running_loss = 0.0
        correct = 0
        total = 0

        net.train()
        timestart = time.time()

        train_bar = tqdm(train_loader)
        for i, data in enumerate(train_bar):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            running_loss += l.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_acc = 100.0 * correct / total
            # writer.add_scalar('loss', running_loss / sum_step, global_step=global_step)
            # writer.add_scalar('acc', train_acc, global_step=global_step)
            train_bar.desc = "train epoch[%d/%d] loss:%.3f train_acc:%.3f lr:%.7f" % \
                             (epoch, epochs, running_loss / sum_step, train_acc, optimizer.param_groups[0]['lr'])
            sum_step += 1
            global_step += 1
            # schedular.step(metrics=train_acc)
        # print('epoch %d, loss: %.4f,tran Acc: %.3f%%,time:%3f sec, lr: %.7f'
        #       % (epoch+1, running_loss / sum_step, train_acc, time.time() - timestart, schedular.param_groups[0]['lr']))
        # print(schedular.last_epoch)

        # test
        net.eval()
        valtotal = 0
        valcorrect = 0
        with torch.no_grad():
            for data in tqdm(validate_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # print(outputs.shape)
                _, predicted = torch.max(outputs.data, 1)
                valtotal += labels.size(0)
                valcorrect += (predicted == labels).sum().item()
            test_acc = 100.0 * valcorrect / valtotal
            print('test Acc: %.3f%%' % (test_acc))
            torch.save(net.state_dict(), './models/best_' + str(test_acc) + '_' + str(train_acc) + '.pth')

        schedular.step(metrics=test_acc)
    print('Finished Training')
    print('best test acc epoch: %d' % epoch + 1)


if __name__ == '__main__':
    model = ghostnet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train(net=model, device=device, epochs=200, learning_rate=1E-5,
          weight_decay=4e-5, dir='H:/ImageNet')
