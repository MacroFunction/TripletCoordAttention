from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn as nn
import time
import os
from tensorboardX import SummaryWriter
from ghost_tca2 import ghostnet

def train(net, device, epochs, learning_rate,
          weight_decay, dir):
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # if loss do not change for 5 epochs, change lr*0.1
    schedular_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True, eps=1e-5)
    schedular = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=schedular_r)
    #schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=5)
    initepoch = 0
    batch_size = 100
    loss = nn.CrossEntropyLoss()
    best_test_acc = 0
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 3])  # number of workers

    writer = SummaryWriter()
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
    sum_step = 1
    running_loss = 0.0
    correct = 0
    total = 0
    for epoch in range(initepoch, epochs):  # loop over the dataset multiple times

        net.train()
        timestart = time.time()
        #print(optimizer.param_groups[0]['lr'])
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
            writer.add_scalar('loss', running_loss / sum_step, global_step=sum_step)
            writer.add_scalar('acc', train_acc, global_step=sum_step)
            train_bar.desc = ("train epoch[%d/%d] loss:%.3f train_acc:%.3f") % \
                             (epoch, epochs, running_loss / sum_step, train_acc)
            sum_step += 1
        print('epoch %d, loss: %.4f,tran Acc: %.3f%%,time:%3f sec, lr: %.7f'
              % (epoch+1, running_loss / sum_step, train_acc, time.time() - timestart, optimizer.param_groups[0]['lr']))
        print(schedular.last_epoch)

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
            # if epoch > 30:
            #     torch.save(net.state_dict(), '/root/Desktop/cifar-100/checkpoint_512_512_100/' + str(test_acc) + '_Resnet18.pth')
            if test_acc > best_test_acc:
                print('find best! save at checkpoint/cnn_best.pth')
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(net.state_dict(),
                           './models/best_' + str(best_test_acc) + '_' + str(train_acc) + '.pth')

        schedular.step(metrics=test_acc)


    print('Finished Training')
    print('best test acc epoch: %d' % epoch+1)

if __name__ == '__main__':
    model = ghostnet(num_classes=1000, width=1.0, dropout=0.2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_weight_path = "./models/model73.29.pth"
    # pre_weights = torch.load(model_weight_path, map_location=device)
    # model.load_state_dict(pre_weights, strict=False)
    model.to(device)
    train(net=model, device=device, epochs=100, learning_rate=2e-1,
          weight_decay=4e-5, dir='D:/ImageNet/data/ImageNet2012')