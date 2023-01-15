import os
import sys
import argparse
import math
import time
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from tqdm import tqdm
from utils import *
from loss import *
from model.SevenCNN import CNN, CLDataTransform
from utils.builder import *
from model.MLPHeader import MLPHead
from model.PreResNet import *
from util import *
from fmix import *
from utils.eval import *
from utils.NoisyUtils import *
from model.ResNet32 import resnet32
from model.resnet import resnet50
from data.imbalance_cifar import *
from data.food101 import *
from data.food101n import *
from data.Clothing1M import *
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import statistics

#CIFAR100 100 1/20 10epoch CE
# model=resnet32(num_classes=100).cuda()
# optimizer=optim.SGD(model.parameters(), lr=0.01, weight_decay=4e-4, momentum=0.9, nesterov=True)
# num_classes = 100
# train_dataset = CIFAR100_im(root="./data/cifar100", train=True, meta=False, num_meta=0,
#                             corruption_prob=0, corruption_type='unif', transform="hard",
#                             target_transform=None, download=True, seed=123, imblance=True,
#                             imb_factor=1/20)
#
# test_set = CIFAR100_im(root="./data/cifar100", train=False, meta=False, num_meta=0,
#                        corruption_prob=0, corruption_type='unif', transform="easy",
#                        target_transform=None, download=True, seed=123, imblance=True,
#                        imb_factor=1/20)
#
# trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8,
#                          pin_memory=True)
# test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=8,
#                          pin_memory=False)
#
# for epoch in range(0,100):
#     model.train()
#     train_loss_meter = AverageMeter()
#     train_accuracy_meter = AverageMeter()
#     pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='training')
#     loss_ = torch.zeros(num_classes)
#     num=torch.zeros(num_classes)
#     for it, sample in enumerate(pbar):
#
#         x, y, indices = sample
#         x, y = x.cuda(), y.cuda()
#         outputs = model(x)
#         logits = outputs['logits'] if type(outputs) is dict else outputs
#         loss = F.cross_entropy(logits, y,reduction="none")
#         for i in range(len(y)):
#             num[y[i]]+=1
#             loss_[y[i]]+=loss[i].cpu()
#
#
#         loss=loss.mean()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         train_acc = accuracy(logits, y, topk=(1,))
#         train_accuracy_meter.update(train_acc[0], x.size(0))
#         train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
#         pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')
#
#     print(
#         f'>> Epoch {epoch}: loss {train_loss_meter.avg:.2f} ,train acc {train_accuracy_meter.avg:.2f}')
#
#     with torch.no_grad():
#         for i in range(num_classes):
#             loss_[i] = loss_[i] / num[i]
#         fig = plt.figure()
#         # 生成axes对象
#         axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#
#         x = np.linspace(0, num_classes, num_classes)
#         y = loss_.numpy()
#         plt.bar(x, num.numpy(), alpha=0.3, color='blue', label=u'class num')
#         axes.legend(loc=2)
#         ax2 = axes.twinx()
#         # 绘制散点
#         ax2.plot(x, y, c="green", label="mean loss", ls='-', alpha=0.6, lw=2, zorder=2)
#         # 设置图像标题
#         ax2.legend(loc=1)
#         ax2.set_ylabel('')
#
#
#
#         # 显示图像
#         plt.savefig("./result_log/"+str(epoch) + ".png")
#         plt.xlabel("class id")
#         print(loss_)
#


#CIFAR100 balance 0.4
model=PreActResNet18(num_classes=100).cuda()
optimizer=optim.SGD(model.parameters(), lr=0.005, weight_decay=4e-4, momentum=0.9, nesterov=True)
num_classes = 100
train_dataset = CIFAR100_im(root="./data/cifar100", train=True, meta=False, num_meta=0,
                            corruption_prob=0.4, corruption_type='unif', transform="hard",
                            target_transform=None, download=True, seed=123, imblance=True,
                            imb_factor=1/20)

test_set = CIFAR100_im(root="./data/cifar100", train=False, meta=False, num_meta=0,
                       corruption_prob=0.4, corruption_type='unif', transform="easy",
                       target_transform=None, download=True, seed=123, imblance=True,
                       imb_factor=1/20)

trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8,
                         pin_memory=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=8,
                         pin_memory=False)

for epoch in range(0,100):
    model.train()
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='training')
    loss_ = torch.zeros(num_classes)
    # num=torch.zeros(num_classes)

    num_clean = torch.zeros(num_classes)
    num_noise = torch.zeros(num_classes)
    loss_clean = torch.zeros(num_classes)
    loss_noise = torch.zeros(num_classes)
    loss_clean_cnt=[[] for i in range(num_classes)]
    loss_noise_cnt = [[] for i in range(num_classes)]


    for it, sample in enumerate(pbar):

        x, y, y_true = sample
        x, y = x.cuda(), y.cuda()
        outputs = model(x)
        logits = outputs['logits'] if type(outputs) is dict else outputs
        loss = F.cross_entropy(logits, y,reduction="none")
        loss_
        for i in range(len(y)):
            if int(y[i])==int(y_true[i]):
                num_clean[y[i]]+=1
                loss_clean[y[i]]+=loss[i].cpu()
                loss_clean_cnt[y[i]].append(float(loss[i].cpu()))
            else:
                num_noise[y[i]]+=1
                loss_noise[y[i]]+=loss[i].cpu()
                loss_noise_cnt[y[i]].append(float(loss[i].cpu()))
        loss=loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')

    print(
        f'>> Epoch {epoch}: loss {train_loss_meter.avg:.2f} ,train acc {train_accuracy_meter.avg:.2f}')
    print(num_clean)
    print(num_noise)

    loss_clean_middle = []
    loss_noise_middle = []

    with torch.no_grad():
        for i in range(num_classes):
            loss_clean[i] = loss_clean[i] / num_clean[i]
            loss_noise[i] = loss_noise[i] / num_noise[i]
            a=(torch.Tensor(loss_clean_cnt[i]).sort(descending=True)[0])
            b=(torch.tensor(loss_noise_cnt[i]).sort()[0])
            loss_clean_middle.append(a[:10].mean())
            loss_noise_middle.append(b[:10].mean())

        x = np.linspace(0, num_classes-1, num_classes)
        # plt.bar(x, num_clean.numpy(), align='center', alpha=0.6, color='orange', label=u'clean samples')
        # plt.bar(x, num_noise.numpy(), bottom=num_clean.numpy() ,alpha=1.0, color='green', label=u'noise samples')
        # plt.bar(x, num_noise.numpy(), align='center', bottom=num_clean.numpy() ,alpha=1.0, color='#A9D18E', label=u'noise samples')
        # plt.ylim([0,500])
        plt.ylabel("mean loss",fontsize=15)
        plt.xlabel("class index",fontsize=15)

        # # 绘制散点
        plt.plot(x, loss_clean.numpy(), c="orange", label="clean samples", ls='-', alpha=0.6, lw=2, zorder=2)
        plt.plot(x, loss_noise.numpy(), c="#A9D18E", label="noise samples", ls='-', alpha=1.0, lw=2, zorder=2)
        plt.ylim([0, 8])

        plt.legend(fontsize="large", loc=2)
        # plt.savefig("./result_log/"+str(epoch) + "a.png")
        # plt.ylabel("class num", fontsize=15)
        # plt.xlabel("class index", fontsize=15)
        # plt.legend(fontsize="large", loc=1)
        #
        plt.savefig("./result_log/"+str(epoch) + "b.png")
        plt.cla()
        # #
        # # ax4 = axes.twinx()
        # # ax4.plot(x, l.numpy(), c="red", label=None, ls='--', alpha=0.4, lw=2, zorder=2)
        # # ax4.set_ylim([-5, 5])
        # # 设置图像标题
        #
        # # 显示图像

