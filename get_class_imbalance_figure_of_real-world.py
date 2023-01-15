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

import torchvision.datasets as datasets
from data.imbalance_cifar import *
from data.food101 import *
from data.food101n import *
from data.Clothing1M import *
from torch.cuda.amp import autocast, GradScaler
from matplotlib import pyplot as plt
import numpy as np

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
def food():
    num_classes = 101
    root = "/data/Food-101N_release/"
    rescale_size = 512
    crop_size = 448
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.CenterCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_dataset = Food101N(root, transform=train_transform)
    test_set = Food101(os.path.join(root, 'food-101'), split='test', transform=test_transform)

    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8,
                             pin_memory=True)
    res_class = torch.zeros(101)
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='get the number of each class')
    res = open("food.txt", "w")
    for it, sample in enumerate(pbar):
        x, y, indices = sample
        batch_size = x.size(0)
        # x, y = x.cuda(), y.cuda()
        for i in range(len(y)):
            res_class[y[i]] += 1
    a = res_class.sort(descending=True)[0]
    b = res_class.sort(descending=True)[1]
    for i in range(len(a)):
        res.write(str(int(a[i].data)) + "\n")
    # 解决中文显示问题

    # 生成figure对象
    fig = plt.figure()
    # 生成axes对象
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    x = np.linspace(0, 101, 101)
    y = a.numpy()
    # 绘制散点
    axes.plot(x, y, c="green", label=r'$food-101n$', ls='-.', alpha=0.6, lw=2, zorder=2)
    # 设置图像标题
    axes.legend()
    # 显示图像
    plt.savefig("./food.png")
    return y/min(y)

def Clothing1M():
    root = "/data/Clothing1M/"
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ])
    train_dataset = clothing_dataset(root, transform=train_transform, mode='all')

    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8,
                             pin_memory=True)
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='Clothing1M')
    res_class = torch.zeros(14)
    res = open("Clothing1M.txt", "w")
    for it, sample in enumerate(pbar):
        x, y= sample
        # x, y = x.cuda(), y.cuda()
        for i in range(len(y)):
            res_class[y[i]] += 1
    a = res_class.sort(descending=True)[0]
    for i in range(len(a)):
        res.write(str(int(a[i].data)) + "\n")
    # 解决中文显示问题

    # 生成figure对象
    fig = plt.figure()
    # 生成axes对象
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    x = np.linspace(0, 14, 14)
    y = a.numpy()
    # 绘制散点
    axes.plot(x, y, c="green", label=r'$Clothing1M$', ls='-.', alpha=0.6, lw=2, zorder=2)
    # 设置图像标题
    axes.legend()
    # 显示图像
    plt.savefig("./Clothing1M.png")
    return y/min(y)
def imagenet():
    root = '/data/imagenet'
    def get_imagenet(root, train=True, transform=None, target_transform=None):
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)


    dataset = "/data/imagenet"
    trainset = get_imagenet(root=dataset, train=True)
    res_class=torch.zeros(1000)
    for item in trainset:
        image, target = item
        res_class[target]+=1

    a = res_class.sort(descending=True)[0]
    res = open("imagenet.txt", "w")
    for i in range(len(a)):
        res.write(str(int(a[i].data)) + "\n")
    # 解决中文显示问题

    # 生成figure对象
    fig = plt.figure()
    # 生成axes对象
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    x = np.linspace(0, 1000, 1000)
    y = a.numpy()
    # 绘制散点
    axes.plot(x, y, c="green", label=r'$imagenet$', ls='-.', alpha=0.6, lw=2, zorder=2)
    # 设置图像标题
    axes.legend()
    # 显示图像
    plt.savefig("./ImageNet.png")


if __name__ == '__main__':
    # food_=food()
    # clothing1m=Clothing1M()
    res = open("food.txt", "r")
    lines=res.readlines()
    food_=[]
    clothing1m_=[]
    for line in lines:
        food_.append(int(line))
    food_=np.array(food_)/min(food_)
    res = open("Clothing1M.txt", "r")
    lines = res.readlines()
    for line in lines:
        clothing1m_.append(int(line))
    clothing1m_=np.array(clothing1m_)/min(clothing1m_)



    fig = plt.figure()
    # 生成axes对象
    axes = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    x = np.linspace(0, 100, 101)

    axes.plot(x,food_ , c="green", label="Food-101N", ls='-', alpha=0.7, lw=3, zorder=2)
    axes.set_ylim([0, 10])
    ax3 = axes.twinx()
    x = np.linspace(0, 13, 14)
    ax3.plot(x, clothing1m_, c="orange", label="Clothing1M", ls='-', alpha=0.7, lw=3, zorder=2)
    ax3.set_ylim([0, 10])
    ax3.set_ylabel("mean loss")
    plt.legend(loc=2)
    #
    # ax4 = axes.twinx()
    # ax4.plot(x, l.numpy(), c="red", label=None, ls='--', alpha=0.4, lw=2, zorder=2)
    # ax4.set_ylim([-5, 5])
    # 设置图像标题

    # 显示图像
    plt.savefig("./result_log/real-world.png")
    plt.xlabel("class id")

    # imagenet()