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

color = ["blue", "green", "orange", "red", "black", "yellow", "grey"]
num_classes=100
fig = plt.figure(figsize=(6,5))
x_ = np.linspace(0, num_classes-1, num_classes)
plt.ylim([0,600])
plt.ylabel("the number of samples",fontsize=15)
plt.xlabel("class index",fontsize=15)
k=0
for i in [10, 20, 50, 100, 150,200]:
    num=torch.zeros(100,dtype=torch.int16)
    train_dataset = CIFAR100_im(root="./data/cifar100", train=True, meta=False, num_meta=0,
                                corruption_prob=0, corruption_type='unif', transform="hard",
                                target_transform=None, download=True, seed=123, imblance=True,
                                imb_factor=1/i)

    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8,
                             pin_memory=True)
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='training')
    for it, sample in enumerate(pbar):

        x, y, y_true = sample
        x, y = x.cuda(), y.cuda()
        for j in y:
            num[j]+=1


    # 绘制散点

    plt.plot(x_, num.numpy(), c=color[k], label="imbalance factor "+str(i), ls='-', lw=2, zorder=2)
    k+=1


        # 显示图像
plt.legend(loc=1)
plt.savefig("./result_log/imbalance_cifar100.png")

