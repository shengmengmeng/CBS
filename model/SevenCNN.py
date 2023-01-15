import torch
import torch.nn as nn
from util import MLPHead

class CLDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w1 = self.transform_weak(sample)
        x_w2 = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w1, x_w2, x_s

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1, activation='tanh'):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(196, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classfier_head = MLPHead(256, mlp_scale_factor=2, projection_size=n_outputs)
        self.proba_head = torch.nn.Sequential(
            MLPHead(256, mlp_scale_factor=1, projection_size=3, init_method='He', activation=activation),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        logits = self.classfier_head(x)
        prob = self.proba_head(x)
        return {'logits': logits, 'prob': prob}

