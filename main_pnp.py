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
from torch.cuda.amp import autocast, GradScaler
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class UHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels, init_method='He', lpn_head=None, npn_head=None, mlp_activation='relu', mlp_use_bn=True):
        super().__init__()
        if isinstance(lpn_head, str):
            if lpn_head.startswith('mlp'):
                mlp_factor = float(lpn_head.split('-')[1])
                self.label_cls_head = MLPHead(in_channels, mlp_factor, out_channels, init_method, mlp_activation, mlp_use_bn)
            elif lpn_head.startswith('linear'):
                raise AssertionError('lpn head is set to be linear, please specify linear layer explicitly!')
            else:
                raise AssertionError(f'{lpn_head} classifier head is not supported!')
        else:
            assert isinstance(lpn_head, nn.Module), f'type of lpn_head is {type(lpn_head)}!'
            self.label_cls_head = lpn_head

        if npn_head is None:
            self.use_npn_head = False
        else:
            self.use_npn_head = True
            assert isinstance(npn_head, dict), 'npn_head should be either None or a dict!'
            self.noise_cls_head = MLPHead(in_channels, npn_head['mlp_factor'], npn_head['num_types'], init_method, mlp_activation, mlp_use_bn)

    def forward(self, x):
        return_dict = {'logits': self.label_cls_head(x), 'feature_embeddings': x}
        if self.use_npn_head: return_dict['noise_probs'] = self.noise_cls_head(x)
        return return_dict
class ResNet(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True, activation='tanh', classifier='linear'):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.classfier_head = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.classfier_head, init_method='He')
        elif classifier.startswith('mlp'):
            sf = float(classifier.split('-')[1])
            self.classfier_head = MLPHead(self.feat_dim, mlp_scale_factor=sf, projection_size=num_classes, init_method='He', activation='relu')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')
        self.proba_head = torch.nn.Sequential(
            MLPHead(self.feat_dim, mlp_scale_factor=1, projection_size=3, init_method='He', activation=activation),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        x = self.neck(x).view(N, -1)
        logits = self.classfier_head(x)
        prob = self.proba_head(x)
        return {'logits': logits, 'prob': prob}



class UnifiedNet(nn.Module):
    def __init__(self, task_index_major, task_index_minor, num_classes, classifier_type='original', params_init='none'):
        super().__init__()

        self.task_index_major = task_index_major
        self.task_index_minor = task_index_minor
        self.num_classes = num_classes
        self.classifier_type = classifier_type
        self.params_init = params_init

        self.dataset_is_twitter = (task_index_major in ['1', '2'] and task_index_minor in ['7', '8'])
        self.dataset_is_sst = (task_index_major in ['1', '2'] and task_index_minor in ['9', '10'])
        self.dataset_is_webvision = (task_index_major == '4')
        self.dataset_is_task5_img = (task_index_major == '5' and (int(task_index_minor) < 89 or int(task_index_minor)>99))
        self.dataset_is_task5_txt = (task_index_major == '5' and (int(task_index_minor) >= 89 and int(task_index_minor)<=99))
        self.dataset_is_task6 = (task_index_major == '6')
        self.dataset_is_task7 = (task_index_major == '7' )#pnp
        self.dataset_is_task8 = (task_index_major == '8')#数据不平衡
        self.dataset_is_task9 = (task_index_major == '9')  # 基于PES
        self.dataset_is_task10 = (task_index_major == '10')  # 基于CurveNet

        # load required backbone network
        if task_index_major in ['1', '2'] and task_index_minor in ['1', '2', '5', '6']:
            print(f'>>> Task{task_index_major}_{task_index_minor} - Using ResNet18 backbone!')
            self.net = ResNet18(num_classes=num_classes)
        elif task_index_major in ['2'] and task_index_minor in ['3', '4']:
            print(f'>>> Task{task_index_major}_{task_index_minor} - Using ResNet18 backbone!')
            self.net = ResNet18(num_classes=num_classes)
        elif task_index_major in ['1'] and task_index_minor in ['3', '4']:
            print(f'>>> Task{task_index_major}_{task_index_minor} - Using ResNet34 backbone!')
            self.net = ResNet34(num_classes=num_classes)
        elif task_index_major == '3':
            print(f'>>> Task{task_index_major}_{task_index_minor} - Using ResNet18 backbone!')
            self.net = ResNet18(num_classes=num_classes)
        elif self.dataset_is_webvision:
            print(f'>>> Task{task_index_major}_{task_index_minor} - Using ResNet50 backbone!')
            self.net = ResNet50(num_classes=num_classes)
        elif self.dataset_is_twitter:
            print(f'>>> Task{task_index_major}_{task_index_minor} - Using ThreeLayerNet backbone!')
            self.net = ThreeLayerNet(num_classes)
        elif self.dataset_is_sst:
            print(f'>>> Task{task_index_major}_{task_index_minor} - Using WordAveragingLinear backbone!')
            self.net = WordAveragingLinear(num_classes)
        elif self.dataset_is_task5_img:
            if (int(task_index_minor)<89) or (task_index_minor in ['103','104']):
                print(f'>>> Task{task_index_major}_{task_index_minor} - Using ResNet18 backbone!')
                self.net = ResNet18(num_classes=num_classes)
            elif task_index_minor in ['100','101','102','105']:
                print(f'>>> Task{task_index_major}_{task_index_minor} - Using ResNet34 backbone!')
                self.net = ResNet34(num_classes=num_classes)
            elif task_index_minor =='106':
                print(f'>>> Task{task_index_major}_{task_index_minor} - Using ResNet50 backbone!')
                self.net = ResNet50(num_classes=num_classes)
            else:
                raise AssertionError('Task{task_index_major}_{task_index_minor} is out of scope!')

        elif self.dataset_is_task5_txt:
            if task_index_minor in['90','93','94','96']:
                print(f'>>> Task{task_index_major}_{task_index_minor} - Using Classifier backbone!')
                self.net = Classifier(num_classes)
            elif task_index_minor in['91','92','95']:
                print(f'>>> Task{task_index_major}_{task_index_minor} - Using Model backbone!')
                self.net = Model()
            elif task_index_minor in['89','97','98','99']:
                print(f'>>> Task{task_index_major}_{task_index_minor} - Using my_lin backbone!')
                self.net = my_lin(num_classes=num_classes)
        elif self.dataset_is_task6:
            print(f'>>> Task{task_index_major}_{task_index_minor} - Using InceptionResNetV2 backbone!')
            self.net = InceptionResNetV2(num_classes=num_classes)
        elif self.dataset_is_task7:
            print(f'>>> Task{task_index_major}_{task_index_minor} - Using SevenCNN backbone!')
            self.net = PreActResNet18(num_classes=num_classes)#其中activation可更改
        elif self.dataset_is_task8:
            print(f'>>> Task{task_index_major}_{task_index_minor} - Using PreResNet18 backbone!')
            self.net =  PreActResNet18(num_classes=num_classes)  # 其中activation可更改
        elif self.dataset_is_task9:
            print(f'>>> Task{task_index_major}_{task_index_minor} - Using PreResNet18 backbone!')
            self.net = PreActResNet18(num_classes=num_classes)  # 其中activation可更改
        elif self.dataset_is_task10:
            if num_classes in [10,100]:
                print(f'>>> Task{task_index_major}_{task_index_minor} - Using ResNet32 backbone!')
                self.net = resnet32(num_classes=num_classes)
            else:
                print(f'>>> Task{task_index_major}_{task_index_minor} - Using ResNet50 backbone!')
                self.net = ResNet(arch="resnet50", num_classes=n_classes, pretrained=False)
                # self.net = resnet50(num_classes=num_classes)

        else:
            raise AssertionError('Task{task_index_major}_{task_index_minor} is out of scope!')
        init_weights(self.net, init_method=params_init)

        if classifier_type != 'original':
            self.update_cls_head()

    def update_cls_head(self):
        # update fc linear if needed
        if self.dataset_is_twitter:
            in_channels = self.net.main[0].in_features
            self.net = UHead(in_channels, self.num_classes, self.params_init, lpn_head=self.net)
        elif self.dataset_is_sst:
            in_channels = self.net.out.in_features
            lpn_head = self.net.out if self.classifier_type in ['original', 'linear'] else self.classifier_type
            self.net.out = UHead(in_channels, self.num_classes, self.params_init, lpn_head=lpn_head)
        elif self.dataset_is_webvision:
            in_channels = self.net.fc.in_features
            lpn_head = self.net.fc if self.classifier_type in ['original', 'linear'] else self.classifier_type
            self.net.fc = UHead(in_channels, self.num_classes, self.params_init, lpn_head=lpn_head)
        else:
            in_channels = self.net.linear.in_features
            lpn_head = self.net.linear if self.classifier_type in ['original', 'linear'] else self.classifier_type
            self.net.linear = UHead(in_channels, self.num_classes, self.params_init, lpn_head=lpn_head)

    def forward(self, x):
        return self.net(x)


class textTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_name):
        assert data_name in ['sst', 'twitter']
        self.data_name = data_name

        train_label_all, train_label, val_data_all, val_label, test_data_all = dataloader.prepare_data()
        self.train_label_all, self.train_label = train_label_all, train_label
        self.val_data_all, self.val_label = val_data_all, val_label
        self.test_data_all = test_data_all

        if data_name == 'sst':
            self.n_classes = 2
            self.n_train_samples = train_label.shape[0]
            self.n_test_samples = test_data_all.shape[0]
            self.n_val_samples = val_label.shape[0]
        else:
            self.n_classes = 10
            self.n_train_samples = len(train_label_all)
            self.n_test_samples = len(test_data_all)
            self.n_val_samples = len(val_data_all)

        if data_name == 'twitter':
            from helper_functions_twitter import embeddings_to_dict, word_list_to_embedding
            embedding_dimension = 50
            embeddings = embeddings_to_dict('embeddings-twitter.txt')
            self.to_embeds = lambda x: word_list_to_embedding(x, embeddings, embedding_dimension)


    def __getitem__(self, index):
        if self.data_name == 'sst':
            txt = self.train_label_all[index]
            label = self.train_label[index]
            return torch.from_numpy(txt), label, index
        else:
            txt = self.to_embeds(self.train_label_all[index:index+1])
            label = self.train_label[index]
            return torch.from_numpy(txt).squeeze(), label, index

    def __len__(self):
        return self.n_train_samples


class text_dataloader:
    def __init__(self, data_name, batch_size, num_workers):
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.all_dataset = textTrainDataset(data_name)

    def run(self, mode):
        if mode == 'train':
            trainloader = torch.utils.data.DataLoader(dataset=self.all_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
            return trainloader
        elif mode == 'test':
            test_dataset = self.all_dataset.test_data_all
            return test_dataset
        elif mode == 'trainset':
            return self.all_dataset.train_label_all, self.all_dataset.train_label


def freeze_model_parts(net, part, task_index_major='1', re_init=False):
    if part == 1:
        for name, child_model in net.named_children():
            freeze_list = ['conv1', 'layer1', 'layer2', 'layer3']
            reinit_list = ['layer4', 'linear'] if task_index_major != '4' else ['layer4', 'fc']
            if name in freeze_list:
                freeze_layer(child_model)
            if name in reinit_list and re_init:
                init_weights(child_model, init_method='He')
    elif part == 2:
        for name, child_model in net.named_children():
            freeze_list = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
            reinit_list = ['linear'] if task_index_major != '4' else ['fc']
            if name in freeze_list:
                freeze_layer(child_model)
            if name in reinit_list and re_init:
                init_weights(child_model, init_method='He')
    else:
        raise AssertionError(f'part can only be 0, 1 or 2')


# define inference function
def get_label(net, test_loader, dev):
    net.eval()
    end_pre = torch.zeros(len(test_loader.dataset))
    n = 0
    with torch.no_grad():
        for _, (inputs) in enumerate(test_loader):
            # print('ii')
            inputs = inputs.cuda()
            outputs = net(inputs)
            if type(outputs) is dict:
                outputs = outputs['logits']
            outputs = torch.argmax(outputs, -1)
            for b in range(inputs.size(0)):
                end_pre[n] = outputs[b]
                n += 1
    return end_pre


def get_label_txt(net, test_data, dev, to_embeds=None):
    net.eval()
    with torch.no_grad():
        if to_embeds is None:
            data = torch.from_numpy(test_data).cuda()
        else:
            data = torch.from_numpy(to_embeds(test_data)).cuda()
        outputs = net(data)
        if type(outputs) is dict:
            outputs = outputs['logits']
        end_pre = outputs.data.max(1)[1].type(torch.LongTensor)
    return end_pre


def get_relabel(net, trainloader, dev):
    end_relabel = torch.zeros(len(trainloader.dataset))
    with torch.no_grad():
        pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='get relabeling')
        for it, sample in enumerate(pbar):
            x, y, indices = sample
            x = x.cuda()
            outputs = net(x)
            if type(outputs) is dict:
                outputs = outputs['logits']
            pseudo_labels = torch.argmax(outputs, -1)
            for b in range(x.size(0)):
                end_relabel[indices[b]] = pseudo_labels[b]
    end_relabel = end_relabel.type(torch.LongTensor)
    return np.array(end_relabel.cpu())


def warmup(scaler, net, optimizer, trainloader, dev, train_loss_meter,train_accuracy_meter, no_penalty=False, is_sst=False):
    if isinstance(trainloader, dict):
        warmup_txt(scaler, net, optimizer, trainloader, dev,train_loss_meter,train_accuracy_meter, no_penalty, is_sst)
    else:
        warmup_img(scaler, net, optimizer, trainloader, dev,train_loss_meter,train_accuracy_meter, no_penalty)


def warmup_img(scaler, net, optimizer, trainloader, dev,train_loss_meter,train_accuracy_meter, no_penalty=False):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='training')
    for it, sample in enumerate(pbar):

        x, y, indices = sample
        x, y = x.cuda(), y.cuda()
        # with autocast():
        if True:
            outputs = net(x)
            logits = outputs['logits'] if type(outputs) is dict else outputs
            loss_ce = F.cross_entropy(logits, y)
            if no_penalty:
                loss = loss_ce
            else:
                penalty = conf_penalty(logits)
                loss = loss_ce + penalty
        # scaler.scale(loss).backward()
        # try:
        #     scaler.step(optimizer)
        # except RuntimeError:  # in case of "RuntimeError: Function 'CudnnBatchNormBackward' returned nan values in its 0th output."
        #     print('Runtime Error occured! Have unscaled losses and clipped grads before optimizing!')
        #     scaler.unscale_(optimizer)
        #     torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2, norm_type=2.0)
        #     scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')


def warmup_txt(scaler, net, optimizer, trainloader, dev,train_loss_meter,train_accuracy_meter, no_penalty=False, is_sst=False):
    net.train()

    train_label_all, train_label, to_embeds, batch_size = trainloader['train_label_all'], trainloader['train_label'], trainloader['to_embeds'], trainloader['batch_size']
    num_examples = train_label.shape[0]  if is_sst else len(train_label_all)
    num_batches = num_examples // batch_size
    indices_all = np.arange(num_examples)
    np.random.shuffle(indices_all)
    for i in range(num_batches):
        optimizer.zero_grad()

        offset = i * batch_size
        x_batch = train_label_all[indices_all[offset:offset + batch_size]] if is_sst else to_embeds(train_label_all[indices_all[offset:offset + batch_size]])
        y_batch = train_label[indices_all[offset:offset + batch_size]]
        x, y = torch.from_numpy(x_batch).cuda(), torch.from_numpy(y_batch).cuda()
        outputs = net(x)
        logits = outputs['logits'] if type(outputs) is dict else outputs
        loss_ce = F.cross_entropy(logits, y)
        if no_penalty:
            loss = loss_ce
        else:
            penalty = conf_penalty(logits)
            loss = loss_ce + penalty
        loss.backward()
        optimizer.step()
        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))


def eval_train(net, num_classes, num_samples, trainloader, clean_rate, dev, targets_list=None, criterion='jsdiv'):
    net.eval()
    unclean_criterions = torch.zeros(num_samples)
    if targets_list is None:
        targets_list = torch.zeros(num_samples)
        init_target_list = True
    else:
        init_target_list = False
    is_text_task = isinstance(trainloader, dict)
    logits_ = torch.zeros(num_samples, num_classes).cuda()
    with torch.no_grad():
        if not is_text_task:
            for it, sample in enumerate(trainloader):
                inputs, y, index = sample
                index = index.cuda()
                if init_target_list:
                    targets_list[index] = y.float()
                inputs, y = inputs.cuda(), y.cuda()
                with autocast():
                    outputs = net(inputs)
                    logits = outputs['logits'] if type(outputs) is dict else outputs
                    if criterion == 'loss':
                        loss = F.cross_entropy(logits, y, reduction='none')
                    elif criterion == 'jsdiv':
                        given_labels = get_smoothed_label_distribution(y, num_classes, epsilon=1e-12)
                        probs = logits.softmax(dim=1)
                        loss = js_div(probs, given_labels)
                    else:
                        raise AssertionError('Not Supported!')

                    unclean_criterions[index] = loss.cpu()
                    # logits_[index] = logits.float()
        else:
            train_label_all, train_label, to_embeds = trainloader['train_label_all'], trainloader['train_label'], trainloader['to_embeds']
            inputs = torch.from_numpy(to_embeds(train_label_all))
            y = torch.from_numpy(train_label)
            if init_target_list:
                targets_list = y.float()
            # inputs, y = inputs.cuda(), y.cuda()
            with autocast():
                outputs = net.cpu()(inputs)
                net = net.cuda()
                logits = outputs['logits'] if type(outputs) is dict else outputs
                if criterion == 'loss':
                    loss = F.cross_entropy(logits, y, reduction='none')
                elif criterion == 'jsdiv':
                    given_labels = get_smoothed_label_distribution(y, num_classes, epsilon=1e-12)
                    probs = logits.softmax(dim=1)
                    loss = js_div(probs, given_labels)
                else:
                    raise AssertionError('Not Supported!')

                unclean_criterions = loss.cpu()
                # logits_ = logits.cuda()

    unclean_criterions = (unclean_criterions - unclean_criterions.min()) / (unclean_criterions.max() - unclean_criterions.min())

    prob_clean = np.zeros(num_samples)
    idx_chosen_sm = []
    bs_j = num_samples * (1.0 / num_classes)
    for j in range(num_classes):
        indices = np.where(targets_list.numpy() == j)[0]
        if len(indices) == 0:
            continue
        pseudo_unclean_criterion_vec_j = unclean_criterions[indices]
        sorted_idx_j = pseudo_unclean_criterion_vec_j.sort()[1].numpy()  # sort ascending order
        partition_j = max(min(int(math.ceil(bs_j*clean_rate)), len(indices)), 1)  # at least one sample
        idx_chosen_sm.append(indices[sorted_idx_j[:partition_j]])
    idx_chosen_sm = np.concatenate(idx_chosen_sm)
    prob_clean[idx_chosen_sm] = 1
    return prob_clean, targets_list, None


def robust_train(scaler, net, optimizer, trainloader, dev, train_loss_meter,train_accuracy_meter, aug, fmix, num_class, ep, p_clean, logits_, loss_weight, corr_label_dist, ema_label_dist, params, is_sst=False):
    if isinstance(trainloader, dict):
        robust_train_txt(scaler, net, optimizer, trainloader, dev, train_loss_meter,train_accuracy_meter, aug, fmix, num_class, ep, p_clean, logits_, loss_weight, corr_label_dist, ema_label_dist, params, is_sst)
    else:
        robust_train_img(scaler, net, optimizer, trainloader, dev, train_loss_meter,train_accuracy_meter, aug, fmix, num_class, ep, p_clean, logits_, loss_weight, corr_label_dist, ema_label_dist, params)


def robust_train_img(scaler, net, optimizer, trainloader, dev, train_loss_meter,train_accuracy_meter, aug, fmix, num_class, ep, p_clean, logits_, loss_weight, corr_label_dist, ema_label_dist, params):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='training')
    for it, sample in enumerate(pbar):

        x, y, indices = sample
        batch_size = x.size(0)
        x, y = x.cuda(), y.cuda()
        # with autocast():
        if True:
            labels_x = torch.zeros(batch_size, num_class).cuda().scatter_(1, y.view(-1, 1), 1)
            w_x = torch.FloatTensor(p_clean[indices]).view(-1, 1).cuda() if isinstance(p_clean, np.ndarray) else p_clean[indices].view(-1, 1)

            with torch.no_grad():
                outputs = net(x)
                logits = outputs['logits'] if type(outputs) is dict else outputs
                # logits=logits_[indices]
                px = logits.softmax(dim=1)
                pred_net = F.one_hot(px.max(dim=1)[1], num_class).float()
                high_conf_cond = (labels_x * px).sum(dim=1) > params.tau
                w_x[high_conf_cond] = 1
                if params.useEMA:
                    label_ema = ema_label_dist[indices, :].clone().cuda()
                    aph = params.aph # 0.1 - (0.1 - params.aph) * linear_rampup(ep, params.start_expand)
                    label_ema = label_ema * aph + px * (1 - aph)
                    pseudo_label_l = labels_x * w_x + label_ema * (1 - w_x)
                else:
                    pseudo_label_l = labels_x * w_x + pred_net * (1 - w_x)
                idx_chosen = torch.where(w_x == 1)[0]   # idx_chosen includes clean samples only
                n_clean = idx_chosen.size(0)

                idx_noises = torch.where(w_x != 1)[0]
                overlay_num = min(int(params.overlay_ratio * batch_size), idx_chosen.size(0))
                idx_noises = torch.cat((idx_noises, idx_chosen[torch.randperm(idx_chosen.size(0))[:overlay_num]]), dim=0)

                if idx_noises.size(0)>0:
                    x2 = aug(x, mode='s').cuda()
                    outputs2 = net(x2)
                    logits2 = outputs2['logits'] if type(outputs2) is dict else outputs2

                with torch.no_grad():
                    if epoch > params.start_expand:
                        expected_ratio = params.bs_threshold
                        px2 = torch.zeros_like(px) - 0.1
                        px2[idx_noises] = logits2.softmax(dim=1)
                        score1 = px.max(dim=1)[0]
                        score2 = px2.max(dim=1)[0]
                        match = px.max(dim=1)[1] == px2.max(dim=1)[1]
                        hc2_sel_wx1 = high_conf_sel2(idx_chosen, w_x, batch_size, score1, score2, match, params.tau_expand, expected_ratio)
                        idx_chosen = torch.where(hc2_sel_wx1 == 1)[0]    # idx_chosen includes clean & ID noisy samples
                    n_semi = idx_chosen.size(0) - n_clean

                ni = max(int((1 + params.overlay_ratio) * batch_size) - idx_chosen.size(0), 0)
                idx_for_idx_noises = torch.randperm(idx_noises.size(0))[:ni]

            # Loss Over Chosen Clean Samples & ID Noisy Samples
            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            if params.use_mixup and idx_chosen.size(0)>0:
                idx2 = idx_chosen[torch.randperm(idx_chosen.size(0))]

                x_mix = l * x[idx_chosen] + (1 - l) * x[idx2]
                # if is_sst: x_mix = x_mix.long()
                pseudo_label_mix = l * pseudo_label_l[idx_chosen] + (1 - l) * pseudo_label_l[idx2]
                outputs_mix = net(x_mix)
                logits_mix = outputs_mix['logits'] if type(outputs_mix) is dict else outputs_mix
                loss_mix = F.cross_entropy(logits_mix, pseudo_label_mix)

                if params.CABC:
                    pseudo_label_mix = copy.deepcopy(pseudo_label_l[idx_chosen])
                    x_mix = copy.deepcopy(x[idx_chosen])
                    confidence = px.max(dim=1)[0]
                    for item in range(idx_chosen.size(0)):
                        l = np.random.beta(4, 4)
                        l = max(l, 1 - l)
                        if confidence[idx_chosen[item]] > confidence[idx2[item]]:
                            x_mix[item] = l * x[idx_chosen[item]] + (1 - l) * x[idx2[item]]
                            pseudo_label_mix[item] = l * pseudo_label_l[idx_chosen[item]] + (1 - l) * pseudo_label_l[
                                idx2[item]]
                        else:
                            x_mix[item] = (1 - l) * x[idx_chosen[item]] + l * x[idx2[item]]
                            pseudo_label_mix[item] = (1 - l) * pseudo_label_l[idx_chosen[item]] + l * pseudo_label_l[
                                idx2[item]]

                    outputs_mix = net(x_mix)
                    logits_mix = outputs_mix['logits'] if type(outputs_mix) is dict else outputs_mix
                    loss_mix = loss_mix * config.alpha +(1-config.alpha) * F.cross_entropy(logits_mix, pseudo_label_mix)
            else:
                outputs_mix = net(x[idx_chosen])
                logits_mix = outputs_mix['logits'] if type(outputs_mix) is dict else outputs_mix
                loss_mix = F.cross_entropy(logits_mix, pseudo_label_l[idx_chosen])
                # loss_mix = torch.tensor(0).float()

            # Employ fmix_ratio to meet the GPU usage requirement when `use-fmix` is enabled
            if params.use_fmix:
                n_fmix_samples = int(idx_chosen.size(0) * params.fmix_ratio)
                i_fmix_samples = torch.randperm(idx_chosen.size(0))[:n_fmix_samples]
                x_fmix = fmix(x[idx_chosen[i_fmix_samples]])
                # if is_sst: x_fmix = x_fmix.long()
                outputs_fmix = net(x_fmix)
                logits_fmix = outputs_fmix['logits'] if type(outputs_fmix) is dict else outputs_fmix
                loss_fmix = fmix.loss(logits_fmix, (pseudo_label_l[idx_chosen[i_fmix_samples]].detach()).long())
            else:
                loss_fmix = torch.tensor(0).float()

            # if params.use_cons:
            loss_cr = F.cross_entropy(logits2, pseudo_label_l)
            # else:
            #     # loss_cr = torch.tensor(0).float()
            #     if params.use_mixup:
            #         loss_cr = torch.tensor(0).float()
            #     else:
            #         loss_cr = F.cross_entropy(logits2, pseudo_label_l[idx_noises])

            loss = loss_mix+ loss_cr
        # scaler.scale(loss).backward()
        # try:
        #     scaler.step(optimizer)
        # except RuntimeError:  # in case of "RuntimeError: Function 'CudnnBatchNormBackward' returned nan values in its 0th output."
        #     scaler.unscale_(optimizer)
        #     torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2, norm_type=2.0)
        #     scaler.step(optimizer)
        # scaler.update()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')

        corr_label_dist[indices] = pseudo_label_l.detach().clone().cpu().data
        if params.useEMA:
            ema_label_dist[indices] = label_ema.detach().clone().cpu().data
            del label_ema


def robust_train_txt(scaler, net, optimizer, trainloader, dev, aug, fmix, num_class, ep, p_clean, logits_, loss_weight, corr_label_dist, ema_label_dist, params, is_sst=False):
    net.train()

    train_label_all, train_label, to_embeds, batch_size = trainloader['train_label_all'], trainloader['train_label'], trainloader['to_embeds'], trainloader['batch_size']
    num_examples = train_label.shape[0] if is_sst else len(train_label_all)
    num_batches = num_examples // batch_size
    indices_all = np.arange(num_examples)
    np.random.shuffle(indices_all)
    for i in range(num_batches):
        optimizer.zero_grad()

        offset = i * batch_size
        x_batch = train_label_all[indices_all[offset:offset + batch_size]] if is_sst else to_embeds(train_label_all[indices_all[offset:offset + batch_size]])
        y_batch = train_label[indices_all[offset:offset + batch_size]]
        x, y = torch.from_numpy(x_batch).cuda(), torch.from_numpy(y_batch).cuda()
        indices = torch.from_numpy(indices_all[offset:offset+batch_size])
        batch_size = x.size(0)

        labels_x = torch.zeros(batch_size, num_class).cuda().scatter_(1, y.view(-1, 1), 1)
        w_x = torch.FloatTensor(p_clean[indices]).view(-1, 1).cuda() if isinstance(p_clean, np.ndarray) else p_clean[indices].view(-1, 1)

        with torch.no_grad():
            # outputs = net(x)
            # logits = outputs['logits'] if type(outputs) is dict else outputs
            logits=logits_[indices]
            px = logits.softmax(dim=1)
            pred_net = F.one_hot(px.max(dim=1)[1], num_class).float()
            high_conf_cond = (labels_x * px).sum(dim=1) > params.tau
            w_x[high_conf_cond] = 1
            if params.useEMA:
                label_ema = ema_label_dist[indices, :].clone().cuda()
                aph = params.aph # 0.1 - (0.1 - params.aph) * linear_rampup(ep, params.start_expand)
                label_ema = label_ema * aph + px * (1 - aph)
                pseudo_label_l = labels_x * w_x + label_ema * (1 - w_x)
            else:
                pseudo_label_l = labels_x * w_x + pred_net * (1 - w_x)
            idx_chosen = torch.where(w_x == 1)[0]   # idx_chosen includes clean samples only
            n_clean = idx_chosen.size(0)

            idx_noises = torch.where(w_x != 1)[0]
            overlay_num = min(int(params.overlay_ratio * batch_size), idx_chosen.size(0))
            idx_noises = torch.cat((idx_noises, idx_chosen[torch.randperm(idx_chosen.size(0))[:overlay_num]]), dim=0)

        if idx_noises.size(0)>0:
            x2 = aug(x[idx_noises], mode='s').cuda()
            outputs2 = net(x2)
            logits2 = outputs2['logits'] if type(outputs2) is dict else outputs2

        with torch.no_grad():
            if epoch > params.start_expand:
                expected_ratio = params.bs_threshold
                px2 = torch.zeros_like(px) - 0.1
                px2[idx_noises] = logits2.softmax(dim=1)
                score1 = px.max(dim=1)[0]
                score2 = px2.max(dim=1)[0]
                match = px.max(dim=1)[1] == px2.max(dim=1)[1]
                hc2_sel_wx1 = high_conf_sel2(idx_chosen, w_x, batch_size, score1, score2, match, params.tau_expand, expected_ratio)
                idx_chosen = torch.where(hc2_sel_wx1 == 1)[0]    # idx_chosen includes clean & ID noisy samples
            n_semi = idx_chosen.size(0) - n_clean

        ni = max(int((1 + params.overlay_ratio) * batch_size) - idx_chosen.size(0), 0)
        idx_for_idx_noises = torch.randperm(idx_noises.size(0))[:ni]

        # Loss Over Chosen Clean Samples & ID Noisy Samples
        l = np.random.beta(4, 4)
        l = max(l, 1 - l)
        if params.use_mixup and idx_chosen.size(0)>0:
            with torch.no_grad():
                idx2 = idx_chosen[torch.randperm(idx_chosen.size(0))]
                x_mix = l * x[idx_chosen] + (1 - l) * x[idx2]
                if is_sst: x_mix = x_mix.long()
                pseudo_label_mix = l * pseudo_label_l[idx_chosen] + (1 - l) * pseudo_label_l[idx2]
            outputs_mix = net(x_mix)
            logits_mix = outputs_mix['logits'] if type(outputs_mix) is dict else outputs_mix
            loss_mix = F.cross_entropy(logits_mix, pseudo_label_mix)
        else:
            loss_mix = torch.tensor(0).float()

        # Employ fmix_ratio to meet the GPU usage requirement when `use-fmix` is enabled
        if params.use_fmix:
            n_fmix_samples = int(idx_chosen.size(0) * params.fmix_ratio)
            i_fmix_samples = torch.randperm(idx_chosen.size(0))[:n_fmix_samples]
            x_fmix = fmix(x[idx_chosen[i_fmix_samples]])
            if is_sst: x_fmix = x_fmix.long()
            outputs_fmix = net(x_fmix)
            logits_fmix = outputs_fmix['logits'] if type(outputs_fmix) is dict else outputs_fmix
            loss_fmix = fmix.loss(logits_fmix, (pseudo_label_l[idx_chosen[i_fmix_samples]].detach()).long())
        else:
            loss_fmix = torch.tensor(0).float()

        if params.use_cons and idx_noises.size(0)>0:
            loss_cr = F.cross_entropy(logits2[idx_for_idx_noises], pseudo_label_l[idx_noises[idx_for_idx_noises]])
        else:
            if params.use_mixup:
                loss_cr = torch.tensor(0).float()
            else:
                loss_cr = F.cross_entropy(logits2, pseudo_label_l[idx_noises])

        loss = loss_mix + loss_cr * loss_weight
        loss.backward()
        optimizer.step()
        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))

        corr_label_dist[indices] = pseudo_label_l.detach().clone().cpu().data
        if params.useEMA:
            ema_label_dist[indices] = label_ema.detach().clone().cpu().data
            del label_ema


def high_conf_sel2(idx_chosen, w_x, batch_size, score1, score2, match, tau, expected_r):
    w_x2 = w_x.clone()
    if (1. * idx_chosen.shape[0] / batch_size) < expected_r:
        # when clean data is insufficient, try to incorporate more examples
        high_conf_cond2 = (score1 > tau) * (score2 > tau) * match
        # both nets agrees
        high_conf_cond2 = (1. * high_conf_cond2 - w_x.squeeze()) > 0
        # remove already selected examples; newly selected
        hc2_idx = torch.where(high_conf_cond2)[0]

        max_to_sel_num = int(batch_size * expected_r) - idx_chosen.shape[0]
        # maximally select batch_size * expected_r; idx_chosen.shape[0] select already
        if high_conf_cond2.sum() > max_to_sel_num:
            # to many examples selected, remove some low conf examples
            score_mean = (score1 + score2) / 2
            idx_remove = (-score_mean[hc2_idx]).sort()[1][max_to_sel_num:]
            # take top scores
            high_conf_cond2[hc2_idx[idx_remove]] = False
        w_x2[high_conf_cond2] = 1
    return w_x2


def build_model(task_index_major, task_index_minor, num_classes, params_init, dev):
    net = UnifiedNet(task_index_major, task_index_minor, num_classes, classifier_type='original', params_init=params_init)
    net = net.cuda()
    return net


def build_optimizer(net, params):
    if params.opt == 'adam':
        return build_adam_optimizer(net.parameters(), params.lr, params.weight_decay, amsgrad=False)
    elif params.opt == 'sgd':
        return build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet!')


def build_loader(task_index_major, task_index_minor, params):
    DATASET_IS_CIFAR10 = (task_index_major in ['1', '2'] and task_index_minor in ['1', '2']) or (
                task_index_major == '3' and task_index_minor in ['1', '2'])
    DATASET_IS_CIFAR100 = (task_index_major in ['1', '2'] and task_index_minor in ['3', '4']) or (
                task_index_major == '3' and task_index_minor not in ['1', '2'])
    DATASET_IS_TINY_IMAGENET = (task_index_major == '1' and task_index_minor in ['5', '6'])
    DATASET_IS_HYBRID = (task_index_major == '2' and task_index_minor in ['5', '6'])
    DATASET_IS_TWITTER = (task_index_major in ['1', '2'] and task_index_minor in ['7', '8'])
    DATASET_IS_SST = (task_index_major in ['1', '2'] and task_index_minor in ['9', '10'])
    DATASET_IS_WEBVISION = (task_index_major == '4')
    DATASET_IS_TASK5 = (task_index_major == '5' )
    DATASET_IS_TASK6 = (task_index_major == '6')
    DATASET_IS_TASK7 = (task_index_major == '7')#适用于PNP实验基础的，含有sym，asym，以及分布外噪声
    DATASET_IS_TASK8 = (task_index_major == '8')  # 适用于PNP实验基础的，含有sym，asym，以及分布外噪声
    DATASET_IS_TASK9 = (task_index_major == '9')  # 适用于PES实验基础的，含有sym，pair ,instance
    DATASET_IS_TASK10 = (task_index_major == '10') # 基于CoverNet

    if DATASET_IS_CIFAR10:
        dataset = 'cifar10'
        num_classes = 10
        loader = dataloader.cifar_dataloader(batch_size=params.batch_size, num_workers=params.num_workers)
        trainloader = loader.run('warmup')
        test_loader = loader.run('test')
    elif DATASET_IS_CIFAR100:
        dataset = 'cifar100'
        num_classes = 100
        loader = dataloader.cifar_dataloader(batch_size=params.batch_size, num_workers=params.num_workers)
        trainloader = loader.run('warmup')
        test_loader = loader.run('test')
    elif DATASET_IS_TINY_IMAGENET:
        dataset = 'tiny_imagenet'
        num_classes = 200
        loader = dataloader.tinyimagenet_dataloader(batch_size=params.batch_size, num_workers=params.num_workers)
        trainloader = loader.run('warmup')
        test_loader = loader.run('test')
    elif DATASET_IS_HYBRID:
        dataset = 'mix_data'
        num_classes = 24
        loader = dataloader.cifar_dataloader(batch_size=params.batch_size, num_workers=params.num_workers)
        trainloader = loader.run('warmup')
        test_loader = loader.run('test')
    elif DATASET_IS_WEBVISION:
        dataset = 'webvision'
        num_classes = 50 if task_index_minor in [1, 3] else 100
        loader = dataloader.webvision_dataloader(batch_size=params.batch_size, num_workers=params.num_workers)
        web_valloader = loader.run('test')
        imagenet_valloader = loader.run('imagenet')
        trainloader = loader.run('warmup')
    elif DATASET_IS_SST:
        dataset = 'sst'
        num_classes = 2
        loader = text_dataloader(data_name=dataset, batch_size=params.batch_size, num_workers=params.num_workers)
        trainloader = loader.run('train')
        test_loader = loader.run('test')
        trainset = loader.run('trainset')
    elif DATASET_IS_TWITTER:
        dataset = 'twitter'
        num_classes = 10
        loader = text_dataloader(data_name=dataset, batch_size=params.batch_size, num_workers=params.num_workers)
        trainloader = loader.run('train')
        test_loader = loader.run('test')
        trainset = loader.run('trainset')
        to_embeds = loader.all_dataset.to_embeds
    elif DATASET_IS_TASK5:
        dataset = 'cifar10'
        loader = dataloader.cifar_dataloader()
        trainloader = loader.run('train')
        num_classes = trainloader.dataset.num_class
        if int(task_index_minor) >=89 and int(task_index_minor) <=99:
            dataset = 'cross_model_data'
        test_loader = None
    elif DATASET_IS_TASK6:
        dataset = 'webvision'
        num_classes = 1000
        transform_train = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_dataset = webvision_dataset(root_dir='./data/', transform=transform_train, mode="train",
                                          num_class=num_classes)
        trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params.batch_size,
                                                  shuffle=True, num_workers=params.num_workers,
                                                  pin_memory=True, sampler=None)
        test_loader=None
    elif DATASET_IS_TASK7:
        dataset = 'cifar100'
        num_classes = 100
        transform = build_transform(rescale_size=params.rescale_size, crop_size=params.crop_size)
        if dataset == 'cifar100':
            # trainset = build_cifar100n_dataset(os.path.join(params.database, dataset),
            #                                   CLDataTransform(transform['cifar_train'],
            #                                                   transform['cifar_train_strong_aug']),
            #                                   transform['cifar_test'], noise_type=params.noise_type,
            #                                   openset_ratio=params.openset_ratio, closeset_ratio=params.closeset_ratio)
            trainset = build_cifar100n_dataset(os.path.join(params.database, dataset),transform['cifar_train'],
                                               transform['cifar_test'], noise_type=params.noise_type,
                                               openset_ratio=params.openset_ratio, closeset_ratio=params.closeset_ratio)#选择弱数据增强
        else:
            raise AssertionError(f'{dataset} dataset is not supported yet.')

        # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=params.batch_size,
        #                                           shuffle=True, num_workers=params.num_workers,
        #                                           pin_memory=True, sampler=None)
        trainloader = DataLoader(trainset['train'], batch_size=params.batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True)
        test_loader = DataLoader(trainset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)
        # test_loader = None
    elif DATASET_IS_TASK8:
        dataset = 'cifar100'
        num_classes = 100
        transform = build_transform(rescale_size=params.rescale_size, crop_size=params.crop_size)
        if dataset == 'cifar100':
            # trainset = build_cifar100n_dataset(os.path.join(params.database, dataset),
            #                                   CLDataTransform(transform['cifar_train'],
            #                                                   transform['cifar_train_strong_aug']),
            #                                   transform['cifar_test'], noise_type=params.noise_type,
            #                                   openset_ratio=params.openset_ratio, closeset_ratio=params.closeset_ratio)
            trainset = build_cifar100n_class_imbalanced_dataset(os.path.join(params.database, dataset),transform['cifar_train'],
                                               transform['cifar_test'], noise_type=params.noise_type,
                                               openset_ratio=params.openset_ratio, closeset_ratio=params.closeset_ratio)#选择弱数据增强
        else:
            raise AssertionError(f'{dataset} dataset is not supported yet.')

        # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=params.batch_size,
        #                                           shuffle=True, num_workers=params.num_workers,
        #                                           pin_memory=True, sampler=None)
        trainloader = DataLoader(trainset['train'], batch_size=params.batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True)
        test_loader = DataLoader(trainset['test'], batch_size=params.batch_size*2, shuffle=False, num_workers=8, pin_memory=False)
        # test_loader = None
    elif DATASET_IS_TASK9:
        dataset = params.dataset
        if dataset == 'cifar10' :
            num_classes = 10
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            train_set = CIFAR10(root=os.path.join(params.database, dataset), train=True, download=True)
            test_set = CIFAR10(root=os.path.join(params.database, dataset), train=False, transform=transform_test, download=True)
        elif dataset == 'cifar100':
            num_classes = 100
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            transform_test = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
            train_set = CIFAR100(root=os.path.join(params.database, dataset), train=True, download=True)
            test_set = CIFAR100(root=os.path.join(params.database, dataset), train=False, transform=transform_test, download=True)

        if params.noise_type == "symmetric":
            noise_include = True
        else:
            noise_include = False

        data=train_set.data
        targets=train_set.targets
        if params.imbalance:
            print("train_data before:" + str(len(data)))
            data_list_val = {}
            for j in range(100):
                data_list_val[j] = [i for i, label in enumerate(targets) if label == j]

            idx_to_train = []
            class_choise = []
            while (len(class_choise) < 50):
                c = int(random.uniform(0, 99))
                if c not in class_choise:
                    class_choise.append(c)
            print(class_choise)
            for cls_idx, img_id_list in data_list_val.items():
                if int(cls_idx) in class_choise:
                    # np.random.shuffle(img_id_list)
                    img_num = int(len(img_id_list) / 5)
                    idx_to_train.extend(img_id_list[:img_num])
                else:
                    np.random.shuffle(img_id_list)
                    idx_to_train.extend(img_id_list[:])

            # self.train_idx = idx_to_train
            data = data[idx_to_train]
            targets = list(np.array(targets)[idx_to_train])
            print("train_data after:" + str(len(data)))


        data, _, noisy_labels, _, clean_labels, _ = dataset_split(data, np.array(targets),
                                                                  params.closeset_ratio, params.noise_type, params.data_percent,
                                                                  params.seed, num_classes, noise_include)
        train_dataset = Train_Dataset(data, noisy_labels, transform_train)


        trainloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=8,
                                 pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=params.batch_size * 2, shuffle=False, num_workers=8, pin_memory=False)
        # test_loader = None
    elif DATASET_IS_TASK10:
        dataset = params.dataset
        if dataset=="cifar100":
            num_classes = 100
            train_dataset = CIFAR100_im(root="./data/cifar100", train=True, meta=False, num_meta=0,
                                        corruption_prob=params.closeset_ratio, corruption_type='unif', transform="hard",
                                        target_transform=None, download=True, seed=params.seed, imblance=params.imbalance,
                                        imb_factor=params.imb_factor)

            test_set = CIFAR100_im(root="./data/cifar100", train=False, meta=False, num_meta=0,
                                   corruption_prob=params.closeset_ratio, corruption_type='unif', transform="easy",
                                   target_transform=None, download=True, seed=params.seed, imblance=params.imbalance,
                                   imb_factor=params.imb_factor)

            trainloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=8,
                                     pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=False, num_workers=8,
                                     pin_memory=False)

        if dataset=="cifar10":
            num_classes = 10
            train_dataset = CIFAR10_im(root="./data/cifar10", train=True, meta=False, num_meta=0,
                                        corruption_prob=params.closeset_ratio, corruption_type='unif', transform="hard",
                                        target_transform=None, download=True, seed=params.seed, imblance=params.imbalance,
                                        imb_factor=params.imb_factor)

            test_set = CIFAR10_im(root="./data/cifar10", train=False, meta=False, num_meta=0,
                                   corruption_prob=params.closeset_ratio, corruption_type='unif', transform="easy",
                                   target_transform=None, download=True, seed=params.seed, imblance=params.imbalance,
                                   imb_factor=params.imb_factor)
            trainloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=8,
                                     pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=False, num_workers=8,
                                     pin_memory=False)

        if dataset == "food-101n":
            num_classes = 101
            root="/data/Food-101N_release/"
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

            trainloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=8,
                                     pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=False, num_workers=8, pin_memory=False)

        if dataset == "Clothing1M":
            num_classes = 14
            root="/data/Clothing1M/"
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ])
            train_dataset = clothing_dataset(root, transform=train_transform, mode='all')
            test_set = clothing_dataset(root, transform=test_transform, mode='test')

            trainloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=8,
                                     pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=False, num_workers=8, pin_memory=False)
        # test_loader = None
        # test_loader = None

    else:
        raise AssertionError('Not implemented yet!')

    num_samples = len(trainloader.dataset)
    return_dict = {'trainloader': trainloader, 'num_classes': num_classes, 'num_samples': num_samples, 'dataset': dataset}
    if DATASET_IS_WEBVISION:
        return_dict['web_valloader'] = web_valloader
        return_dict['imagenet_valloader'] = imagenet_valloader
    else:
        return_dict['test_loader'] = test_loader
    if DATASET_IS_TWITTER:
        return_dict['to_embeds'] = to_embeds
    elif DATASET_IS_SST:
        return_dict['to_embeds'] = lambda x: x
    if DATASET_IS_TWITTER or DATASET_IS_SST:
        return_dict['trainset'] = trainset

    return return_dict


def get_img_size(dataset):
    if dataset in ['cifar10', 'cifar100','cross_model_data']:
        return (32, 32)
    elif dataset in ['tiny_imagenet', 'mix_data','food-101n','Clothing1M']:
        return (56, 56)
    elif dataset == 'webvision':
        return (227, 227)
    elif dataset == 'sst':
        return (30,)
    elif dataset == 'twitter':
        return (150,)
    else:
        raise AssertionError(f'ReAug for {dataset} is not implemented yet!')


def init_corrected_labels(num_samples, num_classes, trainloader, soft=True):
    corr_label_dist = torch.zeros((num_samples, num_classes)).float()
    with torch.no_grad():
        for sample in trainloader:
            _, y, indices = sample
            y_dist = F.one_hot(y, num_classes).float()
            if soft: y_dist = F.softmax(y_dist*10, dim=1)
            assert y_dist.device.type == 'cpu'
            corr_label_dist[indices] = y_dist
    return corr_label_dist


def use_PES(params, net, optimizer, ep, lrs, task_index_major):
    if params.use_pes:
        pes_start_epoch = int(params.warmup_epochs * (1 - params.pes_rate))
        if ep == pes_start_epoch:
            freeze_model_parts(net, 2, task_index_major, re_init=params.pes_reinit)
        elif ep == params.warmup_epochs:
            unfreeze_layer(net)
        if pes_start_epoch <= ep < params.warmup_epochs:
            adjust_lr(optimizer, lrs[ep] * 0.01)


# parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args', type=str, default=None)
    parser.add_argument('--task', type=str, default='7-0')
    parser.add_argument('--logger-root', type=str, default='./result/')
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--CABC', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=str, default='cosine:20,5e-4,100')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--warmup-lr', type=float, default=0.001)
    parser.add_argument('--warmup-gradual', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--params-init', type=str, default='none')

    parser.add_argument('--criterion', type=str, default='loss')
    parser.add_argument('--aug-type', type=str, default='auto')
    parser.add_argument('--use-cons', type=bool, default=False)
    parser.add_argument('--use-mixup', type=bool, default=False)
    parser.add_argument('--use-fmix', type=bool, default=False)
    parser.add_argument('--fmix-ratio', type=float, default=1.0)
    parser.add_argument('--rho-range', type=str, default='0.2:0.2:60', help='Format: stop:start:step')
    parser.add_argument('--omega-range', type=str, default='1.0:1.0:60', help='Format: stop:start:step')
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--tau-expand', type=float, default=0.8)
    parser.add_argument('--start-expand', type=int, default=80)
    parser.add_argument('--bs-threshold', type=float, default=0.9)

    parser.add_argument('--use-pes', type=bool, default=False)
    parser.add_argument('--pes-rate', type=float, default=0.1)
    parser.add_argument('--pes-reinit', type=bool, default=False)

    parser.add_argument('--useEMA', type=bool, default=True)
    parser.add_argument('--aph', type=float, default=0.55)
    parser.add_argument('--overlay-ratio', type=float, default=0.15)
    parser.add_argument('--no-penalty', type=bool, default=False)

    #基于PNP的对比实验增加参数
    parser.add_argument('--rescale-size', type=int, default=32)
    parser.add_argument('--crop-size', type=int, default=32)
    parser.add_argument('--synthetic-data', type=str, default='cifar100nc')
    parser.add_argument('--noise-type', type=str, default='symmetric')
    parser.add_argument('--closeset-ratio', type=float, default=0.2)
    parser.add_argument('--database', type=str, default='./dataset')

    #基于PES的对比实验增加参数
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model', type=str, default='resnet32')
    parser.add_argument('--imbalance', type=bool, default=False)
    parser.add_argument('--data_percent', default=1, type=float, help='data number percent')

    #基于Curvenet的参数
    parser.add_argument('--imb-factor', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--save-weights', type=bool, default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--restart-epoch', type=int, default=0)

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    config = parse_args()
    init_seeds(config.seed)
    device = set_device(config.gpu)
    task_id, sub_task_id = config.task.split('-')
    #适配PNP
    config.openset_ratio = 0.0 if config.synthetic_data == 'cifar100nc' else 0.2

    rho_range_items = [float(ele) for ele in config.rho_range.split(':')]
    rho_begin, rho_final = rho_range_items[0], rho_range_items[1]
    T_rho = 1 if len(rho_range_items) == 2 else int(rho_range_items[2])
    omega_range_items = [float(ele) for ele in config.omega_range.split(':')]
    omega_begin, omega_final = omega_range_items[0], omega_range_items[1]
    T_omega = 1 if len(omega_range_items) == 2 else int(omega_range_items[2])

    # create dataloader
    loader_dict = build_loader(task_id, sub_task_id, config)
    dataset_name, n_classes, n_samples = loader_dict['dataset'], loader_dict['num_classes'], loader_dict['num_samples']

    # create model
    model = build_model(task_id, sub_task_id, n_classes, config.params_init, device)

    if config.resume!=None:
        path = config.resume
        dict_s = torch.load(path, map_location='cpu')
        # net_dict = model.state_dict()
        # idy = 0
        # for k, v in dict.items():
        #     k = k.replace('module.', '')
        #     if k in net_dict:
        #         net_dict[k] = v
        #         idy += 1
        # print(len(net_dict), idy, 'update state dict already')
        model.load_state_dict(dict_s)
        model.cuda()



    # create optimizer & lr_plan or lr_scheduler
    optim = build_optimizer(model, config)
    lr_plan = build_lr_plan(config.lr, config.epochs, config.warmup_epochs, config.warmup_lr, decay=config.lr_decay,
                            warmup_gradual=config.warmup_gradual)

    # init re-augmenter
    re_aug = ReAug(strong_aug=config.aug_type, dataset=dataset_name)

    # Training
    corrected_label_distributions = init_corrected_labels(n_samples, n_classes, loader_dict['trainloader'], soft=False)
    if config.useEMA:
        ema_label_distributions = copy.deepcopy(corrected_label_distributions)
    else:
        ema_label_distributions = None
    targets_all = None
    img_size = get_img_size(dataset_name)
    fmix_helper = FMix(alpha=1, size=img_size) if config.use_fmix else None

    start_time = time.time()
    best_accuracy, best_epoch = 0.0, None
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    scaler = GradScaler()

    epoch = 0
    if config.restart_epoch != 0:
        epoch = config.restart_epoch
        config.restart_epoch = 0
    while epoch < config.epochs:
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        adjust_lr(optim, lr_plan[epoch])
        # use_PES(config, model, optim, epoch, lr_plan, task_id)
        torch.cuda.synchronize()
        T1 = time.time()
        if dataset_name in ['sst', 'twitter']:
            input_loader = {'train_label_all': loader_dict['trainset'][0], 'train_label': loader_dict['trainset'][1],
                            'to_embeds': loader_dict['to_embeds'], 'batch_size': config.batch_size}
        else:
            input_loader = loader_dict['trainloader']
        if epoch < config.warmup_epochs:
            warmup(scaler, model, optim, input_loader, device, train_loss_meter,train_accuracy_meter, config.no_penalty, dataset_name=='sst')
        else:
            rho = rho_begin - (rho_begin - rho_final) * linear_rampup(epoch - config.warmup_epochs, T_rho)
            approx_clean_probs, targets_all,logits_  = eval_train(model, n_classes, n_samples, input_loader, rho, device,
                                                         targets_all, criterion=config.criterion)  # np.array, (num_samples, )
            omega = omega_begin - (omega_begin - omega_final) * linear_rampup(epoch - config.warmup_epochs, T_omega)
            robust_train(scaler, model, optim, input_loader, device, train_loss_meter,train_accuracy_meter, re_aug, fmix_helper, n_classes, epoch, approx_clean_probs, logits_, omega,
                         corrected_label_distributions, ema_label_distributions, config, dataset_name=='sst')

        runtime = time.time() - T1
        gpu_max_mem = torch.cuda.max_memory_allocated()
        # print(
        #     f'>> Epoch {epoch}: loss {train_loss_meter.avg:.2f} , acc {train_accuracy_meter.avg:.2f} , {runtime:.2f} sec, {gpu_max_mem :.2f} b ({gpu_max_mem / (2 ** 10):.2f} Kb)')

        #测试集上输出结果：
        if dataset_name in ['sst', 'twitter']:
            eval_result = evaluate_txt_cls_acc(loader_dict['local_val_data'], model, device, to_embeds=loader_dict['to_embeds'])
        else:
            eval_result = evaluate_cls_acc(loader_dict['test_loader'], model, device)
        test_accuracy = eval_result['accuracy']

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            if config.save_weights:
                torch.save(model.state_dict(), f'./result_log/'+dataset_name+"_"+str(epoch)+'_best_model.pth')
        print(
            f'>> Epoch {epoch}: loss {train_loss_meter.avg:.2f} ,train acc {train_accuracy_meter.avg:.2f} ,test acc {test_accuracy:.2f}, best acc {best_accuracy:.2f}')
        epoch+=1
    # end_time = time.time()
    # elapse = end_time - start_time
    # gpu_max_mem = torch.cuda.max_memory_allocated()
    # print(
    #     f'Total time elapse: {elapse:.2f} seconds; Max GPU Memory Allocated: {gpu_max_mem :.2f} b ({gpu_max_mem / (2 ** 10):.2f} Kb)')
    #
    # result_dir = os.path.join(config.logger_root, f'task{task_id}', f'data{sub_task_id}')
    # if not os.path.isdir(result_dir):
    #     os.makedirs(result_dir, exist_ok=True)
    # np.save(f'{result_dir}/model.npy', model.state_dict())
    #
    # train_corr_results = np.array(corrected_label_distributions.argmax(dim=-1).long().cpu())
    # np.save(f'{result_dir}/label_train.npy', train_corr_results)

