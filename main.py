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
from tqdm import tqdm
from utils import *
from utils.builder import *
from model.MLPHeader import MLPHead
from util import *
from utils.eval import *
from model.ResNet32 import resnet32
from data.imbalance_cifar import *
from data.Clothing1M import *
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
    def __init__(self, num_classes, classifier_type='original', params_init='none'):
        super().__init__()

        self.num_classes = num_classes
        self.classifier_type = classifier_type
        self.params_init = params_init

        if num_classes in [10,100]:
            print(f'>>> - Using ResNet32 backbone!')
            self.net = resnet32(num_classes=num_classes)
        else:
            print(f'>>>- Using ResNet50 backbone!')
            self.net = ResNet(arch="resnet50", num_classes=n_classes, pretrained=True)
            # self.net = resnet50(num_classes=num_classes)
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




def warmup(net, optimizer, trainloader, dev, train_loss_meter,train_accuracy_meter):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='training')
    for it, sample in enumerate(pbar):

        x, y, indices = sample
        x, y = x.cuda(), y.cuda()
        outputs = net(x)
        logits = outputs['logits'] if type(outputs) is dict else outputs
        loss_ce = F.cross_entropy(logits, y)
        penalty = conf_penalty(logits)
        loss = loss_ce + penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')


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



def robust_train(net, optimizer, trainloader, dev, train_loss_meter,train_accuracy_meter, aug, num_class, ep, p_clean, logits_, loss_weight, corr_label_dist, ema_label_dist, params):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='training')
    for it, sample in enumerate(pbar):
        x, y, indices = sample
        batch_size = x.size(0)
        x, y = x.cuda(), y.cuda()

        labels_ = torch.zeros(batch_size, num_class).cuda().scatter_(1, y.view(-1, 1), 1)
        weight_ = torch.FloatTensor(p_clean[indices]).view(-1, 1).cuda() if isinstance(p_clean, np.ndarray) else p_clean[indices].view(-1, 1)

        with torch.no_grad():
            outputs = net(x)
            logits = outputs['logits'] if type(outputs) is dict else outputs
            px = logits.softmax(dim=1)
            pred_net = F.one_hot(px.max(dim=1)[1], num_class).float()
            high_conf_cond = (labels_ * px).sum(dim=1) > params.tau
            weight_[high_conf_cond] = 1
            if params.useEMA:
                label_ema = ema_label_dist[indices, :].clone().cuda()
                aph = params.aph
                label_ema = label_ema * aph + px * (1 - aph)
                pseudo_label_l = labels_ * weight_ + label_ema * (1 - weight_)
            else:
                pseudo_label_l = labels_ * weight_ + pred_net * (1 - weight_)
            idx_chosen = torch.where(weight_ == 1)[0]

            idx_noises = torch.where(weight_ != 1)[0]
            idx_noises = torch.cat((idx_noises, idx_chosen[torch.randperm(idx_chosen.size(0))]), dim=0)

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
                    hc2_sel_wx1 = high_conf_sel2(idx_chosen, weight_, batch_size, score1, score2, match, params.tau_expand,
                                                 expected_ratio)
                    idx_chosen = torch.where(hc2_sel_wx1 == 1)[0]

            idx_for_idx_noises = torch.randperm(idx_noises.size(0))

        l = np.random.beta(4, 4)
        l = max(l, 1 - l)
        if params.use_mixup:
            idx2 = idx_chosen[torch.randperm(idx_chosen.size(0))]

            x_mix = l * x[idx_chosen] + (1 - l) * x[idx2]
            pseudo_label_mix = l * pseudo_label_l[idx_chosen] + (1 - l) * pseudo_label_l[idx2]
            outputs_mix = net(x_mix)
            logits_mix = outputs_mix['logits'] if type(outputs_mix) is dict else outputs_mix
            loss_mix = F.cross_entropy(logits_mix, pseudo_label_mix)

            pseudo_label_mix = copy.deepcopy(pseudo_label_l[idx_chosen])
            x_mix = copy.deepcopy(x[idx_chosen])
            confidence = px.max(dim=1)[0]
            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            for item in range(idx_chosen.size(0)):
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
            loss_mix = loss_mix * config.alpha + (1 - config.alpha) * F.cross_entropy(logits_mix, pseudo_label_mix)
        else:
            outputs_mix = net(x[idx_chosen])
            logits_mix = outputs_mix['logits'] if type(outputs_mix) is dict else outputs_mix
            loss_mix = F.cross_entropy(logits_mix, pseudo_label_l[idx_chosen])


        if params.use_cons:
            loss_cr = F.cross_entropy(logits2[idx_for_idx_noises], pseudo_label_l[idx_for_idx_noises])
        else:
            if params.use_mixup:
                loss_cr = torch.tensor(0).float()
            else:
                loss_cr = F.cross_entropy(logits2[idx_for_idx_noises], pseudo_label_l[idx_for_idx_noises])

        loss = loss_mix + loss_cr*loss_weight

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

def high_conf_sel2(idx_chosen, weight_, batch_size, score1, score2, match, tau, expected_r):
    weight_2 = weight_.clone()
    if (1. * idx_chosen.shape[0] / batch_size) < expected_r:
        # when clean data is insufficient, try to incorporate more examples
        high_conf_cond2 = (score1 > tau) * (score2 > tau) * match
        # both nets agrees
        high_conf_cond2 = (1. * high_conf_cond2 - weight_.squeeze()) > 0
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
        weight_2[high_conf_cond2] = 1
    return weight_2


def build_model(num_classes, params_init, dev):
    net = UnifiedNet(num_classes, classifier_type='original', params_init=params_init)
    net = net.cuda()
    return net


def build_optimizer(net, params):
    if params.opt == 'adam':
        return build_adam_optimizer(net.parameters(), params.lr, params.weight_decay, amsgrad=False)
    elif params.opt == 'sgd':
        return build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet!')


def build_loader(params):
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

    num_samples = len(trainloader.dataset)
    return_dict = {'trainloader': trainloader, 'num_classes': num_classes, 'num_samples': num_samples, 'dataset': dataset}
    return_dict['test_loader'] = test_loader

    return return_dict

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



# parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args', type=str, default=None)
    parser.add_argument('--logger-root', type=str, default='./result/')
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch-size', type=int, default=128)
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
    parser.add_argument('--use-cons', type=bool, default=True)
    parser.add_argument('--use-mixup', type=bool, default=True)
    parser.add_argument('--rho-range', type=str, default='0.2:0.2:60', help='Format: stop:start:step')
    parser.add_argument('--omega-range', type=str, default='1.0:1.0:60', help='Format: stop:start:step')
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--tau-expand', type=float, default=0.8)
    parser.add_argument('--start-expand', type=int, default=80)
    parser.add_argument('--bs-threshold', type=float, default=0.9)

    parser.add_argument('--useEMA', type=bool, default=True)
    parser.add_argument('--aph', type=float, default=0.55)

    parser.add_argument('--rescale-size', type=int, default=32)
    parser.add_argument('--crop-size', type=int, default=32)
    parser.add_argument('--synthetic-data', type=str, default='cifar100nc')
    parser.add_argument('--noise-type', type=str, default='symmetric')
    parser.add_argument('--closeset-ratio', type=float, default=0.2)
    parser.add_argument('--database', type=str, default='./dataset')

    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model', type=str, default='resnet32')
    parser.add_argument('--imbalance', type=bool, default=False)
    parser.add_argument('--data_percent', default=1, type=float, help='data number percent')

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

    rho_range_items = [float(ele) for ele in config.rho_range.split(':')]
    rho_begin, rho_final = rho_range_items[0], rho_range_items[1]
    T_rho = 1 if len(rho_range_items) == 2 else int(rho_range_items[2])
    omega_range_items = [float(ele) for ele in config.omega_range.split(':')]
    omega_begin, omega_final = omega_range_items[0], omega_range_items[1]
    T_omega = 1 if len(omega_range_items) == 2 else int(omega_range_items[2])

    # create dataloader
    loader_dict = build_loader(config)
    dataset_name, n_classes, n_samples = loader_dict['dataset'], loader_dict['num_classes'], loader_dict['num_samples']

    # create model
    model = build_model(n_classes, config.params_init, device)

    if config.resume!=None:
        path = config.resume
        dict_s = torch.load(path, map_location='cpu')
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
    best_accuracy, best_epoch = 0.0, None
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()

    epoch = 0
    if config.restart_epoch != 0:
        epoch = config.restart_epoch
        config.restart_epoch = 0
    while epoch < config.epochs:
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        adjust_lr(optim, lr_plan[epoch])
        input_loader = loader_dict['trainloader']
        if epoch < config.warmup_epochs:
            warmup(model, optim, input_loader, device, train_loss_meter,train_accuracy_meter)
        else:
            rho = rho_begin - (rho_begin - rho_final) * linear_rampup(epoch - config.warmup_epochs, T_rho)
            approx_clean_probs, targets_all,logits_  = eval_train(model, n_classes, n_samples, input_loader, rho, device,
                                                         targets_all, criterion=config.criterion)  # np.array, (num_samples, )
            omega = omega_begin - (omega_begin - omega_final) * linear_rampup(epoch - config.warmup_epochs, T_omega)
            robust_train(model, optim, input_loader, device, train_loss_meter,train_accuracy_meter, re_aug,n_classes, epoch, approx_clean_probs, logits_, omega,
                         corrected_label_distributions, ema_label_distributions, config)

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
