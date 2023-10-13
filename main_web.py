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
from loss import *
from utils.builder import *
from model.MLPHeader import MLPHead
from util import *
from utils.eval import *
from data.imbalance_cifar import *
from torch.cuda.amp import autocast as autocast
from torch.utils.data import Dataset, DataLoader
import os

#解决报错：OSError: image file is truncated (5 bytes not processed)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')


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
        # prob = self.proba_head(x)
        return {'logits': logits, 'prob': logits}
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
def warmup(net, optimizer, trainloader, dev, train_loss_meter,train_accuracy_meter, aum_calculator,config):
    # net = torch.compile(net)
    net.train()
    aum_calculator.switch_threshold_examples(set())
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='training')
    for it, sample in enumerate(pbar):
        if type(sample) is dict:
            indices = sample['index']
            x, _ = sample['data']
            x = x.cuda()
            y = sample['label'].cuda()
        else:
            data, y, indices, _ = sample
            x, _ = data
            x, y = x.cuda(), y.cuda()
        with autocast():
            outputs = net(x)
            logits = outputs['logits'] if type(outputs) is dict else outputs
            loss_ce = F.cross_entropy(logits, y,label_smoothing=config.label_smoothing)
            penalty = conf_penalty(logits)
            loss = loss_ce + penalty

        with torch.no_grad():
            max_logits = torch.max(logits, dim=-1)
            mask = logits != max_logits.values[:, None]
            partial = logits - mask * max_logits.values[:, None]
            second_largest = torch.max(mask * logits, dim=-1)
            second_largest = ~mask * second_largest.values[:, None]
            margins = partial - second_largest
            aum_calculator.update_aums(indices.cpu().numpy(), margins.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')
def eval_train(config, net, num_classes, num_samples, trainloader, clean_rate, dev, targets_list=None, criterion='jsdiv'):
    net.eval()
    unclean_criterions = torch.zeros(num_samples)
    if targets_list is None:
        targets_list = torch.zeros(num_samples)
        init_target_list = True
    else:
        init_target_list = False

    with torch.no_grad():
        pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='eval training')

        for it, sample in enumerate(pbar):
            if type(sample) is dict:
                index = sample['index'].cuda()
                inputs, _ = sample['data']
                y = sample['label']
            else:
                data, y, index, clean_or_not = sample
                inputs, _ = data
                index = index.cuda()
            if init_target_list:
                targets_list[index] = y.float()
            inputs, y = inputs.cuda(), y.cuda()
            with autocast():
                outputs = net(inputs)
                logits = outputs['logits'] if type(outputs) is dict else outputs
                if criterion == 'loss':
                    loss = F.cross_entropy(logits, y, reduction='none',label_smoothing=config.label_smoothing)
                elif criterion == 'jsdiv':
                    given_labels = get_smoothed_label_distribution(y, num_classes, epsilon=1e-12)
                    probs = logits.softmax(dim=1)
                    loss = js_div(probs, given_labels)
                else:
                    raise AssertionError('Not Supported!')

            unclean_criterions[index] = loss.cpu()

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
    return prob_clean, targets_list
def robust_train(net, optimizer, trainloader, dev, train_loss_meter,train_accuracy_meter, num_class, ep, p_clean, loss_weight, corr_label_dist, ema_label_dist, params, aum_calculator):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='training')
    num_threshold_examples =len(trainloader.dataset) // 100
    threshold_data_ids = random.sample(list(range(len(trainloader.dataset))), num_threshold_examples)
    aum_calculator.switch_threshold_examples(threshold_data_ids)
    for it, sample in enumerate(pbar):
        if type(sample) is dict:
            indices = sample['index']
            x, x_s = sample['data']
            x, x_s = x.cuda(), x_s.cuda()
            y = sample['label'].cuda()
        else:
            data, y, indices, _ = sample
            x, x_s = data
            x, x_s = x.cuda(), x_s.cuda()
            y = y.cuda()
        batch_size = x.size(0)

        labels_ = torch.zeros(batch_size, num_class).cuda().scatter_(1, y.view(-1, 1), 1)
        weights_ = torch.FloatTensor(p_clean[indices]).view(-1, 1).cuda() if isinstance(p_clean, np.ndarray) else \
            p_clean[indices].view(-1, 1)

        with torch.no_grad():
            with autocast():
                outputs = net(x)
                logits = outputs['logits'] if type(outputs) is dict else outputs
                px = logits.softmax(dim=1)
                pred_net = F.one_hot(px.max(dim=1)[1], num_class).float()
                high_conf_cond = (labels_ * px).sum(dim=1) > params.tau
                weights_[high_conf_cond] = 1

                if params.useEMA:
                    label_ema = ema_label_dist[indices, :].clone().cuda()
                    aph = params.aph
                    label_ema = label_ema * aph + px * (1 - aph)
                    pseudo_label_l = labels_ * weights_ + label_ema * (1 - weights_)
                else:
                    pseudo_label_l = labels_ * weights_ + pred_net * (1 - weights_)
                idx_chosen = torch.where(weights_ == 1)[0]

        train_acc = accuracy(logits, y, topk=(1,))
        outputs2 = net(x_s)
        logits2 = outputs2['logits'] if type(outputs2) is dict else outputs2

        with torch.no_grad():
            if epoch > params.start_expand:
                expected_ratio = params.bs_threshold
                px2 = logits2.softmax(dim=1)
                score1 = px.max(dim=1)[0]
                score2 = px2.max(dim=1)[0]
                match = px.max(dim=1)[1] == px2.max(dim=1)[1]
                hc2_sel_wx1 = high_conf_sel2(idx_chosen, weights_, batch_size, score1, score2, match,
                                             params.tau_expand, expected_ratio)
                idx_chosen = torch.where(hc2_sel_wx1 == 1)[0]

        l = np.random.beta(4, 4)
        l = max(l, 1 - l)
        with autocast():
            if params.use_mixup and idx_chosen.size(0) > 0:
                idx2 = idx_chosen[torch.randperm(idx_chosen.size(0))]

                pseudo_label_mix = pseudo_label_l[idx_chosen]
                x_mix = copy.deepcopy(x[idx_chosen])
                confidence = px.max(dim=1)[0]
                res = (confidence[idx_chosen] >= confidence[idx2]).float()
                idx_L = torch.where(res == 1)
                idx_S = torch.where((res == 0))

                x_mix[idx_L] = l * x[idx_chosen[idx_L]] + (1 - l) * x[idx2[idx_L]]
                x_mix[idx_S] = (1 - l) * x[idx_chosen[idx_S]] + l * x[idx2[idx_S]]
                pseudo_label_mix[idx_L] = l * pseudo_label_l[idx_chosen[idx_L]] + (1 - l) * pseudo_label_l[idx2[idx_L]]
                pseudo_label_mix[idx_S] = (1 - l) * pseudo_label_l[idx_chosen[idx_S]] + l * pseudo_label_l[idx2[idx_S]]

                outputs_mix = net(x_mix)
                logits_mix = outputs_mix['logits'] if type(outputs_mix) is dict else outputs_mix
                loss_mix = F.cross_entropy(logits_mix, pseudo_label_mix,label_smoothing=params.label_smoothing)

            else:
                outputs_mix = net(x[idx_chosen])
                logits_mix = outputs_mix['logits'] if type(outputs_mix) is dict else outputs_mix
                loss_mix = F.cross_entropy(logits_mix, pseudo_label_l[idx_chosen], label_smoothing=params.label_smoothing)

            if params.use_cons:
                loss_cr = F.cross_entropy(logits2, pseudo_label_l, reduction="none",label_smoothing=params.label_smoothing)
                with torch.no_grad():
                    logits=pseudo_label_l
                    max_logits = torch.max(logits, dim=-1)
                    mask = logits != max_logits.values[:, None]
                    partial = logits - mask * max_logits.values[:, None]
                    second_largest = torch.max(mask * logits, dim=-1)
                    second_largest = ~mask * second_largest.values[:, None]
                    margins = partial - second_largest
                    aum_calculator.update_aums(indices.cpu().numpy(), margins.cpu().numpy())

                    crt_aums = aum_calculator.get_aums(indices.cpu().numpy())
                    crt_aums = crt_aums[np.arange(logits.shape[0]), max_logits.indices.cpu()]
                    crt_aums = torch.tensor(crt_aums).cuda()
                    aum_mask = crt_aums.ge(torch.quantile(crt_aums, params.CM)).float()
                loss_cr = torch.mean(loss_cr*aum_mask)
            else:
                loss_cr = torch.tensor(0).float()

            loss = loss_mix + loss_cr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
def build_model(num_classes, params_init, dev, config):
    net = ResNet(arch="resnet50", num_classes=num_classes, pretrained=True)
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
    if dataset.startswith('web-'):
        class_ = {"web-aircraft": 100, "web-bird": 200, "web-car": 196}
        num_classes = class_[dataset]
        transform = build_transform(rescale_size=448, crop_size=448)
        dataset_ = build_webfg_dataset(os.path.join('Datasets', dataset), CLDataTransform(transform['train'], transform["train_strong_aug"]),  transform['test'])
        trainloader = DataLoader(dataset_["train"], batch_size=params.batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True)
        test_loader = DataLoader(dataset_['test'], batch_size=params.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=False)

    num_samples = len(trainloader.dataset)
    return_dict = {'trainloader': trainloader, 'num_classes': num_classes, 'num_samples': num_samples, 'dataset': dataset}
    return_dict['test_loader'] = test_loader


    return return_dict
def init_corrected_labels(num_samples, num_classes, trainloader, soft=True):
    corr_label_dist = torch.zeros((num_samples, num_classes)).float()
    with torch.no_grad():
        for sample in trainloader:
            if type(sample) is dict:
                indices = sample['index']
                # x, x_s = sample['data']
                # x, x_s = x.cuda(), x_s.cuda()
                y = sample['label']
            else:
                # x, y, _ = sample
                # x, y,_,_ = sample
                # x, y = x.cuda(), y.cuda()
                _, y, indices, _ = sample
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
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lr-decay', type=str, default='cosine:10,1e-4,65')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--warmup-lr', type=float, default=0.005)
    parser.add_argument('--warmup-gradual', action='store_true')
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--params-init', type=str, default='none')
    parser.add_argument('--criterion', type=str, default='loss')
    parser.add_argument('--aug-type', type=str, default='auto')
    parser.add_argument('--use-cons', type=bool, default=True)
    parser.add_argument('--use-mixup', type=bool, default=True)
    parser.add_argument('--rho-range', type=str, default='0.2:0.2:60', help='Format: stop:start:step')
    parser.add_argument('--omega-range', type=str, default='1.0:1.0:60', help='Format: stop:start:step')
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--tau-expand', type=float, default=0.8)
    parser.add_argument('--start-expand', type=int, default=55)
    parser.add_argument('--bs-threshold', type=float, default=0.9)
    parser.add_argument('--useEMA', type=bool, default=True)
    parser.add_argument('--aph', type=float, default=0.55)
    parser.add_argument('--rescale-size', type=int, default=32)
    parser.add_argument('--crop-size', type=int, default=32)
    parser.add_argument('--synthetic-data', type=str, default='cifar100nc')
    parser.add_argument('--noise-type', type=str, default='symmetric')
    parser.add_argument('--closeset-ratio', type=float, default=0.2)
    parser.add_argument('--database', type=str, default='./dataset')
    parser.add_argument('--overlay-ratio', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='web-aircraft')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--imbalance', type=bool, default=False)
    parser.add_argument('--data_percent', default=1, type=float, help='data number percent')
    parser.add_argument('--imb-factor', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--save-weights', type=bool, default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--restart-epoch', type=int, default=0)
    parser.add_argument('--delta', default=0.997, type=float,
                        help='EMA smoothing for marigin calculation')
    parser.add_argument('--CM', default=0.2, type=float)
    parser.add_argument('--label-smoothing', default=0.2, type=float)
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
    model = build_model(n_classes, config.params_init, device, config)

    if config.resume!=None:
        path = config.resume
        dict_s = torch.load(path, map_location='cpu')
        model.load_state_dict(dict_s)
        model.cuda()

    # create optimizer & lr_plan or lr_scheduler
    optim = build_optimizer(model, config)
    lr_plan = build_lr_plan(config.lr, config.epochs, config.warmup_epochs, config.warmup_lr, decay=config.lr_decay,
                            warmup_gradual=config.warmup_gradual)

    corrected_label_distributions = init_corrected_labels(n_samples, n_classes, loader_dict['trainloader'], soft=False)
    if config.useEMA:
        ema_label_distributions = copy.deepcopy(corrected_label_distributions)
    else:
        ema_label_distributions = None
    targets_all = None
    best_accuracy, best_epoch = 0.0, None
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    aum_calculator = AUMCalculator(config.delta, int(
        n_classes), n_samples, 95 / 100)

    compiled_model = torch.compile(model)

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
            warmup(compiled_model, optim, input_loader, device, train_loss_meter,train_accuracy_meter, aum_calculator,config)
        else:
            rho = rho_begin - (rho_begin - rho_final) * linear_rampup(epoch - config.warmup_epochs, T_rho)
            approx_clean_probs, targets_all = eval_train(config,compiled_model, n_classes, n_samples, input_loader, rho, device,
                                                         targets_all, criterion=config.criterion)  # np.array, (num_samples, )
            omega = omega_begin - (omega_begin - omega_final) * linear_rampup(epoch - config.warmup_epochs, T_omega)
            robust_train(compiled_model, optim, input_loader, device, train_loss_meter,train_accuracy_meter, n_classes, epoch, approx_clean_probs, omega,
                         corrected_label_distributions, ema_label_distributions, config,aum_calculator)

        eval_result = evaluate_cls_acc(loader_dict['test_loader'], compiled_model, device)
        test_accuracy = eval_result['accuracy']

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            if config.save_weights:
                torch.save(model.state_dict(), config.dataset+'best_model.pth')
        print(
            f'>> Epoch {epoch}: loss {train_loss_meter.avg:.2f} ,train acc {train_accuracy_meter.avg:.2f} ,test acc {test_accuracy:.2f}, best acc {best_accuracy:.2f}')
        epoch+=1
