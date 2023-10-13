import torch
import numpy as np
from tqdm import tqdm


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count+1e-8)


# evaluate
def accuracy(y_pred, y_actual, topk=(1, ), return_tensor=False):
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res


def evaluate_cls_acc(dataloader, model, dev, topk=(1,)):
    # model = torch.compile(model)
    model.eval()
    test_loss = AverageMeter()
    test_loss.reset()
    test_accuracy = AverageMeter()
    test_accuracy.reset()

    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='evaluating')):
            if type(sample) is dict:
                x = sample['data'].to(dev)
                y = sample['label'].to(dev)
            else:
                # x, y, _ = sample
                x, y,_,_ = sample
                x, y = x.to(dev), y.to(dev)
            output = model(x)
            logits = output['logits'] if type(output) is dict else output
            loss = torch.nn.functional.cross_entropy(logits, y)
            test_loss.update(loss.item(), x.size(0))
            acc = accuracy(logits, y, topk)
            test_accuracy.update(acc[0], x.size(0))
    return {'accuracy': test_accuracy.avg, 'loss': test_loss.avg}
def evaluate_cls_acc_aug(dataloader, model, dev, topk=(1,)):
    model = torch.compile(model)
    model.eval()
    test_loss = AverageMeter()
    test_loss.reset()
    test_accuracy = AverageMeter()
    test_accuracy.reset()

    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='evaluating')):
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
            output = model(x)
            logits_w = output['logits'] if type(output) is dict else output
            output = model(x_s)
            logits_s = output['logits'] if type(output) is dict else output
            logits = (logits_w+logits_s)/2

            loss = torch.nn.functional.cross_entropy(logits, y)
            test_loss.update(loss.item(), x.size(0))
            acc = accuracy(logits, y, topk)
            test_accuracy.update(acc[0], x.size(0))
    return {'accuracy': test_accuracy.avg, 'loss': test_loss.avg}

def evaluate_cls_acc_coteaching(dataloader, model, dev, topk=(1,)):
    # model = torch.compile(model)
    model[0].eval()
    model[1].eval()
    test_loss = AverageMeter()
    test_loss.reset()
    test_accuracy = AverageMeter()
    test_accuracy.reset()
    test_accuracy1 = AverageMeter()
    test_accuracy1.reset()

    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='evaluating')):
            if type(sample) is dict:
                x = sample['data'].to(dev)
                y = sample['label'].to(dev)
            else:
                # x, y, _ = sample
                x, y,_,_ = sample
                x, y = x.to(dev), y.to(dev)
            output0 = model[0](x)
            output1 = model[1](x)
            logits0 = output0['logits'] if type(output0) is dict else output0
            logits1 = output1['logits'] if type(output1) is dict else output1
            loss0 = torch.nn.functional.cross_entropy(logits0, y)
            loss1 = torch.nn.functional.cross_entropy(logits1, y)
            test_loss.update(loss0.item(), x.size(0))
            acc0 = accuracy(logits0, y, topk)
            acc1 = accuracy(logits1, y, topk)
            test_accuracy.update(acc0[0], x.size(0))
            test_accuracy1.update(acc1[0], x.size(0))
    print(
        f'>> test acc {test_accuracy.avg:.2f}, test2 acc {test_accuracy1.avg:.2f}')
    return {'accuracy0': test_accuracy.avg, 'accuracy1': test_accuracy1.avg, 'loss': test_loss.avg}

def evaluate_relabel_pr(given_labels, corrected_labels):
    precision = 0.0
    recall = 0.0
    # TODO: code for evaluation of relabeling (precision, recall)
    return {'relabel-precision': precision, 'relabel-recall': recall}
