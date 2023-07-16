import os
import time
import argparse
import torch
from torch import nn
from torch.backends import cudnn
import dataset as dataset
import numpy as np
import torchvision
import torch.nn.functional as F
from wideresnet import WideResNet
from partial_models.resnet import resnet
from resnet import resnet18, resnet34, resnet50, resnet101
from lenet import LeNet
import logging
import copy

parser = argparse.ArgumentParser(description='Revisiting Consistency Regularization for Deep Partial Label Learning')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--lam', default=1, type=float)

parser.add_argument('--dataset', type=str, choices=['svhn', 'cifar10', 'cifar100', 'fmnist', 'kmnist'],
                    default='svhn')

parser.add_argument('--model', type=str, choices=['resnet', 'resnet18', 'widenet', 'lenet'], default='resnet18')

parser.add_argument('--lr', default=0.1, type=float)

parser.add_argument('--rate', default=0.1, type=float, help='-1 for feature, 0.x for random')

parser.add_argument('--trial', default='1', type=str)

parser.add_argument('--data-dir', default='./data/', type=str)

parser.add_argument('--note', default='', type=str)  # 利用弱监督数据增强图像的分类结果作为 ground truth
parser.add_argument('--tau', default=1, type=float, help='temperature parameter')
parser.add_argument('--log_path', type=str, default='log_cmp')  # log_coteaching_trial, log_ccc
args = parser.parse_args()
num_classes = 100 if args.dataset == 'cifar100' else 10

log_path = '{}{}'.format(args.log_path, args.trial)

if not os.path.exists(log_path): os.makedirs(log_path)
args.filename = 'CMP_PRODEN_DA_{}_{}_p{}_{}_{}'.format(args.model, args.dataset, args.rate, args.note,
                                                    time.strftime("%m%d%H%M%S", time.localtime()))

logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    filename="{}/{}.log".format(log_path, args.filename),
                    level=logging.INFO)

logging.info(args)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def PRODEN_train(train_loader, model, optimizer, epoch, loss_fn, confidence):
    """
        Run one train epoch
    """
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    model.train()

    for i, (x, x_aug1, x_aug2, y, part_y, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # partial label
        part_y = part_y.float().cuda()
        # original samples with pre-processing
        x = x.cuda()
        y_pred = model(x)
        x_aug1 = x_aug1.cuda()
        y_pred_aug1 = model(x_aug1)
        x_aug2 = x_aug2.cuda()
        y_pred_aug2 = model(x_aug2)

        base_confidence = torch.tensor(confidence[index]).float().cuda()
        final_loss, new_label = loss_fn(y_pred, y_pred_aug1, y_pred_aug2, base_confidence)
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        confidence[index, :] = new_label.detach().cpu().numpy()
        losses.update(final_loss.item(), x.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 200 == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    return losses.avg


def validate(valid_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            output, loss = output.float(), loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    logging.info('Test: [{0}/{1}]\t'
                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        i, len(valid_loader), batch_time=batch_time, loss=losses, top1=top1))

    return top1.avg, losses.avg


def PRODEN():
    global args, best_prec1
    # load data
    if args.dataset == "cifar10":
        train_loader, test = dataset.cifar10_dataloaders(args.data_dir, args.rate, flag_aug=True)
        channel = 3
    elif args.dataset == 'svhn':
        train_loader, test = dataset.svhn_dataloaders(args.data_dir + "svhn/", args.rate, flag_aug=True)
        channel = 3
    elif args.dataset == 'cifar100':
        train_loader, test = dataset.cifar100_dataloaders(args.data_dir, args.rate, flag_aug=True)
        channel = 3
    elif args.dataset == 'fmnist':
        train_loader, test = dataset.fmnist_dataloaders(args.data_dir, args.rate)
        channel = 1
    elif args.dataset == 'kmnist':
        train_loader, test = dataset.kmnist_dataloaders(args.data_dir, args.rate)
        channel = 1
    else:
        assert "Unknown dataset"

    # load model
    if args.model == 'widenet':
        model = WideResNet(34, num_classes, widen_factor=10, dropRate=0.0)
    elif args.model == 'resnet':
        model = resnet(depth=32, n_outputs=num_classes)
    elif args.model == 'resnet18':
        model = resnet18(n_outputs=num_classes)
    elif args.model == 'lenet':
        model = LeNet(out_dim=num_classes, in_channel=1, img_sz=28)
    else:
        assert "Unknown model"
    model = model.cuda()

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    loss_fn = partial_loss_da
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=1e-4)
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    cudnn.benchmark = True
    # init confidence
    confidence = copy.deepcopy(train_loader.dataset.partial_labels)
    confidence = confidence / confidence.sum(axis=1)[:, None]

    # Train loop
    best_acc = 0
    bad_counter = 0
    valacc_list = []
    for epoch in range(0, args.epochs):
        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        # training
        trainloss = PRODEN_train(train_loader, model, optimizer, epoch, loss_fn, confidence)

        # evaluate on validation set
        valacc, valloss = validate(test, model, criterion, epoch)
        scheduler.step()

        if epoch >= args.epochs - 10:
            valacc_list.append(valacc)
    logging.info('Avg acc of last 10 epochs: {:.4f}'.format(np.average(np.array(valacc_list))))


def partial_loss_da(output1, output_aug1, output_aug2, target):
    output = F.softmax(output1, dim=1)
    output_aug1 = F.softmax(output_aug1, dim=1)
    output_aug2 = F.softmax(output_aug2, dim=1)
    l = target * torch.log(output + 0.0000001) + target * torch.log(output_aug1 + 0.0000001) + target * torch.log(
        output_aug2 + 0.0000001)
    loss = (-torch.sum(l)) / l.size(0)
    revisedY = target.clone()
    revisedY[revisedY > 0] = 1
    revisedY = revisedY * torch.pow(output, 1 / (2 + 1)) \
                * torch.pow(output_aug1, 1 / (2 + 1)) \
                * torch.pow(output_aug2, 1 / (2 + 1))
    revisedY = revisedY / (revisedY.sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1) + 0.0000001)
    new_target = revisedY
    return loss, new_target


if __name__ == '__main__':
    PRODEN()
