# -*- coding: utf-8 -*-
import shutil
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

import network
from dataset import Dataset

from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params
import pandas as pd
import itertools


# arch_names = list(unet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'


from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="BraTS",
                        help='dataset name')
    parser.add_argument('--input-channels', default=2, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=40, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101')
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument('--source_domain', type=str, default='T1')
    parser.add_argument("--low_dis", type=bool, default=False)
    parser.add_argument("--high_dis", type=bool, default=False)
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument("--local_rank", type=int, default=0)




    args = parser.parse_args()

    return args

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


def train(args, train_loader, source_FE, target_FE, Classifier, PCAM, criterion, contrastive_loss, space_contrastive_loss, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()
    seg_losses = AverageMeter()
    pcam_losses = AverageMeter()
    space_losses =AverageMeter()

    source_FE.train()
    target_FE.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        t1 = input[:, 0, :, :]
        t2 = input[:, 1, :, :]
        target = target.cuda()


        t1 = t1.unsqueeze(1)
        t2 = t2.unsqueeze(1)
        # compute output
        if args.deepsupervision:
            source_features = source_FE(t1)
            target_features = target_FE(t2)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            source_features = source_FE(t1)
            target_features = target_FE(t2)
            source_features_low, source_features_high = source_features['low_level'], source_features['out']
            target_features_low, target_features_high = target_features['low_level'], target_features['out']

            source_features_low = source_features_low.flatten(2).permute(0, 2, 1)
            target_features_low = target_features_low.flatten(2).permute(0, 2, 1)

            target_features_high = target_features_high.flatten(1)
            source_features_high = source_features_high.flatten(1)

            if args.low_dis:
                source_features_low_dis = PCAM(source_features_low)
                target_features_low_dis = PCAM(target_features_low)

                pcam_loss = contrastive_loss(source_features_low_dis, target_features_low_dis)
                space_loss = space_contrastive_loss(source_features_high, target_features_high)
                output_source = Classifier(source_features)

                segmentation_loss = criterion(output_source, target)
                loss = segmentation_loss + pcam_loss + space_loss
                iou = iou_score(output_source, target)
                print('space_contrastive_loss: ', space_loss.item())
                print('segmentation_loss: ', segmentation_loss.item())
                print('pcam_loss:', pcam_loss.item())



            else:
                space_loss = space_contrastive_loss(source_features_high, target_features_high)
                output_source = Classifier(source_features)
                segmentation_loss = criterion(output_source, target)
                loss = segmentation_loss + space_loss

                iou = iou_score(output_source, target)

                print('space_contrastive_loss: ', space_loss.item())
                print('segmentation_loss: ', segmentation_loss.item())

        losses.update(loss.item(), t1.size(0))
        seg_losses.update(segmentation_loss.item(), t1.size(0))
        if args.low_dis:
            pcam_losses.update(pcam_loss.item(), t1.size(0))
        ious.update(iou, t1.size(0))
        space_losses.update(space_loss.item(), t1.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if args.low_dis:
        log = OrderedDict([
            ('loss', losses.avg),
            ('seg_loss', seg_losses.avg),
            ('pcam_loss', pcam_losses.avg),
            ('space_loss', space_losses.avg),
            ('iou', ious.avg),
        ])
    else:
        log = OrderedDict([
            ('loss', losses.avg),
            ('seg_loss', seg_losses.avg),
            ('space_loss', space_losses.avg),
            ('iou', ious.avg),
        ])

    return log


def validate(args, val_loader, source_FE, target_FE, classifier, criterion):
    losses = AverageMeter()
    ious = AverageMeter()

    losses_s = AverageMeter()
    ious_s = AverageMeter()

    losses_t = AverageMeter()
    ious_t = AverageMeter()

    # switch to evaluate mode
    source_FE.eval()
    target_FE.eval()
    classifier.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            t1 = input[:, 0, :, :]
            t2 = input[:, 1, :, :]
            target = target.cuda()

            t1 = t1.unsqueeze(1)
            t2 = t2.unsqueeze(1)

            # compute output
            if args.deepsupervision:
                outputs_source = source_FE(t1)
                outputs_target = target_FE(t2)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                source_features = source_FE(t1)
                target_features = target_FE(t2)
                source_output = classifier(source_features)
                target_output = classifier(target_features)
                loss_s = criterion(source_output, target)
                loss_t = criterion(target_output, target)
                iou_s = iou_score(source_output, target)
                iou_t = iou_score(target_output, target)

            losses_s.update(loss_s.item(), t1.size(0))
            ious_s.update(iou_s, t1.size(0))

            losses_t.update(loss_t.item(), t2.size(0))
            ious_t.update(iou_t, t2.size(0))

    log = OrderedDict([
        ('loss_s', losses_s.avg),
        ('iou_s', ious_s.avg),
        ('loss_t', losses_t.avg),
        ('iou_t', ious_t.avg),
    ])

    return log

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    #args.dataset = "datasets"

    set_seed(42)

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.source_domain, args.model)
        else:
            if args.low_dis:
                args.name = '%s_woDS' %(args.source_domain)
            else:
                args.name = '%s_woDS_wolow_dis' %(args.source_domain)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()

    cudnn.benchmark = True

    contrastive_loss = network.PrototypeClassContrastiveLoss(batch_size=args.batch_size).cuda()
    space_contrastive_loss = network.SpaceContrastiveLoss(batch_size=args.batch_size).cuda()


    # Data loading code
    img_paths = glob(r'/home1/jkwang/dataset/BraTS192D/trainImage/*')
    mask_paths = glob(r'/home1/jkwang/dataset/BraTS192D/trainMask/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))


    # create model
    print("=> creating model %s" %args.model)


    source_FE = network.modeling.__dict__[args.model](num_classes=3, output_stride=args.output_stride)
    target_FE = network.modeling.__dict__[args.model](num_classes=3, output_stride=args.output_stride)

    Classifier = network.DeepLabHeadV3Plus(in_channels=2048, low_level_channels=256, num_classes=3)

    PCAM = network.VCT_Encoder(z_index_dim=4)

    if args.multi_gpu:
        source_FE = nn.DataParallel(source_FE)
        target_FE = nn.DataParallel(target_FE)
        Classifier = nn.DataParallel(Classifier)
        PCAM = nn.DataParallel(PCAM)

    source_FE = source_FE.cuda()
    target_FE = target_FE.cuda()
    Classifier = Classifier.cuda()
    PCAM = PCAM.cuda()

    if args.optimizer == 'Adam':
        optimizer_S = optim.Adam(filter(lambda p: p.requires_grad, source_FE.parameters()), lr=args.lr)
        optimizer_T = optim.Adam(filter(lambda p: p.requires_grad, target_FE.parameters()), lr=args.lr)
        optimizer_C = optim.Adam(filter(lambda p: p.requires_grad, Classifier.parameters()), lr=args.lr)
        optimizer_P = optim.Adam(filter(lambda p: p.requires_grad, PCAM.parameters()), lr=args.lr)
        optimizer = optim.Adam(itertools.chain(source_FE.parameters(), target_FE.parameters(), Classifier.parameters(), PCAM.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer_S = optim.SGD(filter(lambda p: p.requires_grad, source_FE.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer_T = optim.SGD(filter(lambda p: p.requires_grad, target_FE.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer_C = optim.SGD(filter(lambda p: p.requires_grad, Classifier.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer_P = optim.SGD(filter(lambda p: p.requires_grad, PCAM.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer = optim.SGD(itertools.chain(source_FE.parameters(), target_FE.parameters(), Classifier.parameters(), PCAM.parameters()),
                              lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])

    best_iou = 0
    trigger = 0

    writer = SummaryWriter(log_dir='runs_%s/%s' %(args.name, time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))))
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, source_FE, target_FE, Classifier, PCAM, criterion, contrastive_loss, space_contrastive_loss, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, source_FE, target_FE, Classifier, criterion)

        if args.low_dis:
            print(
                'loss %.4f - iou %.4f - seg_loss % .4f - pcam_loss %.4f - space_loss %.4f - val_loss_s %.4f - val_iou_s %.4f - val_loss_t %.4f - val_iou_t %.4f'
                % (train_log['loss'], train_log['iou'], train_log['seg_loss'],
                   train_log['pcam_loss'], train_log['space_loss'], val_log['loss_s'], val_log['iou_s'], val_log['loss_t'], val_log['iou_t']))
        else:
            print(
                'loss %.4f - iou %.4f - seg_loss % .4f  - space_loss %.4f - val_loss_s %.4f - val_iou_s %.4f - val_loss_t %.4f - val_iou_t %.4f'
                % (train_log['loss'], train_log['iou'], train_log['seg_loss'],
                   train_log['space_loss'], val_log['loss_s'], val_log['iou_s'],
                   val_log['loss_t'], val_log['iou_t']))


        writer.add_scalar('loss', train_log['loss'], epoch)
        writer.add_scalar('iou', train_log['iou'], epoch)
        writer.add_scalar('seg_loss', train_log['seg_loss'], epoch)
        if args.low_dis:
            writer.add_scalar('pcam_loss', train_log['pcam_loss'], epoch)
        writer.add_scalar('space_dis', train_log['space_loss'], epoch)
        writer.add_scalar('val_loss_s', val_log['loss_s'], epoch)
        writer.add_scalar('val_iou_s', val_log['iou_s'], epoch)
        writer.add_scalar('val_loss_t', val_log['loss_t'], epoch)
        writer.add_scalar('val_iou_t', val_log['iou_t'], epoch)
        if args.low_dis:
            tmp = pd.Series([
                epoch,
                args.lr,
                train_log['loss'],
                train_log['iou'],
                train_log['seg_loss'],
                train_log['pcam_loss'],
                train_log['space_loss'],
                val_log['loss_s'],
                val_log['iou_s'],
                val_log['loss_t'],
                val_log['iou_t'],
            ], index=['epoch', 'lr', 'loss', 'iou', 'seg_loss', 'pcam_loss', 'space_loss', 'val_loss_s', 'val_iou_s', 'val_loss_t',
                      'val_iou_t'])
        else:
            tmp = pd.Series([
                epoch,
                args.lr,
                train_log['loss'],
                train_log['iou'],
                train_log['seg_loss'],
                train_log['space_loss'],
                val_log['loss_s'],
                val_log['iou_s'],
                val_log['loss_t'],
                val_log['iou_t'],
            ], index=['epoch', 'lr', 'loss', 'iou', 'seg_loss', 'space_loss', 'val_loss_s', 'val_iou_s', 'val_loss_t',
                      'val_iou_t'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.source_domain, index=False)

        trigger += 1

        if val_log['iou_t'] > best_iou:
            # torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            torch.save(
                {
                    'source_FE': source_FE.state_dict(),
                    'target_FE': target_FE.state_dict(),
                    'Classifier': Classifier.state_dict(),
                },
                'models/%s/model.pth' %args.source_domain
            )
            best_iou = val_log['iou_t']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


    # test



if __name__ == '__main__':
    main()
