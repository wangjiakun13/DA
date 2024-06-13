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
from torch.utils import data
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
from dataset import CTDataset, MRDataset

from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from losses import cross_entropy_2d, dice_loss
from utils import str2bool, count_params
import pandas as pd
import itertools


# arch_names = list(unet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
    parser.add_argument('--source_domain', type=str, default='MR')
    parser.add_argument("--low_dis", type=bool, default=False)
    parser.add_argument("--high_dis", type=bool, default=False)
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_works", type=int, default=4)




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


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    mean_dices = AverageMeter()

    model.train()

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        images_source, labels_source, _, = data
        images_source = images_source.cuda()
        labels_source = labels_source.cuda()

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
            # source_features = source_FE(images_source)
            # source_features_low, source_features_high = source_features['low_level'], source_features['out']
            #
            # source_features_low = source_features_low.flatten(2).permute(0, 2, 1)
            #
            # source_features_high = source_features_high.flatten(1)
            #
            # output_source = Classifier(source_features)
            _, output_source, _ = model(images_source)
            interp = nn.Upsample(size=(256, 256), mode='bilinear',
                                 align_corners=True)
            output_source = interp(output_source)

            loss_seg = cross_entropy_2d(output_source, labels_source)
            loss_dice = dice_loss(output_source, labels_source)

            loss = loss_seg + loss_dice

        losses.update(loss.item(), images_source.size(0))

        _, sval_dice_arr, sval_class_number = dice_eval(pred=output_source, label=labels_source,
                                                        n_class=5)
        sval_dice_arr = np.hstack(sval_dice_arr)

        myo_dice = sval_dice_arr[1]
        lac_dice = sval_dice_arr[2]
        lvc_dice = sval_dice_arr[3]
        aa_dice = sval_dice_arr[4]

        mean_dice = (myo_dice + lac_dice + lvc_dice + aa_dice) / 4

        mean_dices.update(mean_dice, images_source.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        current_losses = {'loss_seg': loss_seg.item(), 'loss_dice': loss_dice.item(), 'mean_dice': mean_dice.item()}
        print_losses(current_losses, i)

    log = OrderedDict([
        ('loss', losses.avg),
        ('mean_dice', mean_dices.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    myo_dices = AverageMeter()
    lac_dices = AverageMeter()
    lvc_dices = AverageMeter()
    aa_dices = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            images_source, labels_source, _, = data
            images_source = images_source.cuda()
            labels_source = labels_source.cuda()

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
                _, source_output, _ = model(images_source)
                interp = nn.Upsample(size=(256, 256), mode='bilinear',
                                     align_corners=True)
                source_output = interp(source_output)
                loss_seg = cross_entropy_2d(source_output, labels_source)
                loss_dice = dice_loss(source_output, labels_source)

                loss = loss_seg + loss_dice

            losses.update(loss.item(), images_source.size(0))
            _, sval_dice_arr, sval_class_number = dice_eval(pred=source_output, label=labels_source,
                                                            n_class=5)
            sval_dice_arr = np.hstack(sval_dice_arr)

            myo_dice = sval_dice_arr[1]
            lac_dice = sval_dice_arr[2]
            lvc_dice = sval_dice_arr[3]
            aa_dice = sval_dice_arr[4]

            myo_dices.update(myo_dice, images_source.size(0))
            lac_dices.update(lac_dice, images_source.size(0))
            lvc_dices.update(lvc_dice, images_source.size(0))
            aa_dices.update(aa_dice, images_source.size(0))




    log = OrderedDict([
        ('loss_s', losses.avg),
        ('myo_dice', myo_dices.avg),
        ('lac_dice', lac_dices.avg),
        ('lvc_dice', lvc_dices.avg),
        ('aa_dice', aa_dices.avg),
        ('mean_dice', (myo_dices.avg + lac_dices.avg + lvc_dices.avg + aa_dices.avg)/4),
    ])

    print('Dice')
    print('######## Source Validation Set ##########')
    print('Each Class Number {}'.format(sval_class_number))
    print('Myo:{:.3f}'.format(myo_dices.avg))
    print('LAC:{:.3f}'.format(lac_dices.avg))
    print('LVC:{:.3f}'.format(lvc_dices.avg))
    print('AA:{:.3f}'.format(aa_dices.avg))
    print('Mean:{:.3f}'.format((myo_dices.avg + lac_dices.avg + lvc_dices.avg + aa_dices.avg)/4))
    print('######## Source Validation Set ##########')

    return log

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dice_eval(pred,label,n_class):
    '''
    pred:  b*c*h*w
    label: b*h*w
    '''
    pred     = torch.argmax(pred,dim=1)  # b*h*w
    dice     = 0
    dice_arr = []
    each_class_number = []
    eps      = 1e-7
    for i in range(n_class):
        A = (pred  == i)
        B = (label == i)
        each_class_number.append(torch.sum(B).cpu().data.numpy())
        inse  = torch.sum(A*B).float()
        union = (torch.sum(A)+torch.sum(B)).float()
        dice  += 2*inse/(union+eps)
        dice_arr.append((2*inse/(union+eps)).cpu().data.numpy())

    return dice,dice_arr,np.hstack(each_class_number)

def to_numpy(tensor):
    if isinstance(tensor,(int,float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def print_losses(current_losses,i_iter):
    list_strings = []
    for loss_name,loss_value in current_losses.items():
        list_strings.append(f'{loss_name}={to_numpy(loss_value):.3f}')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter={i_iter} {full_string}')

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



    # Data loading code
    train_mr_data_pth = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/train_mr1.txt'
    train_ct_data_pth = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/train_ct.txt'
    train_mr_gt_pth = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/train_mr1_gt.txt'
    train_ct_gt_pth = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/train_ct_gt.txt'
    val_mr_data_pth = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/val_mr1.txt'
    val_ct_data_pth = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/val_ct.txt'
    val_mr_gt_pth = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/val_mr1_gt.txt'
    val_ct_gt_pth = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/val_ct_gt.txt'

    transforms = None
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    mrtrain_dataset = MRDataset(data_pth=train_mr_data_pth, gt_pth=train_mr_gt_pth,
                                img_mean=img_mean, transform=transforms)

    cttrain_dataset = CTDataset(data_pth=train_ct_data_pth, gt_pth=train_ct_gt_pth,
                                img_mean=img_mean, transform=transforms)

    mrval_dataset = MRDataset(data_pth=val_mr_data_pth, gt_pth=val_mr_gt_pth,
                              img_mean=img_mean, transform=transforms)

    ctval_dataset = CTDataset(data_pth=val_ct_data_pth, gt_pth=val_ct_gt_pth, img_mean=img_mean,
                              transform=transforms)

    if args.source_domain == 'MR':
        strain_dataset = mrtrain_dataset
        strain_loader = data.DataLoader(strain_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_works,
                                        shuffle=True,
                                        pin_memory=True,
                                        )
        trgtrain_dataset = cttrain_dataset
        trgtrain_loader = data.DataLoader(trgtrain_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_works,
                                          shuffle=True,
                                          pin_memory=True,
                                          )
        sval_dataset = mrval_dataset
        sval_loader = data.DataLoader(sval_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_works,
                                      shuffle=True,
                                      pin_memory=True,
                                      )

        trgval_dataset = ctval_dataset
        trgval_loader = data.DataLoader(trgval_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_works,
                                        shuffle=True,
                                        pin_memory=True,
                                        )

    elif args.source_domain == 'CT':

        strain_dataset = cttrain_dataset
        strain_loader = data.DataLoader(strain_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_works,
                                        shuffle=True,
                                        pin_memory=True,
                                        )
        trgtrain_dataset = mrtrain_dataset
        trgtrain_loader = data.DataLoader(trgtrain_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_works,
                                          shuffle=True,
                                          pin_memory=True,
                                          )
        sval_dataset = ctval_dataset
        sval_loader = data.DataLoader(sval_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_works,
                                      shuffle=True,
                                      pin_memory=True,
                                      )

        trgval_dataset = mrval_dataset
        trgval_loader = data.DataLoader(trgval_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_works,
                                        shuffle=True,
                                        pin_memory=True,
                                        )

    print('dataloader finish')


    # create model
    print("=> creating model %s" %args.model)


    # source_FE = network.modeling.__dict__[args.model](num_classes=4, output_stride=args.output_stride)
    # target_FE = network.modeling.__dict__[args.model](num_classes=4, output_stride=args.output_stride)
    #
    # Classifier = network.DeepLabHeadV3Plus(in_channels=2048, low_level_channels=256, num_classes=5)
    #
    # PCAM = network.VCT_Encoder(z_index_dim=4)

    # model = network.modeling.__dict__[args.model](num_classes=5, output_stride=args.output_stride)

    model = network.get_deeplab_v2(num_classes=5, multi_level=True)

    if args.multi_gpu:
        source_FE = nn.DataParallel(source_FE)
        target_FE = nn.DataParallel(target_FE)
        Classifier = nn.DataParallel(Classifier)
        PCAM = nn.DataParallel(PCAM)

    # source_FE = source_FE.cuda()
    # target_FE = target_FE.cuda()
    # Classifier = Classifier.cuda()
    # PCAM = PCAM.cuda()
    model = model.cuda()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        # optimizer = optim.Adam(itertools.chain(source_FE.parameters(), target_FE.parameters(), Classifier.parameters(), PCAM.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=2.5e-4, momentum=0.9, weight_decay=5e-4)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])

    best_dice = 0
    trigger = 0

    writer = SummaryWriter(log_dir='runs_%s/%s' %(args.name, time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))))
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, sval_loader, model, criterion,  optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, sval_loader, model, criterion)


        writer.add_scalar('loss', train_log['loss'], epoch)
        writer.add_scalar('train_dice', train_log['mean_dice'], epoch)
        writer.add_scalar('val_loss_s', val_log['loss_s'], epoch)
        writer.add_scalar('val_myo_dice_s', val_log['myo_dice'], epoch)
        writer.add_scalar('val_lac_dice_s', val_log['lac_dice'], epoch)
        writer.add_scalar('val_lvc_dice_s', val_log['lvc_dice'], epoch)
        writer.add_scalar('val_aa_dice_s', val_log['aa_dice'], epoch)
        writer.add_scalar('val_mean_dice_s', val_log['mean_dice'], epoch)
        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['mean_dice'],
            val_log['loss_s'],
            val_log['myo_dice'],
            val_log['lac_dice'],
            val_log['lvc_dice'],
            val_log['aa_dice'],
            val_log['mean_dice'],
        ], index=['epoch', 'lr', 'loss', 'train_mean_dice', 'val_loss_s', 'val_myo_dice_s', 'val_lac_dice_s',
                  'val_lvc_dice_s', 'val_aa_dice_s', 'val_mean_dice_s'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.source_domain, index=False)

        trigger += 1

        if val_log['mean_dice'] > best_dice:
            # torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            torch.save(
                {
                    'model': model.state_dict(),
                },
                'models/%s/model.pth' %args.source_domain
            )
            best_dice = val_log['mean_dice']
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
