import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

def cross_entropy_2d(pred,label):
    '''
    Args:
        predict:(n, c, h, w)
        target: (n, h, w)
    '''
    assert not label.requires_grad
    assert pred.dim()   == 4
    assert label.dim()  == 3
    assert pred.size(0) == label.size(0), f'{pred.size(0)}vs{label.size(0)}'
    assert pred.size(2) == label.size(1), f'{pred.size(2)}vs{label.size(2)}'
    assert pred.size(3) == label.size(2), f'{pred.size(3)}vs{label.size(3)}'

    n,c,h,w = pred.size()
    label   = label.view(-1)
    class_count = torch.bincount(label).float()
    try:
        assert class_count.size(0) == 5
        new_class_count = class_count
    except:
        new_class_count = torch.zeros(5).cuda().float()
        new_class_count[:class_count.size(0)] = class_count


    weight      = (1 - new_class_count/label.size(0))
    pred    = pred.transpose(1,2).transpose(2,3).contiguous() #n*c*h*w->n*h*c*w->n*h*w*c
    pred    = pred.view(-1,c)
    loss    = F.cross_entropy(input=pred,target=label,weight=weight)

    return loss

def dice_loss(pred, target):
    """
    input is a torch variable of size [N,C,H,W]
    target: [N,H,W]
    """
    n,c,h,w        = pred.size()
    pred           = pred.cuda()
    target         = target.cuda()
    target_onehot  = torch.zeros([n,c,h,w]).cuda()
    target         = torch.unsqueeze(target,dim=1) # n*1*h*w
    target_onehot.scatter_(1,target,1)

    assert pred.size() == target_onehot.size(), "Input sizes must be equal."
    assert pred.dim()  == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target_onehot.cpu().data.numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    eps = 1e-5
    probs = F.softmax(pred,dim=1)
    num   = probs * target_onehot  # b,c,h,w--p*g
    num   = torch.sum(num, dim=3)  # b,c,h
    num   = torch.sum(num, dim=2)  # b,c,

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)  # b,c,

    den2 = target_onehot * target_onehot  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2.0 * (num / (den1 + den2+eps))  # b,c

    dice_total =  torch.sum(dice) / dice.size(0)  # divide by batch_sz
    return 1 - 1.0 * dice_total/5.0