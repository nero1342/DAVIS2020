import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class diceloss(nn.Module):
    def __init__(self):
        super(diceloss, self).__init__()
    def __call__(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

class DiceLoss(nn.Module):
    r'''
    Dice Loss
    Ref: https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/8
    '''

    def __init__(self, weights=None, ignore_index=None, size_average=True, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        if weights is None:
            self.weights = 1
        if isinstance(weights, list):
            self.weights = torch.FloatTensor(weights)
        self.size_average = size_average
        self.eps = eps

    def __call__(self, output, target):
        encoded_target = torch.zeros(output.size()).to(output.device)
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        output = F.softmax(output, dim=1)
        intersection = output * encoded_target  # [B, C, H, W]
        numerator = 2 * intersection.sum((-1, -2))  # [B, C]
        denominator = output + encoded_target  # [B, C, H, W]
        if self.ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum((-1, -2)) + self.eps

        loss_per_channel = self.weights * (1 - (numerator / denominator))

        loss = loss_per_channel.mean(1)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalLoss(nn.Module):
    
    def __init__(self, alpha = 0.25, gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = torch.Tensor([alpha, 1-alpha])
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )