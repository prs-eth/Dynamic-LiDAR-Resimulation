import torch
import torch.nn as nn
from NFLStudio.libs.utils import  _EPS

from NFLStudio.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

from NFLStudio.libs.metrics import compute_iou
from NFLStudio.libs.lovasz_softmax import Lovasz_softmax



def get_ce_weights(gt_label, n_classes, max_weights = 50):
    # get inverse_frequency of each class from ground truth label
    counts =[]
    device = gt_label.device
    for label in range(n_classes):
        counts.append((gt_label == label).sum().item()+_EPS)
    counts = torch.tensor(counts).to(device)
    inv_freq = counts.sum() / counts
    seg_weight = torch.clamp(torch.sqrt(inv_freq), 0, max_weights)
    return seg_weight  


class TukeyLoss(nn.Module):
    def __init__(self,c=4.685,normalized=False) -> None:
        super().__init__()
        self.c=c
        self.normalized = normalized
    def forward(self, y1,y2):
        error = y1 - y2
        if not self.normalized:
            mean = error.mean()
            std =torch.sqrt(error.var())
            error = (error - mean) / std
        abs_error = torch.abs(error)
        mask_lessc = abs_error<self.c
        abs_error[mask_lessc] = self.c*self.c / 6.0
        error_gtc = abs_error[~mask_lessc]/self.c
        tmp = 1 - error_gtc*error_gtc
        error = self.c *self.c / 6.0 * tmp*tmp*tmp
        abs_error[~mask_lessc] = error
        return abs_error.mean()




class NeRFBaseLoss(nn.Module):
    """
    NeRF loss and evaluation metrics
    """
    def __init__(self,configs):
        super(NeRFBaseLoss,self).__init__()
        self.device = configs.device
        self.extent = configs.extent
        self.center =  torch.Tensor(configs.center).to(self.device)

        self.loss_weights = configs.loss
        self.clamp_eps = configs.loss['clamp_eps']

        self.mse_loss = nn.MSELoss(reduction = 'mean')
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.gnll_loss = nn.GaussianNLLLoss(eps=self.clamp_eps)  # MSE loss between input and target, while pushing down the variance
        self.chamfer_loss = chamfer_3DDist()
        self.tukey_loss = TukeyLoss()
        self.huber_loss = nn.HuberLoss()

        self.reweight = configs.loss['reweight']
        self.softmax = nn.Softmax(dim = 1)
        self.lovasz_loss = Lovasz_softmax()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()

    def get_seg_loss(self, gt, est):
        """segmentation loss
        Args:
            gt (tensor): [B] long tens
            est (tensor): [B, C]
        """
        stats = dict()
        
        # compute weights in an online fashion
        if self.reweight:
            seg_weights = get_ce_weights(gt, 2)
            criterion = torch.nn.CrossEntropyLoss(weight=seg_weights, ignore_index=-1)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # 1. compute weighted ce loss and lovasz softmax loss
        ce_loss = criterion(est, gt)
        score_softmax = self.softmax(est)
        lovasz_loss = self.lovasz_loss(score_softmax, gt)
        
        score_logsoftmax = self.logsoftmax(est)
        nll_loss = self.nllloss(score_logsoftmax, gt)
        
        stats['bce_loss'] = ce_loss
        stats['lovasz_loss'] = lovasz_loss
        stats['nll_loss'] = nll_loss

        # 2. update intersection, union, recall, precision, 
        predictions = est.argmax(1)
        stats['metric'] = compute_iou(predictions, gt, 2, -1)
        
        return stats
