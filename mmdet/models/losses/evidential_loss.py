import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones =torch.ones(alpha.size(), dtype=torch.float32, device=device)
    #ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl
    
def _expand_onehot_labels(labels, target_shape, ignore_index):
        """Expand onehot labels to match the size of prediction."""
        
        bin_labels = labels.new_zeros(target_shape)
        valid_mask = (labels >= 0) & (labels != ignore_index)
        inds = torch.nonzero(valid_mask, as_tuple=True)
        if inds[0].numel() > 0:
            if labels.dim() == 3:
                bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
            else:
                bin_labels[inds[0],labels[valid_mask],inds[1]] = 1

        valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
        #if label_weights is None:
        bin_label_weights = valid_mask
        #else:
        #    bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        #    bin_label_weights *= valid_mask
        
        return bin_labels, bin_label_weights

def mask_expand_onehot_labels(labels, target_shape, ignore_index):
        """Expand onehot labels to match the size of prediction."""
        bin_labels = labels.new_zeros(target_shape,dtype=torch.int64)
        valid_mask = (labels >= 0) & (labels != ignore_index)
        inds = torch.nonzero(valid_mask, as_tuple=True)
        if inds[0].numel() > 0:
            if labels.dim() == 3:
                bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
            else:
                bin_labels[inds[0], labels[valid_mask]] = 1

        #valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
        #if label_weights is None:
        #bin_label_weights = valid_mask
        #else:
        #    bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        #    bin_label_weights *= valid_mask

        return bin_labels#, bin_label_weights

def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, curr_iter,it_per_epoch, device=None):
    if not device:
        device = get_device()
    alpha1 = alpha.to(device)
    y2,_ =  _expand_onehot_labels(y1, alpha1.shape, 255)
    y2 = y2.to(device)
    #print(y[0,25], y2[0][:, 25])
    loglikelihood = loglikelihood_loss(y2, alpha, device=device)
    curr_epoch = epoch_num
    total_epochs = 50
    den = total_epochs*it_per_epoch
    numer = ((curr_epoch)*it_per_epoch)+curr_iter
    annealing_coef = torch.min(torch.tensor([[0.03]]), torch.tensor((numer/den))).to(device)
    #annealing_coef =(torch.min(torch.tensor([[1.0]]).to(device),torch.tensor([[(((curr_epoch)*curr_iter)+curr_iter)/(total_epochs*it_per_epoch)]])))).to(device)
        
    kl_alpha = (alpha - 1) * (1 - y2) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div

def edl_loss(func, y, alpha, epoch_num, num_classes, curr_iter, it_per_epoch, device, coef=0.03, max_epoch=30):
    y = y.to(device)
    alpha = alpha.to(device)
    y2,_ =  _expand_onehot_labels(y, alpha.shape, 255)
    y2 = y2.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y2 * (func(S) - func(alpha)), dim=1, keepdim=True)
    curr_epoch = epoch_num
    total_epochs = max_epoch

    den = total_epochs*it_per_epoch
    numer = ((curr_epoch)*it_per_epoch)+curr_iter
    annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32),torch.tensor(numer / den, dtype=torch.float32)).to(device)
        

    kl_alpha = (alpha - 1) * (1 - y2) + 1
    kl_div = (torch.tensor([[coef]]).to(device))*annealing_coef * kl_divergence(kl_alpha, num_classes, device)
    return A + kl_div

def edl_log_loss(output, target, epoch_num,  curr_iter, it_per_epoch,weight=None, reduction='mean', avg_factor=None, coef=0.03, max_epoch=30):
    device = output.device
    num_classes = output.shape[1]
    output = output.permute((1,0))
    output = output[None,:,:].to(output.device)
    target = target[None,:].to(output.device)
    if not device:
        device = get_device()
    evidence = softplus_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes,curr_iter, it_per_epoch, device=device,
             coef=coef, max_epoch=max_epoch
        )
    )
    return loss


def mask_edl_log_loss(output, target, epoch_num,  curr_iter, it_per_epoch,label, reduction='mean', avg_factor=None):
    device = output.device
    num_classes = output.shape[1]
    new_target = (label.unsqueeze(1).unsqueeze(1) * target).to(target.device)
    new_target= new_target.type(torch.int64)
    
    if not device:
        device = get_device()
    evidence = softplus_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mask_edl_loss(
            torch.log, new_target, alpha, epoch_num, num_classes,curr_iter, it_per_epoch, device=device
        )
    )
    return loss

def mask_binary_edl_log_loss(output, target, epoch_num,  curr_iter, it_per_epoch,label, reduction='mean', avg_factor=None, coef=0.03, max_epoch=30):
    device = output.device
    num_rois = output.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=output.device)
    pred_slice = output[inds, label].squeeze(1).to(device)
    rest_slice = (torch.sum(output, dim =1) - pred_slice).to(device)
    combined_logits = torch.cat((rest_slice.unsqueeze(1), pred_slice.unsqueeze(1)), dim =1).to(device)
    num_classes = combined_logits.shape[1]
    if not device:
        device = get_device()
    evidence = softplus_evidence(combined_logits)
    alpha = evidence + 1
    loss = torch.mean(
        mask_edl_loss(
            torch.log, target.type(torch.int64), alpha, epoch_num, num_classes,curr_iter, it_per_epoch, device=device,
            coef=coef, max_epoch=max_epoch
        )
    )
    return loss

def mask_edl_loss(func, y, alpha, epoch_num, num_classes, curr_iter, it_per_epoch, device, coef=0.03, max_epoch=30):

    y = y.to(device)
    alpha = alpha.to(device)
    y2 =  mask_expand_onehot_labels(y, alpha.shape, 255)
    y2 = y2.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y2 * (func(S) - func(alpha)), dim=1, keepdim=True)
    curr_epoch = epoch_num
    total_epochs = max_epoch

    den = total_epochs*it_per_epoch
    numer = ((curr_epoch)*it_per_epoch)+curr_iter
    annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(numer / den, dtype=torch.float32),
        ).to(device)
        

    kl_alpha = (alpha - 1) * (1 - y2) + 1
    kl_div = (torch.tensor([[coef]]).to(device))*annealing_coef * kl_divergence(kl_alpha, num_classes, device)
    return A + kl_div


@LOSSES.register_module
class EvidenceClassLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0,
                 coef=0.03,
                 max_epoch=30):
        super(EvidenceClassLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_evidence = True
        self.coef = coef
        self.max_epoch=max_epoch
        if self.use_evidence and self.use_mask==False:
            self.cls_criterion = edl_log_loss
        elif self.use_mask:
            self.cls_criterion = mask_binary_edl_log_loss
        elif self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self.epoch = 0
        self.iter = 0
        self.max_iter = 1

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        curr_epoch = self.epoch
        curr_iter = self.iter
        it_per_epoch = self.max_iter
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            curr_epoch, curr_iter, it_per_epoch,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            coef=self.coef,
            max_epoch=self.max_epoch,
            **kwargs)
        return loss_cls
