import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
from math import ceil
from torch.autograd import Variable
from mmdet.core import auto_fp16, force_fp32
from mmdet.ops import ConvModule, DepthwiseSeparableConvModule, build_upsample_layer
from ..registry import HEADS

class MC(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.convs = nn.ModuleList()
        for i in range(2):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                DepthwiseSeparableConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        x = F.interpolate(x, size=(x.shape[-2]*2, x.shape[-1]*2), 
                          mode='bilinear', align_corners=False)

        return x


class LSFE(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.convs = nn.ModuleList()
        for i in range(2):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                DepthwiseSeparableConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        return x


class DPC(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.convs = nn.ModuleList()
        dilations = [(1,6), (1,1), (6,21), (18,15), (6,3)]

        for i in range(5):
            padding = dilations[i]
            self.convs.append(
                DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.in_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dilation=dilations[i]))
        self.conv = ConvModule(
                    self.in_channels*5,
                    self.conv_out_channels,
                    1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)

    def forward(self, x):
        x = self.convs[0](x)
        x1 = self.convs[1](x)
        x2 = self.convs[2](x)
        x3 = self.convs[3](x)
        x4 = self.convs[4](x3)
        x = torch.cat([
            x,
            x1,
            x2,
            x3,
            x4 
            ], dim=1)

        x = self.conv(x) 
        return x



@HEADS.register_module
class EfficientPSSemanticHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 conv_out_channels=128,
                 num_classes=183,
                 ignore_label=255,
                 loss_weight=1.0,
                 ohem = 0.25,
                 use_unc=False,
                 use_lovasz = False,
                 coef = 0.03,
                 max_epoch=30,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(EfficientPSSemanticHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.ohem = ohem
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.fp16_enabled = False
        self.epoch = 0
        self.iter = 0
        self.max_iter = 1
        self.use_unc = use_unc
        self.use_lovasz = use_lovasz
        self.coef = coef
        self.max_epoch = max_epoch
    
        if self.ohem is not None:
            assert (self.ohem >= 0 and self.ohem < 1)

        self.lateral_convs_ss = nn.ModuleList()
        self.lateral_convs_ls = nn.ModuleList()
        self.aligning_convs = nn.ModuleList()
        self.ss_idx = [3,2]
        self.ls_idx = [1,0] 
        for i in range(2):
            self.lateral_convs_ss.append(
                DPC(
                    self.in_channels,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        for i in range(2):
            self.lateral_convs_ls.append(
                LSFE(
                    self.in_channels,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        for i in range(2):
            self.aligning_convs.append(
                  MC(
                    self.conv_out_channels,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.conv_logits = nn.Conv2d(conv_out_channels * 4, self.num_classes, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none')

    def init_weights(self):
        kaiming_init(self.conv_logits)

    def forward(self, feats):
        feats = list(feats)
        ref_size = tuple(feats[0].shape[-2:])
        for idx, lateral_conv_ss in zip(self.ss_idx, self.lateral_convs_ss):
            feats[idx] = lateral_conv_ss(feats[idx])
       
        x = self.aligning_convs[0](feats[self.ss_idx[1]] + F.interpolate(
                    feats[self.ss_idx[0]], size=tuple(feats[self.ss_idx[1]].shape[-2:]), 
                    mode='bilinear', align_corners=False))

        for idx, lateral_conv_ls in zip(self.ls_idx, self.lateral_convs_ls):
            feats[idx] = lateral_conv_ls(feats[idx])
            feats[idx] = feats[idx] + F.interpolate(
                      x, size=tuple(feats[idx].shape[-2:]), 
                      mode='bilinear', align_corners=False)
            if idx != 0:
                x = self.aligning_convs[1](feats[idx])
        
        for i in range(1,4):
            feats[i] = F.interpolate(
                      feats[i], size=ref_size, 
                      mode='bilinear', align_corners=False)

        x = torch.cat(feats, dim=1)
        x = self.conv_logits(x)
        x = F.interpolate(
                      x, size=(ref_size[0]*4, ref_size[1]*4), 
                      mode='bilinear', align_corners=False)

        return x


    def loss(self, mask_pred, labels):
        loss = dict()
        labels = labels.squeeze(1).long()

        if self.use_lovasz and self.use_unc:
            loss_semantic_seg_lovasz = self.lovasz_softmax(mask_pred, labels,self.use_unc)
            loss_semantic_seg = self.edl_log_loss(mask_pred, labels)
            loss_semantic_seg = loss_semantic_seg.view(-1)
            if self.ohem is not None:
                top_k = int(ceil(loss_semantic_seg.numel() * self.ohem))
                if top_k != loss_semantic_seg.numel():
                    loss_semantic_seg, _ = loss_semantic_seg.topk(top_k)
            loss_semantic_seg = loss_semantic_seg.mean() + loss_semantic_seg_lovasz

        elif self.use_lovasz:
            loss_semantic_seg_lovasz = self.lovasz_softmax(mask_pred, labels)
            loss_semantic_seg = self.criterion(mask_pred, labels)
            loss_semantic_seg = loss_semantic_seg.view(-1)
            if self.ohem is not None:
                top_k = int(ceil(loss_semantic_seg.numel() * self.ohem))
                if top_k != loss_semantic_seg.numel():
                    loss_semantic_seg, _ = loss_semantic_seg.topk(top_k)
            loss_semantic_seg = loss_semantic_seg_lovasz

        elif self.use_unc:
            loss_semantic_seg = self.edl_log_loss(mask_pred, labels)
            loss_semantic_seg = loss_semantic_seg.view(-1)
            if self.ohem is not None:
                top_k = int(ceil(loss_semantic_seg.numel() * self.ohem))
                if top_k != loss_semantic_seg.numel():
                    loss_semantic_seg, _ = loss_semantic_seg.topk(top_k)
            loss_semantic_seg = loss_semantic_seg.mean()

        else:
            loss_semantic_seg = self.criterion(mask_pred, labels)
            loss_semantic_seg = loss_semantic_seg.view(-1)

            if self.ohem is not None:
                top_k = int(ceil(loss_semantic_seg.numel() * self.ohem))
                if top_k != loss_semantic_seg.numel():
                        loss_semantic_seg, _ = loss_semantic_seg.topk(top_k)
    
            loss_semantic_seg = loss_semantic_seg.mean()

        loss_semantic_seg *= self.loss_weight
        loss['loss_semantic_seg'] = loss_semantic_seg
        return loss


    def _expand_onehot_labels(self,labels, target_shape, ignore_index):
        """Expand onehot labels to match the size of prediction."""
        bin_labels = labels.new_zeros(target_shape)
        valid_mask = (labels >= 0) & (labels != ignore_index)
        inds = torch.nonzero(valid_mask, as_tuple=True)
        if inds[0].numel() > 0:
            if labels.dim() == 3:
                bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
            else:
                bin_labels[inds[0], labels[valid_mask]] = 1

        valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
        #if label_weights is None:
        bin_label_weights = valid_mask
        #else:
        #    bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        #    bin_label_weights *= valid_mask

        return bin_labels, bin_label_weights
    
    def softplus_evidence(self, y):
        return F.softplus(y)

    def kl_divergence(self,alpha, num_classes, device =None):
        if not device:
            device = alpha.device
        ones =torch.ones(alpha.size(), dtype=torch.float32, device=device)
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
    
    def lovasz_grad(self,gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1: # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
    
    def edl_loss(self,func, y, alpha, num_classes, device=None):
        y = y.to(device)
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
        curr_epoch = self.epoch
        curr_iter = self.iter
        it_per_epoch = self.max_iter
        total_epochs = self.max_epoch
        den = total_epochs*it_per_epoch
        numer = ((curr_epoch)*it_per_epoch)+curr_iter
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(numer / den, dtype=torch.float32),
        ).to(device)
        
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = (torch.tensor([[self.coef]]).to(device))*annealing_coef * self.kl_divergence(kl_alpha, num_classes, device=device)
        return A + kl_div

    def edl_log_loss(self, output, target, reduction='mean',avg_factor=None,class_weight=None, ignore_index=None):
        device = output.device

        gt_onehot,_ = self._expand_onehot_labels(target, output.shape, 255)
        gt_onehot = gt_onehot.to(device)
        if not device:
            device = get_device()
        evidence = self.softplus_evidence(output)
        num_classes = evidence.shape[1]
        alpha = evidence + 1
        loss = self.edl_loss(
            torch.log, gt_onehot, alpha, num_classes, device
            )
        return loss
    
    def mse_loss(self, sem_logits,label,
        reduction='mean',
        avg_factor=None,
        class_weight=None,
        ignore_index=None):
        ev =self.softplus_evidence(sem_logits)
        num_classes = ev.shape[1]
        alpha = ev +1
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = torch.div(alpha[:],S)
        gt_onehot,_ = self._expand_onehot_labels(label, sem_logits.shape, 255)
        E = (alpha - 1).to(prob.device)
        m = torch.div(alpha[:],S).to(prob.device)
        A = torch.sum((gt_onehot-m)**2, dim=1, keepdim=True).to(prob.device) 
        B = (torch.sum(alpha*(S-alpha)/(S*S*(S+1)),dim=1, keepdim=True)).to(prob.device)
        alp = (E*(1-gt_onehot) + 1).to(prob.device) 
        curr_epoch = self.epoch
        curr_iter = self.iter
        it_per_epoch = self.max_iter
        total_epochs = self.max_epoch
        annealing_coef = (torch.min(torch.tensor([[1.0]]),torch.tensor([[(((curr_epoch)*it_per_epoch)+curr_iter)/(total_epochs*it_per_epoch)]]))).to(prob.device)
        C =  (torch.tensor([[self.coef]]).to(prob.device) *annealing_coef*self.kl_divergence(alp, num_classes, prob.device)).to(prob.device)
        sem_loss = ((A + B) + C)
        return torch.mean(sem_loss)

    def lovasz_softmax(self, pred, labels, use_unc=False,classes='present', per_image=False, ignore=255):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
               Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """
        if use_unc:
            alpha = F.softplus(pred) +1
            S = torch.sum(alpha, dim=1, keepdim=True)
            probas = (alpha/S).to(pred.device)
        else:
            probas = F.softmax(pred, dim =1).to(pred.device)
        if per_image:
            loss = mean(self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
        else:
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss

    def lovasz_softmax_flat(self,probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float() # foreground for class c
            if (classes is 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted))))
        return self.mean(losses)

    def flatten_probas(self,probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels
    
    def mean(self,l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

