import pandas as pd
import numpy as np
import os
import torch
from torch import nn as nn, Tensor
import torch.nn.functional as F


class zlpr(nn.Module):
    def __init__(self, eps=1):
        super(zlpr, self).__init__()

        self.eps = eps

    def forward(self, y_pred, y_true):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * self.eps
        y_pred_pos = y_pred - (1 - y_true) * self.eps
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        
        return (neg_loss + pos_loss).mean()
    
class zlpr_smooth(nn.Module):
    def __init__(self, eps=1):
        super(zlpr_smooth, self).__init__()

        self.eps = eps

    def forward(self, y_pred, y_true):
        
        y_true = y_true - 0.1 * y_true
        y_true = y_true + 0.1 * (0.9 -y_true)
        
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * self.eps
        y_pred_pos = y_pred - (1 - y_true) * self.eps
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        
        return (neg_loss + pos_loss).mean()

class multilabel_categorical_crossentropy(nn.Module):
    def __init__(self, eps=1e12):
        super(multilabel_categorical_crossentropy, self).__init__()

        self.eps = eps

    def forward(self, y_pred, y_true):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * self.eps
        y_pred_pos = y_pred - (1 - y_true) * self.eps
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        
        return (neg_loss + pos_loss).mean()


class focal_binary_cross_entropy(nn.Module):
    def __init__(self, gamma=2.0, alpha = 0.25, eps=1e-8):
        super(focal_binary_cross_entropy, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = logp*((1-p)**self.gamma)
        loss = 60*loss.mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-12):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * self.eps
        y_pred_pos = y_pred - (1 - y_true) * self.eps
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        # focal loss 계산
        pos_pt = torch.exp(-pos_loss)
        neg_pt = torch.exp(-neg_loss)
        loss = self.alpha * (1 - pos_pt)**self.gamma * pos_loss + (1 - self.alpha) * (1 - neg_pt)**self.gamma * neg_loss

        return loss.mean()


    
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w         
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss
    
    
nINF = -100
class TwoWayLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
                torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()

# def get_criterion(args):
#     if args.loss == 'TwoWayLoss':
#         return TwoWayLoss(Tp=args.loss_Tp, Tn=args.loss_Tn)
#     else:
#         raise ValueError(f"Not supported loss {args.loss}")





class PartialSelectiveLoss(nn.Module):

    def __init__(self, args):
        super(PartialSelectiveLoss, self).__init__()
        self.args = args
        self.clip = args.clip
        self.gamma_pos = args.gamma_pos
        self.gamma_neg = args.gamma_neg
        self.gamma_unann = args.gamma_unann
        self.alpha_pos = args.alpha_pos
        self.alpha_neg = args.alpha_neg
        self.alpha_unann = args.alpha_unann

        self.targets_weights = None

        if args.prior_path is not None:
            print("Prior file was found in given path.")
            df = pd.read_csv(args.prior_path)
            self.prior_classes = dict(zip(df.values[:, 0], df.values[:, 1]))
            print("Prior file was loaded successfully. ")

    def forward(self, logits, targets):

        # Positive, Negative and Un-annotated indexes
        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unann = (targets == -1).float()

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        prior_classes = None
        if hasattr(self, "prior_classes"):
            prior_classes = torch.tensor(list(self.prior_classes.values())).cuda()

        targets_weights = self.targets_weights
        targets_weights, xs_neg = edit_targets_parital_labels(self.args, targets, targets_weights, xs_neg,
                                                              prior_classes=prior_classes)

        # Loss calculation
        BCE_pos = self.alpha_pos * targets_pos * torch.log(torch.clamp(xs_pos, min=1e-8))
        BCE_neg = self.alpha_neg * targets_neg * torch.log(torch.clamp(xs_neg, min=1e-8))
        BCE_unann = self.alpha_unann * targets_unann * torch.log(torch.clamp(xs_neg, min=1e-8))

        BCE_loss = BCE_pos + BCE_neg + BCE_unann

        # Adding asymmetric gamma weights
        with torch.no_grad():
            asymmetric_w = torch.pow(1 - xs_pos * targets_pos - xs_neg * (targets_neg + targets_unann),
                                     self.gamma_pos * targets_pos + self.gamma_neg * targets_neg +
                                     self.gamma_unann * targets_unann)
        BCE_loss *= asymmetric_w

        # partial labels weights
        BCE_loss *= targets_weights

        return -BCE_loss.sum()


def edit_targets_parital_labels(args, targets, targets_weights, xs_neg, prior_classes=None):
    # targets_weights is and internal state of AsymmetricLoss class. we don't want to re-allocate it every batch
    if args.partial_loss_mode is None:
        targets_weights = 1.0

    elif args.partial_loss_mode == 'negative':
        # set all unsure targets as negative
        targets_weights = 1.0

    elif args.partial_loss_mode == 'ignore':
        # remove all unsure targets (targets_weights=0)
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        targets_weights[targets == -1] = 0

    elif args.partial_loss_mode == 'ignore_normalize_classes':
        # remove all unsure targets and normalize by Durand et al. https://arxiv.org/pdf/1902.09720.pdfs
        alpha_norm, beta_norm = 1, 1
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        n_annotated = 1 + torch.sum(targets != -1, axis=1)    # Add 1 to avoid dividing by zero

        g_norm = alpha_norm * (1 / n_annotated) + beta_norm
        n_classes = targets_weights.shape[1]
        targets_weights *= g_norm.repeat([n_classes, 1]).T
        targets_weights[targets == -1] = 0

    elif args.partial_loss_mode == 'selective':
        if targets_weights is None or targets_weights.shape != targets.shape:
            targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        else:
            targets_weights[:] = 1.0
        num_top_k = args.likelihood_topk * targets_weights.shape[0]

        xs_neg_prob = xs_neg
        if prior_classes is not None:
            if args.prior_threshold:
                idx_ignore = torch.where(prior_classes > args.prior_threshold)[0]
                targets_weights[:, idx_ignore] = 0
                targets_weights += (targets != -1).float()
                targets_weights = targets_weights.bool()

        negative_backprop_fun_jit(targets, xs_neg_prob, targets_weights, num_top_k)

        # set all unsure targets as negative
        # targets[targets == -1] = 0

    return targets_weights, xs_neg


# @torch.jit.script
def negative_backprop_fun_jit(targets: Tensor, xs_neg_prob: Tensor, targets_weights: Tensor, num_top_k: int):
    with torch.no_grad():
        targets_flatten = targets.flatten()
        cond_flatten = torch.where(targets_flatten == -1)[0]
        targets_weights_flatten = targets_weights.flatten()
        xs_neg_prob_flatten = xs_neg_prob.flatten()
        ind_class_sort = torch.argsort(xs_neg_prob_flatten[cond_flatten])
        targets_weights_flatten[
            cond_flatten[ind_class_sort[:num_top_k]]] = 0


class ComputePrior:
    def __init__(self, classes):
        self.classes = classes
        n_classes = len(self.classes)
        self.sum_pred_train = torch.zeros(n_classes).cuda()
        self.sum_pred_val = torch.zeros(n_classes).cuda()
        self.cnt_samples_train,  self.cnt_samples_val = .0, .0
        self.avg_pred_train, self.avg_pred_val = None, None
        self.path_dest = "./outputs"
        self.path_local = "/class_prior/"

    def update(self, logits, training=True):
        with torch.no_grad():
            preds = torch.sigmoid(logits).detach()
            if training:
                self.sum_pred_train += torch.sum(preds, axis=0)
                self.cnt_samples_train += preds.shape[0]
                self.avg_pred_train = self.sum_pred_train / self.cnt_samples_train

            else:
                self.sum_pred_val += torch.sum(preds, axis=0)
                self.cnt_samples_val += preds.shape[0]
                self.avg_pred_val = self.sum_pred_val / self.cnt_samples_val

    def save_prior(self):

        print('Prior (train), first 5 classes: {}'.format(self.avg_pred_train[:5]))

        # Save data frames as csv files
        if not os.path.exists(self.path_dest):
            os.makedirs(self.path_dest)

        df_train = pd.DataFrame({"Classes": list(self.classes.values()),
                                 "avg_pred": self.avg_pred_train.cpu()})
        df_train.to_csv(path_or_buf=os.path.join(self.path_dest, "train_avg_preds.csv"),
                        sep=',', header=True, index=False, encoding='utf-8')

        if self.avg_pred_val is not None:
            df_val = pd.DataFrame({"Classes": list(self.classes.values()),
                                   "avg_pred": self.avg_pred_val.cpu()})
            df_val.to_csv(path_or_buf=os.path.join(self.path_dest, "val_avg_preds.csv"),
                          sep=',', header=True, index=False, encoding='utf-8')

    def get_top_freq_classes(self):
        n_top = 10
        top_idx = torch.argsort(-self.avg_pred_train.cpu())[:n_top]
        top_classes = np.array(list(self.classes.values()))[top_idx]
        print('Prior (train), first {} classes: {}'.format(n_top, top_classes))
        
        
        
