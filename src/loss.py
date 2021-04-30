import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def linear_combination(x, y, epsilon):
        return epsilon*x + (1-epsilon)*y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' \
            else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)


class FocalLoss(nn.Module):
    """
    Origianl code is from https://github.com/richardaecn/class-balanced-loss/blob/master/src/cifar_main.py#L226-L266
    """
    def __init__(self, alpha, gamma, normalize):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, preds, targets):
        cross_entropy = F.binary_cross_entropy_with_logits(
            preds, targets, reduction='none'
        )

        gamma = self.gamma
        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = th.exp(
                -gamma * targets * preds - gamma * th.log1p(
                    th.exp(-1.0 * preds)
                )
            )

        loss = modulator * cross_entropy
        weighted_loss = self.alpha * loss
        focal_loss = reduce_loss(weighted_loss, reduction='sum')

        return focal_loss / targets.sum() if self.normalize else focal_loss


class MultiSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, class_weight=None, label_smoothing_value=0):
        super(MultiSoftmaxCrossEntropyLoss, self).__init__()

        self.class_weight = class_weight
        if self.class_weight is not None:
            self.class_weight = self.class_weight.cuda()

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.label_smoothing_value = label_smoothing_value

    def forward(self, input, target):
        return self.cross_entropy(input, target, self.class_weight)

    def cross_entropy(self, pred, soft_targets, class_weight=None):
        if class_weight is not None:
            class_weight_matrix = class_weight.expand_as(soft_targets)
            used_class_weights = th.where(
                soft_targets > 0, class_weight_matrix, soft_targets
            )
            samples_weight = th.max(used_class_weights, dim=1, keepdim=True)[0]

            loss = th.mean(
                th.sum(
                   -samples_weight*soft_targets*self.logsoftmax(pred), 1
                ), 0
            )
        else:
            if self.label_smoothing_value > 0:
                # label smoothing
                batch_size, total_classes_count = soft_targets.size()
                for sample_index in range(batch_size):
                    pos_indices = np.where(soft_targets[sample_index, :] > 0)
                    pos_classes_count = len(pos_indices[0])
                    if pos_classes_count > 0:
                        neg_p = self.label_smoothing_value / float(total_classes_count - pos_classes_count)
                        pos_p = self.label_smoothing_value / pos_classes_count
                        soft_targets[sample_index, :] += neg_p
                        soft_targets[sample_index, pos_indices[0]] = soft_targets[sample_index, pos_indices[0]] - pos_p - neg_p

            loss = th.sum(-soft_targets * self.logsoftmax(pred))
            loss = loss / soft_targets.sum()

        return loss


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = th.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


def get_criterion(config, train=True):
    if config.AUG.MIXUP_PROB > 0.0 and config.LOSS.LOSS == 'xentropy':
        criterion = SoftTargetCrossEntropy() \
            if train else nn.CrossEntropyLoss()
    elif config.LOSS.LABEL_SMOOTHING > 0.0 and config.LOSS.LOSS == 'xentropy':
        criterion = LabelSmoothingCrossEntropy(config.LOSS.LABEL_SMOOTHING)
    elif config.LOSS.LOSS == 'xentropy':
        criterion = nn.CrossEntropyLoss()
    elif config.LOSS.LOSS == 'sigmoid':
        criterion = nn.MultiLabelSoftMarginLoss(reduction='sum')
    elif config.LOSS.LOSS == 'focal':
        alpha = config.LOSS.FOCAL.ALPHA
        gamma = config.LOSS.FOCAL.GAMMA
        normalize = config.LOSS.FOCAL.NORMALIZE
        criterion = FocalLoss(alpha, gamma, normalize)
    elif config.LOSS.LOSS == 'multisoftmax':
        criterion = MultiSoftmaxCrossEntropyLoss()
    elif config.LOSS.LOSS == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif config.LOSS.LOSS == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError('Unkown loss {}'.format(config.LOSS.LOSS))

    return criterion
