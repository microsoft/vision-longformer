# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right

import torch
import math


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        min_lr=0,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [
                base_lr * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            return [
                self.min_lr + (base_lr - self.min_lr) *
                (1 + math.cos(math.pi * self.last_epoch / self.max_iter)) / 2
                for base_lr in self.base_lrs
            ]


class WarmupLinearSchedule(torch.optim.lr_scheduler._LRScheduler):
    """ Warmup and then linear decay.
        Linearly decreases learning rate from 1. to min_lr
        over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self,
                 optimizer,
                 max_iter,
                 min_lr=0.0,
                 warmup_factor=1.0 / 3,
                 warmup_iters=500,
                 warmup_method="linear",
                 last_epoch=-1):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupLinearSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [
                base_lr * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            rate = max(0.0, float(self.max_iter - self.last_epoch) /
                       float(max(1.0, self.max_iter - self.warmup_iters)))
            return [
                self.min_lr + rate * (base_lr - self.min_lr)
                for base_lr in self.base_lrs
            ]