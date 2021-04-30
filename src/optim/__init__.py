# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import logging

from .optimization import AdamW, Lamb
from .lr_scheduler import WarmupMultiStepLR
from .lr_scheduler import WarmupCosineAnnealingLR
from .lr_scheduler import WarmupLinearSchedule
from .qhm import QHM


def get_opt(cfg, net, resume=False):
    # get optimizer
    lr = cfg.OPTIM.LR
    momentum = cfg.OPTIM.MOM

    # get trainable parameter
    # default no decay parameters for resnets
    no_decay = ['bn.bias', 'bn.weight', 'bn1.bias', 'bn1.weight',
                    'bn2.bias', 'bn2.weight', 'bn3.bias', 'bn3.weight']

    net0 = net.module if hasattr(net, 'module') else net
    if hasattr(net0, 'no_weight_decay'):
        no_decay = list(net0.no_weight_decay())
    params = [
        {'params': [p for n, p in net.named_parameters() if
                    p.requires_grad and not any(nd in n for nd in no_decay)
                    ],
         'weight_decay': cfg.OPTIM.WD},
        {'params': [p for n, p in net.named_parameters() if
                    p.requires_grad and any(nd in n for nd in no_decay)
                    ],
         'weight_decay': cfg.OPTIM.WD0,
         'do_stats': False}
    ]
    print("Parameters without weight decay:")
    print([n for n, p in net.named_parameters() if
           p.requires_grad and any(nd in n for nd in no_decay)])
    if resume:
        for param in params:
            param['initial_lr'] = lr

    if cfg.OPTIM.OPT == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.OPT == 'qhm':
        optimizer = QHM(params, lr=cfg.OPTIM.LR, momentum=momentum,
                        qhm_nu=cfg.OPTIM.NU, weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.OPT == 'adam':
        optimizer = torch.optim.Adam(params, lr=cfg.OPTIM.LR,
                                     betas=(cfg.OPTIM.ADAM.BETA1, cfg.OPTIM.ADAM.BETA2),
                                     weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.OPT == 'lamb':
        optimizer = Lamb(params, lr=lr, eps=cfg.OPTIM.ADAM.EPS)
        logging.info("Using optimizer {}".format(cfg.OPTIM.OPT))
    elif cfg.OPTIM.OPT == 'adamw':
        optimizer = AdamW(params, lr=lr, eps=cfg.OPTIM.ADAM.EPS)
        logging.info("Using optimizer {}".format(cfg.OPTIM.OPT))
    else:
        raise ValueError("Optimizer {} not supported!".format(cfg.OPTIM.OPT))

    return optimizer


def get_lr_scheduler(cfg, optimizer, last_iter=-1):
    lr_policy = cfg.SOLVER.LR_POLICY
    epoch_based = cfg.SOLVER.EPOCH_BASED_SCHEDULE
    if epoch_based:
        warmup_iters = cfg.SOLVER.WARMUP_EPOCHS
        max_iters = int(cfg.OPTIM.EPOCHS)
    else:
        warmup_iters = int(cfg.SOLVER.WARMUP_EPOCHS * cfg.SOLVER.STEPS_PER_EPOCH)
        max_iters = cfg.SOLVER.MAX_ITER
    if lr_policy not in ("multistep", "cosine", 'linear'):
        logging.warning(
            "Only 'multistep', 'cosine' or 'linear' lr policy is accepted, "
            "got {}".format(lr_policy)
        )
        return None
    if lr_policy == "multistep":
        if epoch_based:
            steps = tuple(range(cfg.OPTIM.DROP_FREQ, cfg.OPTIM.EPOCHS,
                                cfg.OPTIM.DROP_FREQ))
        else:
            steps = tuple([epoch*cfg.SOLVER.STEPS_PER_EPOCH for epoch in
                       range(cfg.OPTIM.DROP_FREQ, cfg.OPTIM.EPOCHS, cfg.OPTIM.DROP_FREQ)])
        logging.info("Using scheduler {}".format(lr_policy))
        return WarmupMultiStepLR(
            optimizer,
            steps,
            1.0/cfg.OPTIM.DROP_FACTOR,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=warmup_iters,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            last_epoch=last_iter
        )
    elif lr_policy == "cosine":
        logging.info("Using scheduler {}".format(lr_policy))
        return WarmupCosineAnnealingLR(
            optimizer,
            max_iters,
            cfg.SOLVER.MIN_LR,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=warmup_iters,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            last_epoch=last_iter
        )
    elif lr_policy == "linear":
        logging.info("Using scheduler {}".format(lr_policy))
        return WarmupLinearSchedule(
            optimizer,
            max_iters,
            cfg.SOLVER.MIN_LR,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=warmup_iters,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            last_epoch=last_iter
        )
