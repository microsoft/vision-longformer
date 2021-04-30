# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
# Written by Pengchuan Zhang, penzhan@microsoft.com
import logging
import torch.nn as nn
import torchvision.models as tvmodels
from .msvit import MsViT


def build_model(cfg):
    # ResNet models from torchvision
    resnet_model_names = sorted(name for name in tvmodels.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(tvmodels.__dict__[name]))
    print("torchvision models: \n", resnet_model_names)

    # Vision Transformer models
    vitmodeldict = {
        'msvit': MsViT,
    }
    vit_model_names = list(vitmodeldict.keys())
    print("Vision Transformer models: \n", vit_model_names)

    # Build model
    print('==> Building model..')
    if cfg.MODEL.ARCH in resnet_model_names:
        logging.info("Use torchvision predefined model")
        if cfg.MODEL.PRETRAINED:
            logging.info("=> using pre-trained model '{}'".format(cfg.MODEL.ARCH))
            net = tvmodels.__dict__[cfg.MODEL.ARCH](pretrained=True,)
            if net.fc.out_features != cfg.DATA.NUM_CLASSES:
                net.fc = nn.Linear(net.fc.in_features, cfg.DATA.NUM_CLASSES)
        else:
            logging.info("=> creating model '{}'".format(cfg.MODEL.ARCH))
            net = tvmodels.__dict__[cfg.MODEL.ARCH](num_classes=cfg.DATA.NUM_CLASSES)
    elif cfg.MODEL.ARCH in vit_model_names:
        logging.info("Use vision transformer model")
        args = dict(
            img_size=cfg.INPUT.IMAGE_SIZE,
            drop_rate=cfg.MODEL.VIT.DROP,
            drop_path_rate=cfg.MODEL.VIT.DROP_PATH,
            norm_embed=cfg.MODEL.VIT.NORM_EMBED,
            avg_pool=cfg.MODEL.VIT.AVG_POOL,
        )
        if cfg.MODEL.ARCH.startswith('msvit'):
            args['arch'] = cfg.MODEL.VIT.MSVIT.ARCH
            args['sharew'] = cfg.MODEL.VIT.MSVIT.SHARE_W
            args['attn_type'] = cfg.MODEL.VIT.MSVIT.ATTN_TYPE
            args['share_kv'] = cfg.MODEL.VIT.MSVIT.SHARE_KV
            args['only_glo'] = cfg.MODEL.VIT.MSVIT.ONLY_GLOBAL
            args['sw_exact'] = cfg.MODEL.VIT.MSVIT.SW_EXACT
            args['ln_eps'] = cfg.MODEL.VIT.MSVIT.LN_EPS
            args['mode'] = cfg.MODEL.VIT.MSVIT.MODE
        logging.info("=> creating model '{}'".format(cfg.MODEL.ARCH))
        net = vitmodeldict[cfg.MODEL.ARCH](num_classes=cfg.DATA.NUM_CLASSES, **args)
    else:
        raise ValueError(
            "Unimplemented model architecture: {}".format(cfg.MODEL.ARCH))

    return net

