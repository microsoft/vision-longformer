# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
# Written by Pengchuan Zhang, penzhan@microsoft.com
import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# training data augmentation
_C.INPUT = CN()
_C.INPUT.MEAN = [0.485, 0.456, 0.406]
_C.INPUT.STD = [0.229, 0.224, 0.225]
_C.INPUT.IMAGE_SIZE = 224  # 299 for inception_v3
_C.INPUT.CROP_PCT = 0.875  # 0.816 for inception_v3
_C.INPUT.INTERPOLATION = 2

_C.AMP = CN()
_C.AMP.ENABLED = False
_C.AMP.MEMORY_FORMAT = 'nchw'

# data augmentation
_C.AUG = CN()
_C.AUG.SCALE = (0.08, 1.0)
_C.AUG.RATIO = (3.0/4.0, 4.0/3.0)
_C.AUG.COLOR_JITTER = [0.4, 0.4, 0.4, 0.1, 0.0]
_C.AUG.GRAY_SCALE = 0.0
_C.AUG.GAUSSIAN_BLUR = 0.0
_C.AUG.DROPBLOCK_LAYERS = [3, 4]
_C.AUG.DROPBLOCK_KEEP_PROB = 1.0
_C.AUG.DROPBLOCK_BLOCK_SIZE = 7
_C.AUG.MIXUP_PROB = 0.0
_C.AUG.MIXUP = 0.0
_C.AUG.MIXCUT = 0.0
_C.AUG.MIXCUT_MINMAX = []
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'
_C.AUG.MIXCUT_AND_MIXUP = False
_C.AUG.REPEATED_AUG = False
_C.AUG.TIMM_AUG = CN(new_allowed=True)
_C.AUG.TIMM_AUG.USE_TRANSFORM = False

_C.DATA = CN()
# choices=['toy_ill', 'toy_well', 'mnist', 'cifar', 'cifar100', 'imagenet', 'wikitext-2', 'celeba']
_C.DATA.TRAIN = ('imagenet',)
_C.DATA.TEST = ('imagenet',)
_C.DATA.NUM_CLASSES = 1000
_C.DATA.TARGETMAP = ''
# path to datasets, default=os.getenv('PT_DATA_DIR', './datasets')
_C.DATA.PATH = "./datasets"
# path to other necessary data like checkpoints other than datasets.
_C.DATA.DATA_DIR = "./data"

# choices=['mse', 'xentropy', 'bce'], msr for least regression or xentropy for classification
_C.LOSS = CN()
_C.LOSS.LABEL_SMOOTHING = 0.0
_C.LOSS.LOSS = 'xentropy'
_C.LOSS.FOCAL = CN()
_C.LOSS.FOCAL.NORMALIZE = True
_C.LOSS.FOCAL.ALPHA = 1.0
_C.LOSS.FOCAL.GAMMA = 0.5


# dataloader
_C.DATALOADER = CN()
# batch size
_C.DATALOADER.BSZ = 128
# samples are drawn with replacement if yes
_C.DATALOADER.RE = 'no'
# number of data loading workers
_C.DATALOADER.WORKERS = 0

# optimizer
_C.OPTIM = CN()
# optimizer, default='adamw'
_C.OPTIM.OPT = 'adamw'
# effective learning rate
_C.OPTIM.LR = 1.0
# effective momentum value
_C.OPTIM.MOM = 0.9
# nu value for qhm
_C.OPTIM.NU = 1.0
# weight decay lambda
_C.OPTIM.WD = 5e-4
_C.OPTIM.WD0 = 0.0
# Number of Epochs
_C.OPTIM.EPOCHS = 150
# Warm up: epochs of qhm before switching to sasa/salsa
_C.OPTIM.WARMUP = 0
# Drop frequency and factor for all methods
_C.OPTIM.DROP_FREQ = 50
_C.OPTIM.DROP_FACTOR = 10.0
# use validation dataset to adapt learning rate
_C.OPTIM.VAL = 0
_C.OPTIM.TEST_FREQ = 1000

# ADAM's default parameters
_C.OPTIM.ADAM = CN()
_C.OPTIM.ADAM.BETA1 = 0.9
_C.OPTIM.ADAM.BETA2 = 0.999
_C.OPTIM.ADAM.EPS = 1e-8

# LR scheduler
_C.SOLVER = CN()
_C.SOLVER.LR_POLICY = '' # multistep, cosine, linear
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_EPOCHS = 5.0
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.MIN_LR = 0.0 # MAX_LR is _C.OPTIM.LR
_C.SOLVER.DETECT_ANOMALY = False
_C.SOLVER.EPOCH_BASED_SCHEDULE = False
_C.SOLVER.USE_LARC = False

# models
_C.MODEL = CN()
# choices=model_names + my_model_names + seq_model_names,
#     help='model architecture: ' +
#          ' | '.join(model_names + my_model_names + seq_model_names) +
#          ' (default: resnet18)')
_C.MODEL.ARCH = 'msvit'
# nonlinearity, choices=['celu', 'softplus', 'gelu']
_C.MODEL.NONLINEARITY = 'celu'
# relative path of checkpoint relative to DATA_DIR
_C.MODEL.MODEL_PATH = ""
# use pre-trained model from torchvision
_C.MODEL.PRETRAINED = False
_C.MODEL.FREEZE_CONV_BODY_AT = -1

_C.MODEL.VIT = CN()
_C.MODEL.VIT.DROP = 0.0
_C.MODEL.VIT.DROP_PATH = 0.1
# Add LayerNorm in PatchEmbedding
_C.MODEL.VIT.NORM_EMBED = True
# Use average pooled feature instead of CLS token for classification head
_C.MODEL.VIT.AVG_POOL = False
_C.MODEL.VIT.MSVIT = CN()
# multi-scale model arch: see ReadMe.md for explanation
_C.MODEL.VIT.MSVIT.ARCH = 'l1,h3,d192,n1,s1,g1,p16,f7,a1_l2,h6,d384,n10,s0,g1,p2,f7,a1_l3,h12,d796,n1,s0,g1,p2,f7,a1'
# For vision longformer: whether to share the q/k/v projections of global and local tokens
_C.MODEL.VIT.MSVIT.SHARE_W = True
# choices=['full', 'longformerhand', 'longformerauto', 'linformer', 'srformer', 'performer', 'longformer_cuda']
_C.MODEL.VIT.MSVIT.ATTN_TYPE = 'longformerhand'
# For linformer: whether to share the projection matrices of key and value
_C.MODEL.VIT.MSVIT.SHARE_KV = True
# Only use global attention mechanism
_C.MODEL.VIT.MSVIT.ONLY_GLOBAL = False
# Three masking methods of longformer attention with sliding chunk implementation:
# 1: exact conv-like local attention
# 0: blockwise sliding chunk without padding
# -1: blockwise sliding chunk with cyclic padding
_C.MODEL.VIT.MSVIT.SW_EXACT = 0
# Customized LayerNorm eps
_C.MODEL.VIT.MSVIT.LN_EPS = 1e-6
# mode to control the sampling strategy of neighbor blocks
# 0: all 8 blocks; -1: no neighbor block; >0: random sample one block
_C.MODEL.VIT.MSVIT.MODE = 0
# Switching time from mode 1 to mode 0 during training
_C.MODEL.VIT.MSVIT.VIL_MODE_SWITCH = 0.75

# finetune setting
_C.FINETUNE = CN()
_C.FINETUNE.FINETUNE = False
_C.FINETUNE.USE_TRAIN_AUG = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# default=os.getenv('PT_OUTPUT_DIR', '/tmp')
_C.OUTPUT_DIR = "/tmp"
# default=os.getenv('PHILLY_LOG_DIRECTORY', None)
_C.BACKUP_LOG_DIR = ""
_C.LOG_FREQ = 10
# evaluate model on validation set
_C.EVALUATE = False
_C.OUTPUT_PERCLASS_ACC = False
# Only save the last checkpoint in the checkpointer
_C.ONLY_SAVE_LAST = 0

_C.DISTRIBUTED_BACKEND = "nccl"  # could be "nccl", "gloo" or "mpi"
# whether to use CPU to do gather of predictions. Note that this requires
# running with "gloo" (or "mpi") distributed backend
_C.GATHER_ON_CPU = False