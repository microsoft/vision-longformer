"""Unified script for classification problems."""
import argparse
import logging
import os
import os.path as op
import sys

import torch
import torch.utils.data.distributed
from timm.data import Mixup

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

from models import build_model
from engine import train, validate
from loss import get_criterion
from utils.miscellaneous import mkdir, set_seed, config_iteration
from utils.comm import is_main_process, synchronize, get_rank
from optim import get_opt, get_lr_scheduler
from utils.checkpoint import Checkpointer
from utils.metric_logger import TensorboardLogger

from dat.loader import make_epoch_data_loader

from config import cfg

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))


################### parse experiment settings #####################
parser = argparse.ArgumentParser(description='PyTorch for image cls')
parser.add_argument('--config-file',
                    default="",
                    metavar="FILE",
                    help="path to config file",
                    type=str,
                    )
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--data', metavar='DIR', help='path to datasets',
                    default=os.getenv('PT_DATA_DIR', './datasets'))
parser.add_argument('--output_dir', type=str,
                    default=os.getenv('PT_OUTPUT_DIR', '/tmp'))
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert (device == 'cuda')
# Setup CUDA, GPU & distributed training
args.num_gpus = int(
    os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
cfg.distributed = args.num_gpus > 1

if args.local_rank == -1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend=cfg.DISTRIBUTED_BACKEND, init_method="env://")
    args.n_gpu = 1

logging.info("args.n_gpu: {}".format(args.n_gpu))
# Set the random seed manually for reproducibility.
if args.seed != 0:
    set_seed(args.seed, args.n_gpu)

cfg.DATA.PATH = args.data
cfg.OUTPUT_DIR = args.output_dir

##################### Data ############################
logging.info('==> Preparing data..')
testloaders = make_epoch_data_loader(cfg, is_train=False, drop_last=False,
                                    is_distributed=cfg.distributed)

## fix warmup based on trainset size, batch size.
iteration = 0
if not cfg.EVALUATE:
    trainloader = make_epoch_data_loader(cfg, is_train=True, drop_last=True,
                                         is_distributed=cfg.distributed)
    ntrain = len(trainloader.dataset)
    steps_per_epoch = len(trainloader)
    cfg.SOLVER.STEPS_PER_EPOCH = steps_per_epoch
    logs_per_epoch = steps_per_epoch / cfg.LOG_FREQ
    warmup = cfg.OPTIM.WARMUP * steps_per_epoch
    cfg.OPTIM.WARMUP = warmup
    cfg.SOLVER.MAX_ITER = steps_per_epoch * cfg.OPTIM.EPOCHS
    # get the starting checkpoint's iteration
    iteration = config_iteration(cfg.OUTPUT_DIR, steps_per_epoch)

logging.info("Experiment settings:")
logging.info(cfg)

if cfg.OUTPUT_DIR:
    mkdir(cfg.OUTPUT_DIR)
    # save full config to a file in output_dir for future reference
    with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(str(cfg))

cfg.freeze()

# mix-up
aug = cfg.AUG
mixup_fn = Mixup(
        mixup_alpha=aug.MIXUP, cutmix_alpha=aug.MIXCUT,
        cutmix_minmax=aug.MIXCUT_MINMAX if aug.MIXCUT_MINMAX else None,
        prob=aug.MIXUP_PROB, switch_prob=aug.MIXUP_SWITCH_PROB,
        mode=aug.MIXUP_MODE, label_smoothing=cfg.LOSS.LABEL_SMOOTHING,
        num_classes=cfg.DATA.NUM_CLASSES
    ) if aug.MIXUP_PROB > 0.0 else None

##################### Model ############################
net = build_model(cfg)
net = net.to(device)

if not cfg.EVALUATE and cfg.AMP.ENABLED and cfg.AMP.MEMORY_FORMAT == 'nhwc':
    logging.info('=> convert memory format to nhwc')
    net.to(memory_format=torch.channels_last)

# multi-gpu training (should be after apex fp16 initialization)
if args.n_gpu > 1:
    net = torch.nn.DataParallel(net)
    logging.info("Number of GPUs: {}, using DaraParallel.".format(args.n_gpu))
# Distributed training (should be after apex fp16 initialization)
if args.local_rank != -1 and cfg.distributed:
    process_group = torch.distributed.new_group(list(range(args.num_gpus)))
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net, process_group)

    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank], output_device=args.local_rank,
    )
    logging.info("Number of GPUs: {}, using DistributedDaraParallel.".format(args.num_gpus))

##################### Loss function and optimizer ############################
criterion_eval = get_criterion(cfg, train=False)
criterion_eval.cuda()
optimizer = None
scheduler = None
if not cfg.EVALUATE:
    criterion = get_criterion(cfg)
    criterion.cuda()
    optimizer = get_opt(cfg, net, resume=iteration>0)
    scheduler = get_lr_scheduler(cfg, optimizer, last_iter=iteration)

##################### make a checkpoint ############################
best_acc = 0.0
checkpointer = Checkpointer(net, cfg.MODEL.ARCH, best_acc=best_acc,
                            optimizer=optimizer, scheduler=scheduler,
                            save_dir=cfg.OUTPUT_DIR,
                            is_test=cfg.EVALUATE,
                            only_save_last=cfg.ONLY_SAVE_LAST)

filepath = cfg.MODEL.MODEL_PATH
if not os.path.isfile(filepath):
    filepath = os.path.join(cfg.DATA.DATA_DIR, cfg.MODEL.MODEL_PATH)
extra_checkpoint_data = checkpointer.load(filepath)

############## tensorboard writers #############################
tb_log_dir = os.path.join(args.output_dir, 'tf_logs')
train_tb_log_dir = os.path.join(tb_log_dir, 'train_logs')
task_names = [task_name.replace('.yaml', '').replace('/', '_')
              for task_name in cfg.DATA.TEST]
test_tb_log_dirs = [os.path.join(tb_log_dir, '{}_logs'.format(
    task_name)) for task_name in task_names]
train_meters = TensorboardLogger(
    log_dir=train_tb_log_dir,
    delimiter="  ",
)
test_meters = [
    TensorboardLogger(
    log_dir=test_tb_log_dir,
    delimiter="  ",
    ) for test_tb_log_dir in test_tb_log_dirs
]

if cfg.EVALUATE:
    for task_name, testloader, test_meter in zip(task_names, testloaders, test_meters):
        logging.info("Evaluating dataset: {}".format(task_name))
        validate(testloader, net, criterion_eval, cfg,
                 test_meter, global_step=0, device=device,
                 local_rank=get_rank())

############## training code #############################
if not cfg.EVALUATE:
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.AMP.ENABLED)
    # start from epoch 0 or last checkpoint epoch
    start_epoch = checkpointer.epoch
    for epoch in range(start_epoch, cfg.OPTIM.EPOCHS):
        # wait for all processes before every epoch
        synchronize()
        logging.info("PROGRESS: {}%".format(
            round(100 * epoch / cfg.OPTIM.EPOCHS, 4)))
        global_step = epoch * len(trainloader)

        # an empirical rule for redraw projects in Performer
        if cfg.MODEL.ARCH.startswith('msvit') and cfg.MODEL.VIT.MSVIT.ATTN_TYPE == "performer":
            if hasattr(net, 'module'):
                net.module.feature_redraw_interval = 1 + 5 * epoch
            else:
                net.feature_redraw_interval = 1 + 5 * epoch

        if cfg.MODEL.ARCH.startswith('msvit') and cfg.MODEL.VIT.MSVIT.ATTN_TYPE.startswith('longformer'):
            vil_swith_mode = cfg.MODEL.VIT.MSVIT.VIL_MODE_SWITCH * cfg.OPTIM.EPOCHS
            if cfg.MODEL.VIT.MSVIT.MODE > 0 and epoch >= vil_swith_mode:
                # only reset random sample mode to full mode
                if hasattr(net, 'module'):
                    net.module.reset_vil_mode(mode=0)
                else:
                    net.reset_vil_mode(mode=0)

        # train for one epoch
        with torch.autograd.set_detect_anomaly(cfg.SOLVER.DETECT_ANOMALY):
            train(trainloader, net, criterion, optimizer, scheduler, epoch,
                  cfg, train_meters, global_step=global_step, device=device,
                  mixup_fn=mixup_fn, scaler=scaler)

        # evaluate on validation set
        global_step = (epoch + 1) * len(trainloader)
        accs = []
        for task_name, testloader, test_meter in zip(task_names, testloaders, test_meters):
            logging.info("Evaluating dataset: {}".format(task_name))
            acc = validate(testloader, net, criterion_eval, cfg,
                           test_meter, global_step=global_step,
                           device=device,
                           local_rank=get_rank())
            accs.append(acc)

        # remember best prec@1 and save checkpoint
        is_best = accs[0] > checkpointer.best_acc
        if is_best:
            checkpointer.best_acc = accs[0]
        elif cfg.OPTIM.VAL and cfg.OPTIM.OPT in ['sgd', 'qhm', 'salsa']:
            logging.info("DROPPING LEARNING RATE")
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] * 1.0 / cfg.OPTIM.DROP_FACTOR
            if cfg.OPTIM.OPT in ['salsa']:
                optimizer.state['switched'] = True
                logging.info("Switch due to overfiting!")
        checkpointer.epoch = epoch + 1
        checkpointer.save(is_best)

    # exactly evaluate the best checkpoint
    # wait for all processes to complete before calculating the score
    synchronize()
    best_model_path = os.path.join(checkpointer.save_dir, "model_best.pth")
    if os.path.isfile(best_model_path):
        logging.info("Evaluating the best checkpoint: {}".format(best_model_path))
        cfg.defrost()
        cfg.EVALUATE = True
        checkpointer.is_test = True
        cfg.freeze()
        extra_checkpoint_data = checkpointer.load(best_model_path)
        for task_name, testloader, test_meter in zip(task_names, testloaders, test_meters):
            logging.info("Evaluating dataset: {}".format(task_name))
            validate(testloader, net, criterion_eval, cfg,
                     test_meter, global_step=cfg.SOLVER.MAX_ITER, device=device,
                     local_rank=get_rank())


# Close meters
train_meters.close()
for meter in test_meters:
    meter.close()
