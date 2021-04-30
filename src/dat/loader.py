# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import logging
import os

import torch
import torch.utils.data as TD
import torchvision
import torchvision.transforms as transforms

from utils.comm import get_world_size
from . import dataset as D
from . import samplers
from .transforms import build_transforms


from .dataset.utils.config_args import config_tsv_dataset_args


def build_dataset(cfg, is_train=True):
    """
    Arguments:
        cfg: config file.
        is_train (bool): whether to setup the dataset for training or testing
    """
    datasets = []
    for dataset_name in cfg.DATA.TRAIN if is_train else cfg.DATA.TEST:
        if dataset_name.endswith('.yaml'):
            args, tsv_dataset_name = config_tsv_dataset_args(
                cfg, dataset_name
            )
            img_transforms = build_transforms(cfg, is_train)
            args["transforms"] = img_transforms
            dataset = getattr(D, tsv_dataset_name)(**args)
        elif dataset_name == "imagenet":
            if is_train:
                datapath = os.path.join(cfg.DATA.PATH, 'train.zip')
                data_map = os.path.join(cfg.DATA.PATH, 'train_map.txt')
            else:
                datapath = os.path.join(cfg.DATA.PATH, 'val.zip')
                data_map = os.path.join(cfg.DATA.PATH, 'val_map.txt')
            dataset = D.ZipData(
                datapath, data_map,
                build_transforms(cfg, is_train)
            )
        elif dataset_name == "mnist":
            dataset = torchvision.datasets.MNIST(
                root=cfg.DATA.PATH, train=is_train, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )
        elif dataset_name == "cifar":
            if is_train:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ])
                dataset = torchvision.datasets.CIFAR10(
                    root=cfg.DATA.PATH, train=True, download=True,
                    transform=transform_train
                )
            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ])

                dataset = torchvision.datasets.CIFAR10(
                    root=cfg.DATA.PATH, train=False, download=True,
                    transform=transform_test
                )
        elif dataset_name == "cifar100":
            if is_train:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ])
                dataset = torchvision.datasets.CIFAR100(
                    root=cfg.DATA.PATH, train=True, download=True,
                    transform=transform_train
                )
            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ])
                dataset = torchvision.datasets.CIFAR100(
                    root=cfg.DATA.PATH, train=False, download=True,
                    transform=transform_test
                )
        else:
            raise ValueError("Unimplemented dataset: {}".format(dataset_name))

        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = TD.dataset.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed, is_train, cfg):
    if distributed:
        if cfg.AUG.REPEATED_AUG and is_train:
            logging.info('=> using repeated aug sampler')
            return samplers.RASampler(dataset, shuffle=shuffle)
        else:
            return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_epoch_data_loader(cfg, is_train=True,
    drop_last=True, is_distributed=False, start_iter=0):
    datasets = build_dataset(cfg, is_train)
    num_gpus = get_world_size()
    images_per_batch = cfg.DATALOADER.BSZ
    assert (
        images_per_batch % num_gpus == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
    "of GPUs ({}) used.".format(images_per_batch, num_gpus)
    images_per_gpu = images_per_batch // num_gpus
    logger = logging.getLogger(__name__)
    logger.info("Experiment with {} images per GPU".format(images_per_gpu))

    if is_train:
        shuffle = True
    else:
        shuffle = False if not is_distributed else True

    data_loaders = []
    for i, dataset in enumerate(datasets):
        sampler = make_data_sampler(dataset, shuffle, is_distributed, is_train, cfg)
        # default collator works!
        num_workers = cfg.DATALOADER.WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            sampler=sampler,
            batch_size=images_per_gpu,
            drop_last=drop_last,
            pin_memory=True,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]

    return data_loaders
