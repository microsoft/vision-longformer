# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import os.path as op


def config_tsv_dataset_args(cfg, dataset_file):
    full_yaml_file = op.join(cfg.DATA.PATH, dataset_file)
    assert op.isfile(full_yaml_file)

    args = dict(
        yaml_file=full_yaml_file,
    )

    tsv_dataset_name = "TSVYamlDataset"
    if 'imagenet_22k' in dataset_file:
        tsv_dataset_name = "ClsTsvDataset"

    return args, tsv_dataset_name
