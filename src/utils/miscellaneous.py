import errno
import os
import os.path as op
import logging
import numpy as np
import torch
import random
import shutil
from .comm import is_main_process
import yaml


def mkdir(path):
    # if it is the current folder, skip.
    # otherwise the original code will raise FileNotFoundError
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())


def delete_tsv_files(tsvs):
    for t in tsvs:
        if op.isfile(t):
            try_delete(t)
        line = op.splitext(t)[0] + '.lineidx'
        if op.isfile(line):
            try_delete(line)


def concat_files(ins, out):
    mkdir(op.dirname(out))
    out_tmp = out + '.tmp'
    with open(out_tmp, 'wb') as fp_out:
        for i, f in enumerate(ins):
            logging.info('concating {}/{} - {}'.format(i, len(ins), f))
            with open(f, 'rb') as fp_in:
                shutil.copyfileobj(fp_in, fp_out, 1024*1024*10)
    os.rename(out_tmp, out)


def concat_tsv_files(tsvs, out_tsv):
    concat_files(tsvs, out_tsv)
    sizes = [os.stat(t).st_size for t in tsvs]
    sizes = np.cumsum(sizes)
    all_idx = []
    for i, t in enumerate(tsvs):
        for idx in load_list_file(op.splitext(t)[0] + '.lineidx'):
            if i == 0:
                all_idx.append(idx)
            else:
                all_idx.append(str(int(idx) + sizes[i - 1]))
    with open(op.splitext(out_tsv)[0] + '.lineidx', 'w') as f:
        f.write('\n'.join(all_idx))


def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result


def try_once(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('ignore error \n{}'.format(str(e)))
    return func_wrapper


@try_once
def try_delete(f):
    os.remove(f)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def print_and_run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def write_to_yaml_file(context, file_name):
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, encoding='utf-8')


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)


def config_iteration(output_dir, steps_per_epoch):
    save_file = os.path.join(output_dir, 'last_checkpoint')
    epoch = 0
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            fname = f.read().strip().strip('.pth')
        model_name = os.path.basename(fname)
        # if model_name.startswith('checkpoint_'):
        epoch = int(model_name[11:])
    return epoch*steps_per_epoch - 1
