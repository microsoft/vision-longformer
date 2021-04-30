import os
import math
import logging
import shutil
import torch
from collections import OrderedDict
from .comm import is_main_process


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def resize_pos_embed_1d(posemb, shape_new):
    # Rescale the grid of position embeddings when loading from state_dict.
    ntok_old = posemb.shape[1]
    if ntok_old > 1:
        ntok_new = shape_new[1]
        posemb_grid = posemb.permute(0, 2, 1).unsqueeze(dim=-1)
        posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=[ntok_new, 1], mode='bilinear')
        posemb_grid = posemb_grid.squeeze(dim=-1).permute(0, 2, 1)
        posemb = posemb_grid
    return posemb


def resize_pos_embed_2d(posemb, shape_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = shape_new[0]
    gs_old = int(math.sqrt(len(posemb)))  # 2 * w - 1
    gs_new = int(math.sqrt(ntok_new))  # 2 * w - 1
    posemb_grid = posemb.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(gs_new * gs_new, -1)
    return posemb_grid


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, skip_unmatched_layers=True):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.

    If skip_unmatched_layers is True, it will skip layers when the shape mismatch.
    Otherwise, it will raise error.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if model_state_dict[key].shape != loaded_state_dict[key_old].shape and skip_unmatched_layers:
            if 'x_pos_embed' in key or 'y_pos_embed' in key:
                shape_old = loaded_state_dict[key_old].shape
                shape_new = model_state_dict[key].shape
                new_val = resize_pos_embed_1d(loaded_state_dict[key_old], shape_new)
                if shape_new == new_val.shape:
                    model_state_dict[key] = new_val
                    logger.info("[RESIZE] {} {} -> {} {}".format(
                        key_old, shape_old, key, shape_new))
                else:
                    logger.info("[WARNING]", "{} {} != {} {}, skip".format(
                        key_old, new_val.shape, key, shape_new))
            elif 'local_relative_position_bias_table' in key:
                shape_old = loaded_state_dict[key_old].shape
                shape_new = model_state_dict[key].shape
                new_val = resize_pos_embed_2d(loaded_state_dict[key_old], shape_new)
                if shape_new == new_val.shape:
                    model_state_dict[key] = new_val
                    logger.info("[RESIZE] {} {} -> {} {}".format(
                        key_old, shape_old, key, shape_new))
                else:
                    logger.info("[WARNING]", "{} {} != {} {}, skip".format(
                        key_old, new_val.shape, key, shape_new))
            elif 'head' in key:
                shape_new = model_state_dict[key].shape
                logger.info("Use the first {} classes to initialize head because of size mis-match!".format(shape_new[0]))
                if key.endswith('weight'):
                    model_state_dict[key] = loaded_state_dict[key_old][:shape_new[0], :].to(model_state_dict[key].device)
                elif key.endswith('bias'):
                    model_state_dict[key] = loaded_state_dict[key_old][:shape_new[0]].to(model_state_dict[key].device)
                else:
                    raise RuntimeError("Key {} is not expected".format(key))
            else:
                # if layer weights does not match in size, skip this layer
                logger.info("SKIPPING LAYER {} because of size mis-match".format(key))
            continue
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


class Checkpointer(object):
    def __init__(
            self,
            model,
            arch,
            optimizer=None,
            scheduler=None,
            save_dir="",
            logger=None,
            is_test=False,
            epoch=0,
            best_acc=0.,
            only_save_last=0
    ):
        self.model = model
        self.arch = arch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.is_test = is_test
        self.resume = False
        self.epoch = epoch
        self.best_acc = best_acc
        self.only_save_last = only_save_last

    def save(self, is_best, **kwargs):
        name = 'checkpoint_{}'.format(self.epoch)
        if self.only_save_last:
            name = 'checkpoint_last'

        if not (self.save_dir and is_main_process()):
            return

        data = {"net": self.model.state_dict(), "arch": self.arch,
                "epoch": self.epoch, "best_acc": self.best_acc}
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        # self.tag_last_checkpoint(save_file)
        # use relative path name to save the checkpoint
        self.tag_last_checkpoint("{}.pth".format(name))

        if is_best:
            shutil.copyfile(save_file,
                            os.path.join(self.save_dir, "model_best.pth"))

    def load(self, f=None):
        if self.is_test and os.path.isfile(f):
            # load the weights in config file if it is specified in testing
            # stage otherwise it will load the lastest checkpoint in
            # output_dir for testing
            self.logger.info("Loading checkpoint from {}".format(f))
            checkpoint = self._load_file(f)
            self._load_model(checkpoint)
            return checkpoint

        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
            # get the absolute path
            f = os.path.join(self.save_dir, f)
            self.resume = True
        if not os.path.isfile(f):
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from "
                             "scratch")
            # save the random initialization
            self.save(is_best=False)
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        # if resume training, load optimizer and scheduler,
        # otherwise use the specified LR in config yaml for fine-tuning
        if self.resume:
            if "epoch" in checkpoint:
                self.epoch = checkpoint.pop('epoch')
            if "best_acc" in checkpoint:
                self.best_acc = checkpoint.pop('best_acc')
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f.strip(), map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        model_state_dict = self.model.state_dict()
        if 'arch' in checkpoint:
            assert checkpoint.pop('arch') == self.arch
            # remove the "module" prefix before performing the matching
            loaded_state_dict = strip_prefix_if_present(checkpoint.pop("net"),
                                                        prefix="module.")
        else:
            loaded_state_dict = strip_prefix_if_present(checkpoint,
                                                        prefix="module.")
        align_and_update_state_dicts(model_state_dict, loaded_state_dict)
        self.model.load_state_dict(model_state_dict)

