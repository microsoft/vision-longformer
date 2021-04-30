import time
import logging
import torch
import os
import json
from torch.cuda.amp import autocast

from utils.comm import _accumulate_predictions_from_multiple_gpus, \
    is_main_process

ONLY_OVERLAP_CLASSES = True


def compute_accuracy(output, target, topk=(1,), target_map=None):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        if type(output) is not torch.Tensor:
            # inception v3 model
            output = output[0]

        if target_map is not None and ONLY_OVERLAP_CLASSES:
            overlap_classes = []
            for key, val in target_map.items():
                overlap_classes += val
            output[:, overlap_classes] += output.max() - output.min() + 10

        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        if target_map is None:
            correct = pred.eq(target.view(1, -1).expand_as(pred)).float()
        else:
            target_list = [target_map[t.item()] if t.item() in target_map else
                           [-1] for t in target]
            correct = pred.new_zeros(pred.shape)
            for i, tlist in enumerate(target_list):
                for j in range(maxk):
                    correct[j, i] = pred[j, i] in tlist

        res = []
        for k in topk:
            correct_k = (correct[:k].sum(0, keepdim=False) > 0).float()  # (B, )
            res.append(correct_k)
        return torch.stack(res).t()  # (B, k)


def output_metrics(scores, targets=None):
    avg_scores = 100 * scores.mean(0, keepdim=False)
    metrics = {'top1': avg_scores[0], 'top5': avg_scores[1]}
    if targets is not None:
        assert len(targets) == len(scores), "Length of scores and targets does not match!"
        for label in range(targets.max().item() + 1):
            label_avg_scores = 100 * scores[targets.view(-1) == label].mean(0, keepdim=False)
            metrics.update({'top1/{}'.format(label): label_avg_scores[0],
                            'top5/{}'.format(label): label_avg_scores[1]})
    return metrics


# Training
def train(train_loader, model, criterion, optimizer, scheduler, epoch,
          cfg, meters, global_step=0, device='cuda', mixup_fn=None, scaler=None):
    print('\nEpoch: %d' % epoch)
    if cfg.distributed:
        train_loader.sampler.set_epoch(epoch)

    total_top1 = 0
    total_top5 = 0
    total_cnt = 0
    end = time.time()
    for i, batch in enumerate(train_loader):
        image, target, img_id = batch[0], batch[1], batch[2:]
        # compute output and record loss
        image, target = image.to(device, non_blocking=True), target.to(device,
                                                                       non_blocking=True)
        if mixup_fn:
            image, target = mixup_fn(image, target)

        # measure data loading time
        data_time = time.time() - end

        # switch to train mode
        model.train()

        with autocast(enabled=cfg.AMP.ENABLED):
            if cfg.AMP.ENABLED and cfg.AMP.MEMORY_FORMAT == 'nwhc':
                image = image.contiguous(memory_format=torch.channels_last)
                target = target.contiguous(memory_format=torch.channels_last)

            output = model(image)
            if cfg.MODEL.ARCH == 'inception_v3':
                loss = 0.5 * (criterion(output[0], target) + criterion(output[1],
                                                                       target))
            else:
                loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # closure function defined for line search used in SGD_SLS
        def eval_loss():
            # if cfg.ls_eval:
            if cfg.OPTIM.LS.EVAL:
                model.eval()
            with torch.no_grad():
                return criterion(model(image), target)

        if cfg.OPTIM.OPT in ['salsa', 'ssls', 'slope']:
            scaler.step(optimizer, closure=eval_loss)
        else:
            scaler.step(optimizer)
        scaler.update()
        # update scheduler
        if scheduler and not cfg.SOLVER.EPOCH_BASED_SCHEDULE:
            scheduler.step()

        # measure and record accuracy
        batch_cnt = image.size(0)
        total_cnt += batch_cnt
        if mixup_fn:
            target = torch.argmax(target, dim=1)
        if cfg.LOSS.LOSS == "xentropy":
            precision = compute_accuracy(output, target, topk=(1, 5))
            score = precision.sum(0, keepdim=False)
            total_top1 += score[0].item()
            total_top5 += score[1].item()
        else:
            raise ValueError("Only xentropy loss is supported!")

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()

        metrics_to_log = {
            'time_info': {'compute': batch_time, 'data': data_time},
            'batch_metrics': {'loss': loss, 'qa_cnt': float(batch_cnt),
                              'top1': 100 * score[0] / batch_cnt,
                              'top5': 100 * score[1] / batch_cnt}
        }
        params_to_log = {'params': {'lr': optimizer.param_groups[0]["lr"]}}
        if cfg.OPTIM.OPT in ['salsa', 'sasa']:
            params_to_log.update(
                {
                    'stats': {'stats_x1d': optimizer.state['stats_x1d'],
                              'stats_ld2': optimizer.state['stats_ld2'],
                              'stats_mean': optimizer.state['stats_mean'],
                              'stats_lb': optimizer.state['stats_lb'],
                              'stats_ub': optimizer.state['stats_ub'], }
                }
            )
        meters.update_metrics(metrics_to_log)
        meters.update_params(params_to_log)
        # only log once per cfg.LOG_FREQ param updates. adjust factor because pflug uses
        # 3 batches to make 1 param update.
        if (i + 1) % cfg.LOG_FREQ == 0:
            logging.info(
                meters.delimiter.join(
                    [
                        "iter: {iter}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    iter=global_step + i + 1,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                ) + "\n    " + meters.get_logs(global_step + i + 1)
            )

    # update scheduler
    if scheduler and cfg.SOLVER.EPOCH_BASED_SCHEDULE:
        scheduler.step()

    train_metrics = torch.Tensor([total_top1, total_top5, total_cnt]).to(device)
    if cfg.distributed:
        torch.distributed.all_reduce(train_metrics)
    top1 = 100 * train_metrics[0] / train_metrics[2]
    top5 = 100 * train_metrics[1] / train_metrics[2]
    logging.info(' * Prec@1 {top1:.3f} Prec@5 {top5:.3f}'
                 .format(top1=top1, top5=top5))
    logging.info("Eval Score: %.3f" % top1)
    meters.update_metrics(
        {'epoch_metrics': {'total_cnt': float(train_metrics[2])},
         'accuracy_metrics': {'top1': top1, 'top5': top5, }
         }
    )
    logging.info(
        meters.delimiter.join(
            [
                "iter: {iter}",
                "max mem: {memory:.0f}",
            ]
        ).format(
            iter=global_step + len(train_loader),
            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
        ) + "\n    " + meters.get_logs(int(global_step + len(train_loader)))
    )


def validate(val_loader, model, criterion,
             cfg, meters, global_step=0,
             device='cuda', local_rank=-1):
    # compute target to model output map from target file
    target_map = None
    if cfg.DATA.TARGETMAP:
        target_file = os.path.join(cfg.DATA.PATH, cfg.DATA.TARGETMAP)
        if os.path.isfile(target_file):
            target_file = json.load(open(target_file))
            target_file = {key: val[:val.rfind('(')] for key, val in target_file.items()}
            if hasattr(val_loader.dataset,
                       'labelmap') and val_loader.dataset.labelmap is not None:
                labelmap = val_loader.dataset.labelmap
                target_map = {}
                for objectname, objectid in labelmap.items():
                    target_map[objectid] = []
                    for imagenetname, objectalias in target_file.items():
                        if objectname == objectalias or objectname.startswith(objectalias + '('):
                            target_map[objectid].append(int(imagenetname))
            else:
                logging.warning(
                    "Given validation dataset does not have labelmap!")
        else:
            logging.warning("Given target map file {} does not exists!".format(
                target_file))
    # switch to evaluate mode
    model.eval()
    results_dict = {}
    total_loss = 0
    total_cnt = 0
    total_top1 = 0
    total_top5 = 0
    dataset_len = len(val_loader.dataset)

    with torch.no_grad():
        start = time.time()
        for i, batch in enumerate(val_loader):
            image, target, img_id = batch[0], batch[1], batch[2:]
            if len(img_id) == 0:
                img_id = range(dataset_len * local_rank + total_cnt,
                               dataset_len * local_rank + total_cnt + image.size(
                                   0))
            else:
                img_id = img_id[0].tolist()
            image, target = image.to(device, non_blocking=True), target.to(
                device, non_blocking=True)

            with autocast(enabled=cfg.AMP.ENABLED):
                if cfg.AMP.ENABLED and cfg.AMP.MEMORY_FORMAT == 'nwhc':
                    image = image.contiguous(memory_format=torch.channels_last)
                    target = target.contiguous(
                        memory_format=torch.channels_last)
                # compute output and record loss
                output = model(image)
                loss = criterion(output, target)

            total_loss += loss.item()
            total_cnt += image.size(0)

            # measure and record accuracy
            if cfg.LOSS.LOSS == "xentropy":
                precision = compute_accuracy(output, target, topk=(1, 5),
                                             target_map=target_map)  # B*2
                score = precision.sum(0, keepdim=False)
                total_top1 += score[0].item()
                total_top5 += score[1].item()
                if cfg.EVALUATE:
                    results_dict.update(
                        {im_id: (prec, label) for im_id, prec, label in
                         zip(img_id, precision.to(torch.device("cpu")), target.to(torch.device("cpu")))}
                    )
            else:
                raise ValueError("Only xentropy loss is supported!")

        # measure elapsed time
        total_time = time.time() - start

        # measure epoch metrics
        test_metrics = torch.Tensor([total_loss, total_time, total_cnt, total_top1, total_top5]).to(
            device)
        if cfg.distributed:
            torch.distributed.all_reduce(test_metrics)
            if cfg.EVALUATE:
                results_dict = _accumulate_predictions_from_multiple_gpus(
                    results_dict, cfg.GATHER_ON_CPU,)

        test_loss_gathered = test_metrics[0] / test_metrics[2]
        test_time_gathered = test_metrics[1] / test_metrics[2]
        metrics = {
            'top1': 100 * test_metrics[3] / test_metrics[2],
            'top5': 100 * test_metrics[4] / test_metrics[2]
        }

        output = metrics['top1'].item()
        if not is_main_process():
            # let the main process do the final computing
            return output

        if cfg.EVALUATE:
            assert len(results_dict) == len(val_loader.dataset), \
                "Number of gathered items {} does not match the dataset size {}!" .format(len(results_dict), len(val_loader.dataset))
            scores = torch.stack([val[0] for key, val in results_dict.items()])
            targets = torch.stack([val[1] for key, val in results_dict.items()])
            metrics = output_metrics(scores, targets=targets if cfg.OUTPUT_PERCLASS_ACC else None)
        logging.info("ACCURACY: {}%".format(metrics['top1']))
        meters.update_metrics(
            {'epoch_metrics': {'total_cnt': float(test_metrics[2]),
                               'loss': test_loss_gathered,
                               'time': test_time_gathered},
             'accuracy_metrics': metrics
             }
        )
        logging.info(
            meters.delimiter.join(
                [
                    "iter: {iter}",
                    "max mem: {memory:.0f}",
                ]
            ).format(
                iter=global_step,
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            ) + "\n    " + meters.get_logs(int(global_step))
        )

        # save per image result
        if cfg.EVALUATE and hasattr(val_loader.dataset, 'get_img_key'):
            results_dict = {val_loader.dataset.get_img_key(key): val for key, val in results_dict.items()}
            torch.save(results_dict, os.path.join(meters.tb_logger.logdir, 'results.pth'))

    return output
