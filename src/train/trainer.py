import os
import sys
import time
import random
import copy
from collections import defaultdict
from pathlib import Path
import json
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from ..utils import train_utils
from ..models import multihead

torch.backends.cudnn.benchmark = True

class BaseTrainer(object):

    DEFAULTS_KWS = {
        'max_train_epochs': 200,
        'train_stop_epsilon': 1e-3,
        'eval_epoch_gap': 10,
        'model_selection_crit': 'acc',
        'train_stop_crit': 'loss',
        'enable_amp': True,
        'tqdm_epoch_gap': 1
    }

    MODEL_SELECTION_CRITERIA = {'acc', 'loss'}
    TRAIN_STOP_CRITERIA = {'acc', 'loss'}
    CHECKPOINT_TYPES = {'model', 'state_dict', 'none'}

    def __init__(self, model, optimizer, lr_scheduler=None,
                 device=None, save_dir=None, enable_logging=False,
                 checkpoint_type='model', lock=None):
        """
        - model:
        - opt:
        - lr_scheduler:
        - device: torch.device object
        - save_dir: dir to save {checkpoints, train_stats, train_config}
        - enable_logging: tensorboard logging, log file in save_dir
        - checkpoint_type: save format, {model, state_dict}
        """
        # init
        self.model = model
        self.opt = optimizer
        self.sched = lr_scheduler
        self.device = device or torch.device("cpu")
        self.train_kws = copy.deepcopy(self.DEFAULTS_KWS)

        self.scaler = GradScaler()
        self.lock = lock if lock is not None else nullcontext()

        assert checkpoint_type in self.CHECKPOINT_TYPES
        self.checkpoint_type = checkpoint_type

        # mkdir and setup logging
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=False, exist_ok=True)
            self.train_kws['save_dir'] = save_dir

        self.train_kws['enable_logging'] = self.enable_logging = enable_logging
        if self.enable_logging:
            assert self.save_dir, "save_dir needed for logging"
            self.writer = SummaryWriter(log_dir=self.save_dir, filename_suffix='.log')

    def get_train_config(self):
        return self.train_kws

    def update_train_config(self, **kw):
        for k, v in kw.items():
            assert k in self.train_kws, f"invalid key: {k}"
            self.train_kws[k] = v

    def _get_current_learning_rate(self):
        if self.sched is not None:
            try: return self.sched.get_last_lr()[0]
            except: pass
        return next(iter(self.opt.param_groups))['lr']

    def _backward(self, loss, retain_graph=False):
        if self.train_kws['enable_amp']:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def _opt_step(self):
        if self.train_kws['enable_amp']:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()

    def _eval_step(self, batch):
        return {}

    def _train_step(self, batch):
        return {}

    def run_epoch(self, dl_name, dl, update_model=False):
        # evaluate model on dl_name dataloader
        # update_model: update model with opt if True
        in_tr_mode = self.model.training
        self.model.train() if update_model else self.model.eval()

        meters = defaultdict(lambda: train_utils.AverageMeter())

        with torch.set_grad_enabled(update_model):
            use_tqdm = self._EPOCH_IDX % self.train_kws['tqdm_epoch_gap'] == 0
            tq_dl = tqdm(dl, ascii=True) if use_tqdm else dl

            if use_tqdm:
                tq_dl.set_description(f'E:{self._EPOCH_IDX}, D:{dl_name}')

            for batch in tq_dl:
                update_fn = self._train_step if update_model else self._eval_step

                with autocast(enabled=self.train_kws['enable_amp']):
                    b_stats = update_fn(batch)

                for stat in ['batch_size', 'loss', 'acc']:
                    assert stat in b_stats, f'{stat} not in batch stats output'

                batch_size = b_stats.pop('batch_size')

                for stat, val in b_stats.items():
                    meters[stat].update(val, batch_size)

                cur_mean_stats = dict(sorted([(k, v.mean()) for k,v in meters.items()]))

                if use_tqdm:
                    tq_dl.set_postfix(cur_mean_stats)

        if in_tr_mode and not update_model:
            self.model.train()

        if not in_tr_mode and update_model:
            self.model.eval()

        return {k:v.mean() for k,v in meters.items()}

    def _get_additional_eval_stats(self, dl_name, dl):
        return {}

    def _check_model_selection_criterion(self, cur_stat, best_stat):
        select_crit = self.train_kws['model_selection_crit']

        if best_stat is None:
            return True
        elif select_crit.startswith('acc'):
            return cur_stat > best_stat
        elif select_crit.startswith('loss'):
            return cur_stat < best_stat
        else:
            assert False, f"model selection with metric:{select_crit} not implemented"

    def _check_train_stop_criterion(self, cur_stop_crit_val):
        stop_crit = self.train_kws['train_stop_crit']
        stop_eps = self.train_kws['train_stop_epsilon']

        if stop_crit.startswith('loss'):
            return cur_stop_crit_val <= stop_eps
        elif stop_crit.startswith('acc'):
            return cur_stop_crit_val >= stop_eps
        else:
            assert False, f"stop crit. with metric:{stop_crit} not implemented"

    def get_checkpoint_obj(self, model):
        if self.checkpoint_type == 'model':
            return model
        elif self.checkpoint_type == 'state_dict':
            return model.state_dict()
        else:
            assert False, f'checkpoint type {self.checkpoint_type} not implemented'

    def train(self, train_dl, val_dl_map, model_selection_loader, **train_kws):
        """
        Main model training function, returns epoch-wise train/val stats and
        and state dicts for best and final model
        - train_dl: train dataloader
        - val_dl_map: [name] -> [loader] for eval
        - model_selection_loader: loader in val_dl_map used for model selection
        - train_kws: additional training args
        """
        # setup
        self.update_train_config(**train_kws)

        assert isinstance(val_dl_map, dict)
        assert model_selection_loader in val_dl_map
        assert 'train' not in val_dl_map
        assert self.train_kws['model_selection_crit'] in self.MODEL_SELECTION_CRITERIA
        assert self.train_kws['train_stop_crit'] in self.TRAIN_STOP_CRITERIA

        in_train_mode = self.model.training
        self.model.train()

        stats = defaultdict(lambda: defaultdict(list))
        stop_training = False
        stop_after_eval = False
        best_model_select_stat = None
        cur_model_select_stat = None
        best_model_statedict = None

        self.model.zero_grad()

        # train loop
        for self._EPOCH_IDX in range(1, self.train_kws['max_train_epochs']+1):
            if stop_training:
                break

            try:
                # evaluation + model selection + logging & checkpointing
                if stop_after_eval or self._EPOCH_IDX % self.train_kws['eval_epoch_gap'] == 0:
                    # evaluate loaders
                    eval_stats = {}
                    for dl_name, dl in val_dl_map.items():
                        # compute stats for loader
                        eval_stats[dl_name] = self.run_epoch(dl_name, dl, update_model=False)
                        additional_eval_stats = self._get_additional_eval_stats(dl_name, dl)
                        eval_stats[dl_name].update(additional_eval_stats)

                        # update + log stats
                        for stat_name, stat_scalar in eval_stats[dl_name].items():
                            stats[dl_name][stat_name].append(stat_scalar)

                            if self.enable_logging:
                                self.writer.add_scalar(f'{stat_name}/{dl_name}', stat_scalar, self._EPOCH_IDX)

                    # update best model
                    cur_model_select_stat = eval_stats[model_selection_loader][self.train_kws['model_selection_crit']]
                    update_best_model = self._check_model_selection_criterion(cur_model_select_stat,
                                                                              best_model_select_stat)
                    if update_best_model:
                        best_model_select_stat = cur_model_select_stat
                        train_utils.BCOLORS.print('Best model updated ({:.3f}) | Epoch {}'.format(best_model_select_stat, self._EPOCH_IDX), 'header')

                        if not self.save_dir: # keep track of best sd if not saving
                            best_model_statedict = self.model.state_dict()

                    # save checkpoint(s) and metadata
                    if self.save_dir:
                        latest_ckpt_path = self.save_dir / 'checkpoint_latest.pt'
                        ckpt_obj = self.get_checkpoint_obj(self.model)
                        torch.save(ckpt_obj, latest_ckpt_path)

                        if update_best_model:
                            best_ckpt_path = self.save_dir / 'checkpoint_best.pt'
                            ckpt_obj = self.get_checkpoint_obj(self.model)
                            torch.save(ckpt_obj, best_ckpt_path)

                        train_utils.BCOLORS.print(f'Saved checkpoint(s) at {self.save_dir}', 'bold')

                    if stop_after_eval:
                        stop_training = True

                if stop_training:
                    continue

                # train dataloader epoch
                train_epoch_stats = self.run_epoch('train', train_dl, update_model=True)

                # update + log train stats
                for stat_name, stat_scalar in train_epoch_stats.items():
                    stats['train'][stat_name].append(stat_scalar)

                    if self.enable_logging:
                        self.writer.add_scalar(f'{stat_name}/train', stat_scalar, self._EPOCH_IDX)

                # check stop criterion
                cur_stop_crit_val = train_epoch_stats[self.train_kws['train_stop_crit']]
                stop_after_eval = self._check_train_stop_criterion(cur_stop_crit_val)
                if stop_after_eval:
                    train_utils.BCOLORS.print(f'Stop Criterion Met ({round(cur_stop_crit_val,3)}), Exiting after eval', 'bold')

                # update + log learning rate
                if self.sched is not None:
                    cur_lr = self._get_current_learning_rate()
                    self.sched.step()
                    new_lr = self._get_current_learning_rate()

                    if cur_lr != new_lr:
                        lr_str = 'LR: {:.3f} -> {:.3f}'.format(cur_lr, new_lr)
                        train_utils.BCOLORS.print(lr_str, 'bold')

                    if self.enable_logging:
                        self.writer.add_scalar('learning_rate', new_lr, self._EPOCH_IDX)

                if self.enable_logging:
                    self.writer.flush()

            except KeyboardInterrupt:
                train_utils.BCOLORS.print(f'Interrupt Recvd, Exiting', 'bold')
                stop_training = True

        # wrap up
        if not in_train_mode:
            self.model.eval()

        # save models in self.checkpoint_type format
        # but return final and best models as nn.modules (in cpu)
        checkpoints = {}
        self.model = self.model.cpu()
        checkpoints['latest'] = self.model.eval()

        if self.checkpoint_type=='model' and self.save_dir:
            checkpoints['best'] = torch.load(best_ckpt_path, map_location='cpu')
        elif self.checkpoint_type=='none':
            checkpoints['best'] = None
        else:
            checkpoints['best'] = copy.deepcopy(self.model)
            best_sd = torch.load(best_ckpt_path) if self.save_dir else best_model_statedict
            best_sd = {k:v.cpu() for k,v in best_sd.items()}
            checkpoints['best'].load_state_dict(best_sd)

        if checkpoints['best']:
            checkpoints['best'] = checkpoints['best'].eval()

        # make stats pickle-able ([dl name] -> [stat name] -> stat list)
        stats = json.loads(json.dumps(stats))

        # save stats+config
        config = copy.deepcopy(self.train_kws)
        config['model_info'] = (self.model.__class__.__module__, self.model.__class__.__name__)

        if self.save_dir:
            metadata_path = self.save_dir / 'stats.pkl'
            torch.save({
                'train_stats': stats,
                'train_config': config
            }, metadata_path)

        return {
            'train_stats': stats,
            'train_config': config,
            'checkpoints': checkpoints
        }

class SingleHeadTrainer(BaseTrainer):

    def _eval_step(self, batch):
        # NOTE: in no grad mode
        xb, yb, *zb = batch
        batch_size = len(xb)

        xb = xb.to(self.device, non_blocking=True)
        yb = yb.to(self.device, non_blocking=True)

        with self.lock:
            logits = self.model(xb)

        b_loss = F.cross_entropy(logits, yb)
        b_correct = logits.argmax(-1) == yb
        b_acc = (b_correct.float().mean())

        return {
            'loss': b_loss.item(),
            'acc': b_acc.item(),
            'batch_size': batch_size
        }

    def _train_step(self, batch):
        # NOTE: in grad mode
        stats = {}
        xb, yb, *zb = batch
        stats['batch_size'] = len(xb)

        xb = xb.to(self.device, non_blocking=True)
        yb = yb.to(self.device, non_blocking=True)

        self.opt.zero_grad(set_to_none=True)

        # compute loss and accuracy
        with self.lock:
            logits = self.model(xb)
        b_loss = F.cross_entropy(logits, yb)
        b_correct = logits.argmax(-1) == yb
        b_acc = b_correct.float().mean()

        stats['loss'] = b_loss.item()
        stats['acc'] = b_acc.item()

        # update model
        self._backward(b_loss)
        self._opt_step()

        return stats

class SingleHeadTrainerWorstGroupTracker(SingleHeadTrainer):
    """
    single head trainer + track worst group accuracy
    on {val, test} data for model selection
    """

    MODEL_SELECTION_CRITERIA = {'acc', 'loss', 'acc_wg'}

    def _get_additional_eval_stats(self, dl_name, dl):
        in_tr_mode = self.model.training
        self.model = self.model.eval()

        wg_meters = defaultdict(lambda: train_utils.AverageMeter())

        tq_dl = tqdm(dl, ascii=True)
        # with tqdm(dl, ascii=True) as tq_dl:
        tq_dl.set_description(f'E:{self._EPOCH_IDX}, D:{dl_name}-wg')

        with torch.no_grad():
            for xb, yb, mb in tq_dl:
                xb = xb.to(self.device, non_blocking=True)
                yb, mb = yb.cpu(), mb.cpu().numpy()

                with autocast(enabled=self.train_kws['enable_amp']):
                    with self.lock:
                        logits = self.model(xb)

                preds = logits.argmax(-1).cpu()
                is_correct = (preds==yb).float().numpy()
                yb = yb.numpy()

                for is_c, y, m in zip(is_correct, yb, mb):
                    wg_meters[(y,m)].update(is_c, 1)

                acc_wg = min(v.mean() for v in wg_meters.values())
                tq_dl.set_postfix({'acc_wg': acc_wg})

        acc_wg = min(v.mean() for v in wg_meters.values())

        if in_tr_mode:
            self.model = self.model.train()

        return {
            'acc_wg': acc_wg
        }
