import functools
import time
import random
import os
import numpy as np
import torch
import seaborn as sns
from torch import nn
from types import SimpleNamespace
from scipy.linalg import qr
from ..models import multihead

class NestedDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return self.setdefault(key, NestedDict())

    def __add__(self, other):
        return other

    def __sub__(self, other):
        return other

class NestedNamespace(SimpleNamespace):

    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)

        for key, value in dictionary.items():
            is_dict = isinstance(value, dict)
            all_keys_str = is_dict and all(map(lambda k: type(k) is str, value.keys()))

            if is_dict and all_keys_str:
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

def check_not_multihead(model_arg_index=0):
    """
    asserts for models that do not support multihead.Multihead
    - func: func to wrap
    - model_arg_index: index of model argument in func sig.
    """
    def _check_not_multihead(func):
        """asserts for models that do not support multihead.Multihead"""
        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            model = args[model_arg_index]
            assert isinstance(model, nn.Module), "model type != nn.Module"
            if isinstance(model, multihead.MultiHeadModel):
                assert isinstance(model, multihead.SingleHeadModel), "num_heads > 1 not supported"
                assert model.squeeze, "model.squeeze must be True"
            return func(*args, **kwargs)
        return func_wrapper
    return _check_not_multihead

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def convert_namespace_to_dict(args):
    return {k: (convert_namespace_to_dict(v) if isinstance(v, SimpleNamespace) else v) for k, v in vars(args).items()}

def convert_dict_to_namespace(args):
    return NestedNamespace(args)

def deepcopy_namespace(ns):
    ns_dict  =convert_namespace_to_dict(ns)
    return convert_dict_to_namespace(ns_dict)

def flatten_dictionary(dictionary, join_str='.'):
    fd = {}
    for k, v in dictionary.items():
        is_dict = isinstance(v, dict)
        all_keys_str = is_dict and all(map(lambda c: type(c) is str, v.keys()))

        if is_dict and all_keys_str:
            fd_rec = flatten_dictionary(v, join_str=join_str)
            for k_rec, v_rec in fd_rec.items():
                fd[f'{k}{join_str}{k_rec}'] = v_rec
        else:
            fd[k] = v
    return fd

def update_ax(ax, title=None, xlabel=None, ylabel=None,
              legend_loc='best', legend_cols=1, despine=True,
              ticks_fs=10, label_fs=12, legend_fs=12, legend_title=None,
              title_fs=14, hide_xlabels=False, hide_ylabels=False,
              ticks=True, grid=True, grid_kws=None,
              legend_bbox=None, legend_title_fs=None):

    if title: ax.set_title(title, fontsize=title_fs)
    if xlabel: ax.set_xlabel(xlabel, fontsize=label_fs)
    if ylabel: ax.set_ylabel(ylabel, fontsize=label_fs)
    if legend_loc:
        ax.legend(loc=legend_loc, fontsize=legend_fs, \
                  ncol=legend_cols, title=legend_title, \
                  bbox_to_anchor=legend_bbox, title_fontsize=legend_title_fs)

    if despine: sns.despine(ax=ax)

    if ticks:
        ax.tick_params(direction='in', length=6, width=2, colors='k', which='major', top=False, right=False)
        ax.tick_params(direction='in', length=4, width=1, colors='k', which='minor', top=False, right=False)
        ax.tick_params(labelsize=ticks_fs)

    if hide_xlabels: ax.set_xticks([])
    if hide_ylabels: ax.set_yticks([])

    grid_kws = grid_kws or dict(ls=':', lw=2, alpha=0.5)
    if grid: ax.grid(True, **grid_kws)
    return ax