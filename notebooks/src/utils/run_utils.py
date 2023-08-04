import os
import random
import torch
from pathlib import Path
from fastargs import Section, Param, get_current_config
from fastargs.validation import And, OneOf, InRange
from typing import Iterable, Optional
import numpy as np

def pickle_obj(obj, save_dir, filename):
    if not save_dir: return
    assert filename.endswith('.pkl')
    path = os.path.join(save_dir, filename)
    torch.save(obj, path)

def get_data_fastargs_section():
    return Section('data', 'data arguments').params(
        dataset=Param(str, 'dataset'),
        seed=Param(int, 'seed', default=random.randint(0,10_000)),
    )

def get_model_fastargs_section():
    return Section('model', 'model arguments').params(
        arch=Param(str, 'model architecture name'),
        pretrained=Param(And(int, OneOf([0,1])), 'pretrained model'),
        pretraining_dataset=Param(str, 'pretraining dataset name', default='')
    )

def get_optimizer_fastargs_section():
    return Section('optimizer', 'optimizer arguments').params(
        lr=Param(float, 'learning rate'),
        wd=Param(float, 'weight decay'),
        momentum=Param(float, 'momentum'),
        use_lr_schedule=Param(And(int, OneOf([0,1])), 'use lr schedule', default=1),
        num_decay=Param(int, 'decay learning rate `num_decay` times', default=2),
        lr_decay_gap=Param(int, 'epoch gap for lr decay'),
        lr_decay_factor=Param(float, 'lr decay factor'),
        use_adam=Param(And(int, OneOf([0,1])), 'use adam with default params', default=1),
    )

def get_dataloading_fastargs_section():
    return Section('dataloading', 'dataloader arguments').params(
        batch_size=Param(int, 'batch size'),
        num_workers=Param(int, 'num workers', required=True),
        pin_memory=Param(And(int, OneOf([0,1])), 'pin memory', default=1),
        shuffle=Param(And(int, OneOf([0,1])), 'shuffle', default=1),
        use_ffcv=Param(And(int, OneOf([0,1])), 'use ffcv', default=1),
        os_cache=Param(And(int, OneOf([0,1])), 'os cache', default=1),
        subsample_prob=Param(float, 'probability of including each datapoint', default=1.0)
    )

def get_training_fastargs_section():
    return Section('training', 'training arguments').params(
        model_selection_loader=Param(str, 'dataset to use for model selection'),
        model_selection_crit=Param(str, 'training stat to use for model selection'),
        max_train_epochs=Param(int, 'max training epochs'),
        train_stop_epsilon=Param(float, 'train stop criterion epsilon'),
        train_stop_crit=Param(str, 'train stop criterion'),
        eval_epoch_gap=Param(int, 'epoch gap for model evaluation'),
        device_id=Param(And(int, InRange(min=0)), 'gpu device id', required=True),
        enable_logging=Param(And(int, OneOf([0,1])), 'enable logging', default=0),
        save_root_dir=Param(str, 'save root dir', required=True),
        checkpoint_type=Param(And(str, OneOf(['model', 'state_dict'])), 'checkpoint type'),
        label_smoothing=Param(float, 'label smoothing hparam', default=0),
    )

def get_latest_version_folder(path):
    def _is_version(p):
        p = Path(p)
        folder = p.name
        if not p.is_dir(): return False
        if not folder.startswith('v'): return False
        if not folder[1:].isnumeric():  return False
        return True

    # find current versions
    path = Path(path)
    assert path.is_dir(), "path is_dir=False"
    versions = [int(f.name[1:]) for f in path.iterdir() if _is_version(f)]
    assert versions is not None, "no version folders"

    max_version_path = path / 'v{}'.format(max(versions))
    assert max_version_path.exists()
    return max_version_path

def make_new_version_folder(path):
    def _is_version(p):
        p = Path(p)
        folder = p.name
        if not p.is_dir(): return False
        if not folder.startswith('v'): return False
        if not folder[1:].isnumeric():  return False
        return True

    # find current versions
    path = Path(path)
    assert path.is_dir(), "path is_dir=False"
    versions = [int(f.name[1:]) for f in path.iterdir() if _is_version(f)]

    # make new version folder
    new_version = max(versions)+1 if versions else 0
    version_path = path / 'v{}'.format(new_version)
    version_path.mkdir(parents=False, exist_ok=False)

    return version_path


def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    r"""Convert parameters to one vector
    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # vec.append(param.view(-1))
        vec.append(param.reshape(-1))
    return torch.cat(vec)


def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Convert one vector to the parameters
    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param



def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.
    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.
    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device

def get_model_weights_flat(model):
    buffs = parameters_to_vector(model.buffers()).cpu().numpy()
    params = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    return np.concatenate([params, buffs]).astype('float16')

def init_model_from_weights(model, weights, PARAM_LEN=2274880):
    # PARAM_LEN: len of params, excluding buffers
    params = weights[:PARAM_LEN]
    buffs = weights[PARAM_LEN:]

    vector_to_parameters(torch.tensor(params), model.parameters())
    vector_to_parameters(torch.tensor(buffs), model.buffers())
    model.float().cpu()