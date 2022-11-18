import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as T
from src.data import transforms as data_transforms
from src.data import ffcv_pipelines
from src.utils import train_utils
from src.utils import run_utils
from src.utils import eval_utils
from fastargs import Section, Param, get_current_config
from torch.nn.utils import vector_to_parameters
import PIL
from ffcv.loader import Loader, OrderOption


def get_fastargs_sections():
    sections = {}

    sections['expt'] = Section('expt', 'experiment arguments').params(
        num_models=Param(int, 'number of models to evaluate', default=10),
        model_type=Param(float, 'model type (aug scale)', required=True),
        save_dir=Param(str, 'directory to save stuff'),
        expt_name=Param(str, 'experiment name', required=True),
    )

    sections['patch'] = Section('patch', 'patch args').params(
        filepath=Param(str, 'file path', required=True),
        delta=Param(float, 'perturbation delta', required=True),
        threshold=Param(float, 'patch threshold b/w', required=True),
        random_crop=Param(int, 'apply random crop to patch (0/1)', required=True)
    )

    sections['dataloading'] = Section('dataloading', 'dataloader arguments').params(
        batch_size=Param(int, 'batch size', default=2500),
        num_workers=Param(int, 'num workers', default=4),
    )

    return sections

def _normalize_image(img):
    tf = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return tf(img)

def _unnormalize_image(img):
    tf = T.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1/0.229, 1/0.224, 1/0.225))
    return tf(img)

def _load_image(path):
    img = PIL.Image.open(path)
    return data_transforms.LIVING17_TRANSFORMS['val'](img)

def get_thresholded_image(img, threshold):
    img = _unnormalize_image(img)
    img = (img.mean(0) > threshold).float()
    img = img.unsqueeze(0).repeat((3,1,1))
    return _normalize_image(img)

def get_patch_function(image_fpath, delta, threshold, random_crop=False, device=torch.device(0)):
    resize_tf = T.RandomResizedCrop(224)

    clip_tf = T.Compose([
        T.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1/0.229, 1/0.224, 1/0.225)),
        T.Lambda(lambda x: x.clip(0,1)),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    patch = _load_image(image_fpath)
    patch = get_thresholded_image(patch, threshold)
    patch = patch.to(device)

    def patch_func(xb):
        xb = xb.clone()

        if random_crop:
            for idx in range(len(xb)):
                P = resize_tf(patch)
                xb[idx] = xb[idx] + delta*P
        else:
            xb = xb + delta*patch

        xb = clip_tf(xb)
        return xb

    return patch_func

def get_model(index, model_type):
    weights_map = {
        0.08: '/mnt/cfs/home/spark/store/model_comp/living17/100pct_aug',
        0.3: '/mnt/cfs/home/spark/store/model_comp/living17/100pct_aug_scale_0.3_1.0',
        0.6: '/mnt/cfs/home/spark/store/model_comp/living17/100pct_aug_scale_0.6_1.0',
        0.9: '/mnt/cfs/home/spark/store/model_comp/living17/100pct_aug_scale_0.9_1.0',
        1: '/mnt/cfs/home/spark/store/model_comp/living17/100pct_no_aug'
    }

    fp = Path(weights_map[model_type]) / 'model_weights.npy'
    W = np.lib.format.open_memmap(fp, mode='r')

    # setup model
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 17)
    PARAM_LEN=11185233

    params = W[index][:PARAM_LEN].astype(float)
    buffs = W[index][PARAM_LEN:].astype(float)
    vector_to_parameters(torch.tensor(params), model.parameters())
    vector_to_parameters(torch.tensor(buffs), model.buffers())

    model = model.eval()
    return model

def get_loader(batch_size=400, num_workers=3, device=torch.device(0)):
    pipelines = ffcv_pipelines.get_pipelines('living17', 'val', device)

    loader = Loader('/mnt/cfs/home/harshay/datasets/living17/val.beton',
                   batch_size=batch_size,
                   num_workers=num_workers,
                   indices=None,
                   order=OrderOption.SEQUENTIAL,
                   drop_last=False,
                   pipelines=pipelines)

    return loader

def get_logits(model_index, model_type, patch_fn, loader, device=torch.device(0)):
    model = get_model(model_index, model_type)
    model = model.to(device).eval().half()
    logits = eval_utils.get_logits(model, loader, device, apply_fn=patch_fn, enable_amp=True)
    model = model.cpu()
    torch.cuda.empty_cache()
    return logits

def run(args):
    # load dataloader + patch func
    loader = get_loader(args.dataloading.batch_size, args.dataloading.num_workers)
    patch_fn = get_patch_function(args.patch.filepath, args.patch.delta, args.patch.threshold, args.patch.random_crop)

    # setup output dir and file
    save_dir = run_utils.get_latest_version_folder(Path(args.expt.save_dir))
    assert save_dir.exists(), "save_dir does not exist"

    ts = train_utils.get_timestamp()
    save_filename = '{}:{}.npy'.format(args.expt.expt_name, ts)
    save_filepath = save_dir / save_filename
    assert not save_filepath.exists(), "save_filepath exists"

    # setup mmap
    mmap_shape = (args.expt.num_models, 3400, 17)
    mmap = np.lib.format.open_memmap(save_filepath, dtype=np.float32, mode='w+', shape=mmap_shape)
    print ("mmap'ed file: {} ({})".format(save_filepath, mmap_shape))

    for model_index in range(args.expt.num_models):
        print (f'Model #{model_index}')
        logits = get_logits(model_index, args.expt.model_type, patch_fn, loader)
        mmap[model_index] = logits
        mmap.flush()

if __name__=='__main__':
    sections = get_fastargs_sections()
    config = get_current_config()
    parser = argparse.ArgumentParser(description='Living17 spider web counterfactuals')
    config.augment_argparse(parser)
    config.validate(mode='stderr')
    config.summary()
    args = config.get()
    print (args)
    run(args)


