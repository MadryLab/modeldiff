import argparse
from pathlib import Path
import random
import numpy as np
import torch
from src.data import transforms as data_transforms
from src.data import datasets
from src.utils import train_utils
from src.utils import run_utils
from src.utils import eval_utils
from fastargs import Section, Param, get_current_config

def get_model(index, datamodel_name, checkpoint_type):
    base_dir = Path('/mnt/cfs/home/harshay/out/dmgen/misc/pct100_with_models')
    assert datamodel_name in {'more', 'less', 'med', 'less-2x'}
    assert checkpoint_type in {'best', 'latest'}
    model_path = base_dir / f'cifar_{datamodel_name}_noise' / 'workers' / str(index) / f'checkpoint_{checkpoint_type}.pt'
    if not model_path.exists(): return None
    model = torch.load(model_path).cpu().eval()
    return model

def get_data():
    tf = data_transforms.CIFAR_TRANSFORMS['test']
    dset = datasets.CIFAR(True, False, transform=tf)
    return dset

def get_patch_function(patch_size, x_min, x_max, y_min, y_max):

    def _get_patch(size):
        P = np.ones((size,4))
        P[[-3,-2,-1], 1] = 0
        P[[-3,-2,-1], 2] = 0
        P = P[:,:,None].repeat(3,axis=2)
        return P

    P = _get_patch(patch_size)
    P = data_transforms.CIFAR_TRANSFORMS['test'](P)
    px, py = P.shape[-2:]

    def _patch_fn(xb):
        for img in xb:
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            img[:,x:(x+px),y:(y+py)] = P
        return xb

    return _patch_fn

def get_fastargs_sections():
    sections = {}

    sections['expt'] = Section('expt', 'experiment arguments').params(
        num_models=Param(int, 'number of models to evaluate', default=10),
        datamodel_name=Param(str, 'datamodel name', required=True),
        save_dir=Param(str, 'directory to save stuff'),
        expt_name=Param(str, 'experiment name', required=True),
        checkpoint_type=Param(str, 'model checkpoint type', default='best')
    )

    sections['patch'] = Section('patch', 'patch args').params(
        size=Param(int, 'patch size', required=True),
        x_min=Param(int, 'xmin', default=0),
        x_max=Param(int, 'xmax', default=0),
        y_min=Param(int, 'ymin', default=0),
        y_max=Param(int, 'ymax', default=0)
    )

    sections['dataloading'] = Section('dataloading', 'dataloader arguments').params(
        batch_size=Param(int, 'batch size', default=2500),
        num_workers=Param(int, 'num workers', default=4),
        pin_memory=Param(int, 'pin memory', default=1)
    )

    return sections

def get_logits(model_index, datamodel_name, checkpoint_type, patch_fn, loader):
    device = torch.device(0)
    model = get_model(model_index, datamodel_name, checkpoint_type)
    model = model.to(device).eval()
    logits = eval_utils.get_logits(model, loader, device, apply_fn=patch_fn, enable_amp=True)
    return logits

def run(args):
    # load dataloader
    dset = get_data()
    loader = dset.get_loader(batch_size=args.dataloading.batch_size,
                             num_workers=args.dataloading.num_workers,
                             pin_memory=args.dataloading.pin_memory)

    # load patch fn
    if args.patch.size==0:
        patch_fn = lambda x: x
    else:
        patch_fn = get_patch_function(args.patch.size, args.patch.x_min,
                                    args.patch.x_max, args.patch.y_min,
                                    args.patch.y_max)

    # setup output dir and file
    save_dir = run_utils.get_latest_version_folder(Path(args.expt.save_dir))
    assert save_dir.exists(), "save_dir does not exist"

    ts = train_utils.get_timestamp()
    save_filename = '{}:{}.npy'.format(args.expt.expt_name, ts)
    save_filepath = save_dir / save_filename
    assert not save_filepath.exists(), "save_filepath exists"

    # setup mmap
    mmap_shape = (args.expt.num_models, len(dset), 10)
    mmap = np.lib.format.open_memmap(save_filepath, dtype=np.float32, mode='w+', shape=mmap_shape)
    print ("mmap'ed file: {} ({})".format(save_filepath, mmap_shape))

    for model_index in range(args.expt.num_models):
        print (f'Model #{model_index}')
        logits = get_logits(model_index, args.expt.datamodel_name, args.expt.checkpoint_type, patch_fn, loader)
        mmap[model_index] = logits
        mmap.flush()

if __name__=='__main__':
    # get args
    sections = get_fastargs_sections()
    config = get_current_config()
    parser = argparse.ArgumentParser(description='CIFAR10 black-white texture counterfactuals')
    config.augment_argparse(parser)
    config.validate(mode='stderr')
    config.summary()
    args = config.get()
    print (args)

    run(args)


