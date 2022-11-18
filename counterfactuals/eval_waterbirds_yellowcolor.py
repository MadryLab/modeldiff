import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torchvision.transforms as T
from src.data import transforms as data_transforms
from src.data import datasets
from src.utils import train_utils
from src.utils import run_utils
from src.utils import eval_utils
from fastargs import Section, Param, get_current_config

def get_model(index, args):
    _pmap = {
        'IN': 'waterbirds_imagenet',
        'RI': 'waterbirds_random',
        'NZ': 'waterbirds_noise',
        'IN-R18': 'waterbirds_imagenet_resnet18'
    }

    root_dir = Path(args.expt.models_dir)
    model_path = root_dir / _pmap[args.expt.datamodel_name] / 'workers' / str(index) / 'checkpoint_best.pt'

    if not model_path.exists(): return None
    model = torch.load(model_path).eval()
    return model

def get_data(args):
    # transforms
    transforms = data_transforms.WATERBIRDS_TRANSFORMS['test']

    fn = [data_transforms.get_patch_transform((1,1,0), args.patch.size,
                                              (args.patch.xloc,args.patch.yloc),
                                              patch_prob=args.patch.prob)]
    transforms = T.Compose(transforms.transforms+fn)

    # make dataset
    dset = datasets.Waterbirds(args.expt.data_split, get_metadata=1, transform=transforms)
    return dset

def get_fastargs_sections():
    sections = {}

    sections['patch'] = Section('patch', 'patch-specific arguments').params(
        size=Param(int, 'patch size', default=20),
        loc=Param(int, 'location (x=y)', default=112),
        xloc=Param(int, 'row location', default=112),
        yloc=Param(int, 'col location', default=112),
        prob=Param(float, 'probability', default=1)
    )

    sections['expt'] = Section('expt', 'experiment arguments').params(
        num_models=Param(int, 'number of models to evaluate', default=10),
        transform_type=Param(str, 'type of transform {patch,tint}', required=True),
        datamodel_name=Param(str, 'datamodel name', required=True),
        models_dir=Param(str, 'models base directory', default='/mnt/cfs/home/harshay/out/dmgen/misc/pct100_with_models/'),
        data_split=Param(str, 'dataset split', default='test'),
        save_dir=Param(str, 'directory to save stuff', required=True),
        expt_name=Param(str, 'experiment name', required=True)
    )

    sections['dataloading'] = Section('dataloading', 'dataloader arguments').params(
        batch_size=Param(int, 'batch size', default=1000),
        num_workers=Param(int, 'num workers', default=4),
        pin_memory=Param(int, 'pin memory', default=1)
    )

    return sections

def get_predictions(model_index, loader, args):
    device = torch.device(0)
    model = get_model(model_index, args)
    model = model.to(device).eval()
    preds, labels = eval_utils.get_predictions(model, loader, device, enable_amp=True)
    return preds

def get_logits(model_index, loader, args):
    device = torch.device(0)
    model = get_model(model_index, args)
    model = model.to(device).eval()
    logits = eval_utils.get_logits(model, loader, device, enable_amp=True)
    return logits

def run(args):
    dset = get_data(args)
    loader = dset.get_loader(batch_size=args.dataloading.batch_size,
                             num_workers=args.dataloading.num_workers,
                             pin_memory=args.dataloading.pin_memory)

    save_dir = run_utils.get_latest_version_folder(Path(args.expt.save_dir))
    assert save_dir.exists(), "save_dir does not exist"

    ts = train_utils.get_timestamp()
    save_filename = '{}:{}.npy'.format(args.expt.expt_name, ts)
    save_filepath = save_dir / save_filename
    assert not save_filepath.exists(), "save_filepath exists"

    mmap_shape = (args.expt.num_models, len(dset), 2)
    mmap = np.lib.format.open_memmap(save_filepath, dtype=np.float32, mode='w+', shape=mmap_shape)
    print ("mmap'ed file: {} ({})".format(save_filepath, mmap_shape))

    for model_index in range(args.expt.num_models):
        print (f'Model #{model_index}')
        logits = get_logits(model_index, loader, args)
        #preds = get_predictions(model_index, loader, args)
        mmap[model_index] = logits
        mmap.flush()

if __name__=='__main__':
    # get args
    sections = get_fastargs_sections()
    config = get_current_config()
    parser = argparse.ArgumentParser(description='Waterbirds yellow color counterfactuals')
    config.augment_argparse(parser)
    config.validate(mode='stderr')
    config.summary()
    args = config.get()
    print (args)
    run(args)


