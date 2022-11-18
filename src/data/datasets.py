import pathlib
from pathlib import Path
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image

import wilds
from wilds.common.data_loaders import get_train_loader as get_wilds_train_loader
from wilds.common.data_loaders import get_eval_loader as get_wilds_eval_loader

from ..utils import data_utils
from ..utils import common_utils
from . import transforms as data_transforms

try:
    from ffcv.fields import IntField, RGBImageField, NDArrayField
    from ffcv.writer import DatasetWriter
except:
    print('No FFCV installation found.')

PARENT_DIR = pathlib.Path(__file__).parent
DATA_DIR = os.path.expanduser('~/datasets/')
CUR_FILE_DIR = Path(__file__).parent.resolve()

"""
Classes
- DatasetwithLoadear
- GroupedDataset

- CIFAR
- CorruptCIFAR10
- Waterbirds
"""

class DatasetWithLoader(Dataset):

    def get_loader(self, **loader_kws):
        loader = DataLoader(self, **loader_kws)
        return loader

    def write_ffcv_beton(self, save_dir, file_name=None, num_workers=4):
        raise NotImplementedError

class GroupedDataset(DatasetWithLoader):

    def __init__(self, dataset, class_to_group_map, replace_original_class, class_batch_index=1):
        """
        Partition data into groups of classes / superclasses
        - dataset: instance of torch.utils.data.Dataset
                with dataset[idx] = (..., label)

        - class_to_group_map: maps class index to group / superclass index
                            (delete classes that are not in class_to_group_map)

        - replace_original_class: replace original class if true else add group to data tuple
        - class_map_index: index of label/class (that needs to be mapped) in the batch tuple
        """
        # asserts
        groups = set(class_to_group_map.values())
        assert isinstance(dataset, Dataset)
        assert min(groups) == 0 and max(groups) == len(groups)-1

        # setup
        self.dataset = dataset
        self.class_to_group_map = class_to_group_map
        self.replace_original_class = replace_original_class
        self.class_batch_index = class_batch_index

        self.num_groups = len(groups)
        self.num_classes = len(self.class_to_group_map)

        # map class subset to [0,..,k]
        old_classes = list(class_to_group_map.keys())
        self.old_to_new_class_map = dict(zip(old_classes, range(self.num_classes)))

        # group to new class map
        self.group_to_class_map = defaultdict(list)
        for c, g in self.class_to_group_map.items():
            c_new = self.old_to_new_class_map[c]
            self.group_to_class_map[g].append(c_new)

        # delete datapoints with classes not in class_to_group map
        valid_classes = set(self.class_to_group_map)
        valid_indices = []

        for idx, tup in enumerate(self.dataset):
            label = tup[self.class_batch_index]
            if label in valid_classes:
                valid_indices.append(idx)

        # note: subset-ing retains transforms / target transforms
        self.dataset = Subset(self.dataset, valid_indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)): raise IndexError
        data_tuple = list(self.dataset[idx])
        y_old = data_tuple[self.class_batch_index]

        # group label
        g = self.class_to_group_map[y_old]

        if self.replace_original_class:
            data_tuple[self.class_batch_index] = g
            return data_tuple

        # old_to_new label map
        data_tuple[self.class_batch_index] = self.old_to_new_class_map[y_old]
        data_tuple.append(g)

        return data_tuple

class IndexedDataset(DatasetWithLoader):
    """Dataset wrapper that appends datapoint indices to datapoint tuple"""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        tup = self.dataset[index]
        return *tup, index

# CIFAR datasets

class CIFAR(DatasetWithLoader):
    """Common Dataset class for CIFAR10 and CIFAR100"""

    CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
    CIFAR10_STD = np.array([0.2023, 0.1994, 0.2010])

    CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408])
    CIFAR100_STD = np.array([0.2675, 0.2565, 0.2761])

    def __init__(self, cifar10, train,
                transform=None, target_transform=None):
        super().__init__()
        self.is_cifar10 = cifar10
        self.train = train
        self.split_name = 'train' if self.train else 'test'
        self.transform=transform
        self.target_transform=target_transform

        if self.is_cifar10:
            self.dset_name = 'cifar10'
            self.num_classes = 10
            self.root = os.path.join(DATA_DIR, 'cifar10')
            self.tensor_channel_mean = CIFAR.CIFAR10_MEAN
            self.tensor_channel_std = CIFAR.CIFAR10_STD
            dset_func = torchvision.datasets.CIFAR10
        else:
            self.dset_name = 'cifar100'
            self.num_classes = 100
            self.root = os.path.join(DATA_DIR, 'cifar100')
            self.tensor_channel_mean = CIFAR.CIFAR100_MEAN
            self.tensor_channel_std = CIFAR.CIFAR100_STD
            dset_func = torchvision.datasets.CIFAR100

        self.dataset = dset_func(self.root, download=True,
                                train=self.train, transform=self.transform,
                                target_transform=self.target_transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def write_ffcv_beton(self, save_dir, file_name=None, num_workers=4):
        save_dir = pathlib.Path(save_dir)
        assert save_dir.exists()

        if file_name:
            assert file_name.endswith('.beton')
        else:
            file_name = f'{self.dset_name}_{self.split_name}.beton'

        dset_path = save_dir / file_name

        writer = DatasetWriter(str(dset_path), {
            'image': RGBImageField(),
            'label': IntField()
        }, num_workers=num_workers)

        writer.from_indexed_dataset(self)
        return dset_path

class IndexedCIFAR(IndexedDataset):

    def __init__(self, is_cifar10, is_train, transform=None, target_transform=None):
        dset = CIFAR(is_cifar10, is_train, transform=transform, target_transform=target_transform)
        super().__init__(dset)
        self.dset_name = f'indexed-{self.dataset.dset_name}'
        self.split_name = self.dataset.split_name

    def write_ffcv_beton(self, save_dir, file_name=None, num_workers=4):
        save_dir = pathlib.Path(save_dir)
        assert save_dir.exists()

        if file_name:
            assert file_name.endswith('.beton')
        else:
            file_name = f'{self.dset_name}_{self.split_name}.beton'

        dset_path = save_dir / file_name

        writer = DatasetWriter(str(dset_path), {
            'image': RGBImageField(),
            'label': IntField(),
            'index': IntField()
        }, num_workers=num_workers)

        writer.from_indexed_dataset(self)
        return dset_path

# Waterbirds datasets

class Waterbirds(DatasetWithLoader):
    """
    Waterbirds dataset (Birds x Places)
        Used in subpopulation robustness lit.
        Two classes: {water bird, land bird}
        Two domains/metadata: {water bg, land bg}
        Most water (land) birds on water (land) bg (~95%)
    """

    ROOT_DIR = DATA_DIR

    def __init__(self, data_split,  get_metadata=1, transform=None, root_dir=None):
        """
        Args
            data split: {train, val, test}
            transform: data augmentation fn
            get_metadata:
                0: (x,y)
                1: (x, y, z)
                2: (x, y, (z, y, split_id)) # raw
        """
        assert data_split in {'train', 'test', 'val'}, "invalid split name"

        super().__init__()
        self.dset_name = 'waterbirds'
        self.data_split =  data_split
        self.transform = transform
        self.get_metadata = get_metadata>0
        self.get_raw_metadata = get_metadata==2

        # setup dataset
        self.root_dir = Waterbirds.ROOT_DIR if root_dir is None else root_dir
        self.root_dir = pathlib.Path(self.root_dir)
        assert self.root_dir.exists()
        self.dataset = wilds.get_dataset('waterbirds', download=True,
                                         root_dir=self.root_dir)

        self.dataset = self.dataset.get_subset(self.data_split,
                                               transform=self.transform)
        self.collate = self.dataset.collate

        self.metadata_fields = ['background', 'label']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        x, y, (z, y_, split_id) = self.dataset[idx] # uses self.transform
        assert y==y_

        if self.get_metadata:
            if self.get_raw_metadata:
                return x, y, (z, y, split_id)
            else:
                return x, y, z

        return x, y

    def get_loader(self, **loader_kws):
        is_train = self.data_split == 'train'
        loader_fn = get_wilds_train_loader if is_train else get_wilds_eval_loader
        return loader_fn('standard', self, **loader_kws)

    def write_ffcv_beton(self, save_dir, file_name=None, num_workers=4):
        assert self.get_metadata and not self.get_raw_metadata
        save_dir = pathlib.Path(save_dir)
        assert save_dir.exists()

        if file_name:
            assert file_name.endswith('.beton')
        else:
            file_name = f'{self.dset_name}_{self.data_split}.beton'

        dset_path = save_dir / file_name

        writer = DatasetWriter(str(dset_path), {
            'image': RGBImageField(),
            'label': IntField(),
            'group': IntField()
        }, num_workers=num_workers)

        writer.from_indexed_dataset(self)
        return dset_path

    def evaluate(self, y_pred, y_true, metadata):
        """
        - y_pred: array of predictions
        - y_true: array of ground truth labels
        - metadata: array of metadata [group_id clas_id split_id]
        """
        return self.dataset.eval(y_pred, y_true, metadata)

class IndexedWaterbirds(IndexedDataset):

    def __init__(self, *args, **kwargs):
        dset = Waterbirds(*args, **kwargs)
        super().__init__(dset)
        self.dset_name = f'indexed-{self.dataset.dset_name}'
        self.data_split = self.dataset.data_split

    def write_ffcv_beton(self, save_dir, file_name=None, num_workers=4):
        assert self.dataset.get_metadata and not self.dataset.get_raw_metadata
        save_dir = pathlib.Path(save_dir)
        assert save_dir.exists()

        if file_name:
            assert file_name.endswith('.beton')
        else:
            file_name = f'{self.dset_name}_{self.data_split}.beton'

        dset_path = save_dir / file_name

        writer = DatasetWriter(str(dset_path), {
            'image': RGBImageField(),
            'label': IntField(),
            'group': IntField(),
            'index': IntField()
        }, num_workers=num_workers)

        writer.from_indexed_dataset(self)
        return dset_path

class WaterbirdsIndicesSubset(Waterbirds):

    def __init__(self, data_split, indices, get_metadata=1, transform=None):
        super().__init__(data_split,
                         get_metadata=get_metadata,
                         transform=transform)

        self.indices = indices
        self.dataset = Subset(self.dataset, self.indices)

class WaterbirdsGroupSubset(WaterbirdsIndicesSubset):

    def __init__(self, data_split, groups, get_metadata=1,
                 transform=None):

        self.data_split = data_split
        self.groups = groups
        self.indices = self._get_indices()

        super().__init__(data_split,
                         self.indices,
                         get_metadata=get_metadata,
                         transform=transform)

    def _get_indices(self):
        labels_path = os.path.join(self.ROOT_DIR,
                                   'proc_data/waterbirds_labels_v2.pkl')
        labels = torch.load(labels_path)

        indices = []
        birds = labels[self.data_split]['bird']
        backgrounds = labels[self.data_split]['background']
        groups_set = set(self.groups)

        for idx, (bird, bg) in enumerate(zip(birds, backgrounds)):
            if (bird,bg) not in groups_set: continue
            indices.append(idx)

        return indices

class WaterbirdsMajority(WaterbirdsGroupSubset):

    def __init__(self, data_split, get_metadata=1, transform=None):
        super().__init__(data_split,
                         groups=[(0,0),(1,1)],
                         get_metadata=get_metadata,
                         transform=transform)

class WaterbirdsMinority(WaterbirdsGroupSubset):

    def __init__(self, data_split, get_metadata=1, transform=None):
        super().__init__(data_split,
                         groups=[(0,1),(1,0)],
                         get_metadata=get_metadata,
                         transform=transform)

# ImageNet-related

class Living17(DatasetWithLoader):

    ROOT_DIR = Path(DATA_DIR) / 'living17' # change this to imagenet parent directory

    def __init__(self, split, transform, get_metadata=0, root_dir=None, imagenet_dir=None):
        """
        - split: {train, val}
        - transform: data aug func
        - metadata
            - 0: (image, class)
            - 1: (image, class, imagenet_class)
            - 2: (image, class, imagenet_class, imagenet_index)
        """
        assert split in {'train', 'val'}
        assert get_metadata in {0,1,2}

        # setup
        super().__init__()
        self.split = split
        self.transform = transform
        self.metadata_mode = get_metadata
        self.imagenet_dir = self.ROOT_DIR / 'imagenet' if imagenet_dir is None else Path(imagenet_dir)
        assert self.imagenet_dir.exists(), "invalid imagenet parent directory"

        # metadata
        df = self.df = pd.read_pickle(CUR_FILE_DIR / 'metadata' / f'{self.split}_metadata.df')
        df['class_name'] = df['class_name'].apply(lambda c: c.split(',')[0])
        indices = df.index.tolist()

        self.in_index_map = dict(zip(indices, df['IN:index']))
        self.subclass_map = dict(zip(df['IN:class'], df['class']))

        self.class_maps = { # index -> class ID
            'living17': dict(zip(indices, df['class'])),
            'imagenet': dict(zip(indices, df['IN:class']))
        }

        self.classname_maps = { # class ID -> class name
            'living17': dict(zip(df['class'], df['class_name'])),
            'imagenet': dict(zip(df['IN:class'], df['IN:class_name']))
        }

        # dataset
        self.dset = torchvision.datasets.ImageFolder(self.imagenet_dir / self.split, transform=self.transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)): raise IndexError
        in_idx = self.in_index_map[idx]
        img, in_class = self.dset[in_idx]
        l17_class = self.class_maps['living17'][idx]

        if self.metadata_mode==0:
            return img, l17_class

        if self.metadata_mode==1:
            return img, l17_class, in_class

        if self.metadata_mode==2:
            return img, l17_class, in_class, in_idx