import torchvision.transforms as T
import numpy as np
import torch
import random

def get_patch_transform(patch_rgb,
                        patch_size,
                        patch_loc,
                        patch_prob=1.,
                        mean=None,
                        std=None):
    """
    - patch_rgb_tuple: tuple of size 3 in [0,1] range
    - patch_size: patch square length
    - patch_location: 2D location wrt top left corner
    - patch_prob: probability of applying patch
    - mean/std: channel-wise stats (default: imagenet)
    """
    mean = mean if mean else np.array([0.485, 0.456, 0.406])
    std = std if std else np.array([0.229, 0.224, 0.225])
    rgb = (np.array(patch_rgb)-mean)/std

    x1, x2 = patch_loc[0], (patch_loc[0]+patch_size)
    y1, y2 = patch_loc[1], (patch_loc[1]+patch_size)

    def _transform(image):
        if random.random() < patch_prob:
            image = image.clone()
            for idx, val in enumerate(rgb):
                image[idx,x1:x2,y1:y2] = val
        return image

    return T.Lambda(_transform)

def get_tint_transform(rgb, prob=1.,
                       mean=None,
                       std=None):
    """
    - rgb_tuple: tuple of size 3 in [0,1] range
    - prob: probability of applying tint
    - mean/std: channel-wise stats (default: imagenet)
    """
    mean = mean if mean else np.array([0.485, 0.456, 0.406])
    std = std if std else np.array([0.229, 0.224, 0.225])

    rgb = ((np.array(rgb))/std)[:,None,None]
    max_rgb = ((np.array([1,1,1])-mean)/std)[:,None,None]
    min_rgb = ((np.array([0,0,0])-mean)/std)[:,None,None]
    rgb, max_rgb, min_rgb = map(torch.FloatTensor, [rgb, max_rgb, min_rgb])

    def _transform(image):
        if random.random() < prob:
            image = image.clone()
            image = torch.clamp(image+rgb, min=min_rgb, max=max_rgb)
        return image

    return T.Lambda(_transform)

MEAN_STDDEV_MAP = {
    'CIFAR': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'WATERBIRDS': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'IMAGENET': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'LIVING17':  ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}

# taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
CIFAR_TRANSFORMS = {
    'train':  T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['CIFAR'])
    ]),
    'test': T.Compose([
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['CIFAR'])
    ])
}

WATERBIRDS_TRANSFORMS = {
    'train': T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['WATERBIRDS'])
    ]),
    'test': T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['WATERBIRDS'])
    ])
}

LIVING17_TRANSFORMS = {
    'train_aug': T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['LIVING17'])
    ]),
    'train_no_aug': T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['LIVING17'])
    ]),
    'val': T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['LIVING17'])
    ])
}