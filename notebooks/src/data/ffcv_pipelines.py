from functools import partial
import torch
from ffcv.fields import decoders as DC
from ffcv import transforms as TF
import torchvision.transforms as T

DS_TO_MEAN = {
    'CIFAR': [125.307, 122.961, 113.8575],
    'WATERBIRDS': [123.675, 116.28, 103.53],
    'LIVING17': [0.485*255, 0.456*255, 0.406*255],
}

DS_TO_STD = {
    'CIFAR': [51.5865, 50.847 , 51.255],
    'WATERBIRDS': [58.395, 57.12 , 57.375],
    'LIVING17': [0.229*255, 0.224*255, 0.225*255],
}

INT_LABEL_PIPELINE = lambda device: [DC.IntDecoder(), TF.ToTensor(), TF.ToDevice(device), TF.Squeeze()]
FLOAT_LABEL_PIPELINE = lambda device: [DC.FloatDecoder(), TF.ToTensor(), TF.ToDevice(device), TF.Squeeze()]

IMAGE_PIPELINES = {
    'cifar': {
        'train': lambda device: [ # ffcv
            get_image_decoder('simple', 32),
            TF.RandomHorizontalFlip(),
            TF.RandomTranslate(padding=2, fill=(0,0,0)),
            TF.Cutout(4, tuple(map(int, DS_TO_MEAN['CIFAR']))),
            *get_standard_image_pipeline(device,  DS_TO_MEAN['CIFAR'], DS_TO_STD['CIFAR'])
        ],
        'flip': lambda device: [ # patch+tint+binary
            get_image_decoder('simple', 32),
            TF.RandomHorizontalFlip(),
            *get_standard_image_pipeline(device,  DS_TO_MEAN['CIFAR'], DS_TO_STD['CIFAR'])
        ],
        'train_alt': lambda device: [ # without cutout
            get_image_decoder('simple', 32),
            TF.RandomHorizontalFlip(),
            TF.RandomTranslate(padding=2, fill=(0,0,0)),
            *get_standard_image_pipeline(device, DS_TO_MEAN['CIFAR'], DS_TO_STD['CIFAR'])
        ],
        'test': lambda device: [
            get_image_decoder('simple', 32),
            *get_standard_image_pipeline(device, DS_TO_MEAN['CIFAR'], DS_TO_STD['CIFAR'])
        ]
    },
    'waterbirds': {
        'train': lambda device: [ # sagawa numbers
            get_image_decoder('resized', 224),
            *get_standard_image_pipeline(device, DS_TO_MEAN['WATERBIRDS'], DS_TO_STD['WATERBIRDS'])
        ],
        'train_creager': lambda device: [ # +: center crop
            get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
            *get_standard_image_pipeline(device, DS_TO_MEAN['WATERBIRDS'], DS_TO_STD['WATERBIRDS'])
        ],
        'train_heavyaug': lambda device: [
            get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
            TF.RandomHorizontalFlip(),
            TF.RandomTranslate(padding=14, fill=(0,0,0)),
            TF.Cutout(28, tuple(map(int, DS_TO_MEAN['WATERBIRDS']))),
            *get_standard_image_pipeline(device, DS_TO_MEAN['WATERBIRDS'], DS_TO_STD['WATERBIRDS'])
        ],
        'test': lambda device: [
            get_image_decoder('resized', 224),
            *get_standard_image_pipeline(device, DS_TO_MEAN['WATERBIRDS'], DS_TO_STD['WATERBIRDS'])
        ]
    },
    'living17': {
        'train_aug': lambda device: [
            get_image_decoder('random_resized_crop', 224),
            TF.RandomHorizontalFlip(),
            *get_standard_image_pipeline(device, DS_TO_MEAN['LIVING17'], DS_TO_STD['LIVING17'])
        ],
        'train_no_aug': lambda device: [
            get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
            *get_standard_image_pipeline(device, DS_TO_MEAN['LIVING17'], DS_TO_STD['LIVING17'])
        ],
        'val': lambda device: [
            get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
            *get_standard_image_pipeline(device, DS_TO_MEAN['LIVING17'], DS_TO_STD['LIVING17'])
        ]
    }
}

def get_image_decoder(decoder_name, image_size, center_crop_ratio=224/256.):
    img_decoders = {
        'simple': lambda sz: DC.SimpleRGBImageDecoder(),
        'resized': lambda sz: DC.CenterCropRGBImageDecoder((sz,sz),1),
        'center_crop': lambda sz,rt: DC.CenterCropRGBImageDecoder((sz,sz),rt),
        'random_resized_crop': lambda sz: DC.RandomResizedCropRGBImageDecoder((sz,sz)),
    }

    assert decoder_name.lower() in img_decoders
    img_decoders['center_crop'] = partial(img_decoders['center_crop'], rt=center_crop_ratio)

    return img_decoders[decoder_name](image_size)

def get_standard_image_pipeline(device, mean, std):
    return [
        TF.ToTensor(),
        TF.ToDevice(device),
        TF.ToTorchImage(),
        TF.Convert(torch.float16),
        T.Normalize(mean, std)
    ]

def get_pipelines(dataset_name, aug_name, device):
    proxy_dset_map = {
        'cifar10': 'cifar',
        'waterbirds': 'waterbirds',
        'living17': 'living17',
    }

    dataset_name, aug_name = dataset_name.lower(), aug_name.lower()
    proxy_dataset_name = proxy_dset_map[dataset_name]

    assert proxy_dataset_name in IMAGE_PIPELINES
    assert aug_name in IMAGE_PIPELINES[proxy_dataset_name], 'aug_name: {}'.format(aug_name)

    img_pipeline = IMAGE_PIPELINES[proxy_dataset_name][aug_name]

    if dataset_name == 'cifar10':
        base = {
            'image': img_pipeline(device),
            'label': INT_LABEL_PIPELINE(device)
        }

        if 'indexed' in dataset_name:
            base['index'] = INT_LABEL_PIPELINE(device)

        return base

    elif dataset_name == 'waterbirds':
        base = {
            'image': img_pipeline(device),
            'label': INT_LABEL_PIPELINE(device),
            'group': INT_LABEL_PIPELINE(device)
        }

        if 'indexed' in dataset_name:
            base['index'] = INT_LABEL_PIPELINE(device)

        return base

    elif dataset_name == 'living17':
        base = {
            'image': img_pipeline(device),
            'label': INT_LABEL_PIPELINE(device),
            'orig_label': INT_LABEL_PIPELINE(device),
            'index': INT_LABEL_PIPELINE(device)
        }

        return base

    else:
        raise NotImplementedError