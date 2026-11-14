import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


def build_transform(cfg, is_train=True):
    if is_train:
        transform = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(size=cfg.size, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg.mean, std=cfg.std),
            v2.RandomErasing(),
        ])
    else:
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((cfg.size, cfg.size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg.mean, std=cfg.std),
        ])
    return transform


def build_loader(cfg, dataset, shuffle=None, drop_last=False):
    if shuffle is None:
        shuffle = cfg.shuffle
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, drop_last=drop_last)


def download_cifar10(cfg, transform, is_train=True, download=True):
    return CIFAR10(root=cfg.data_dir, train=is_train,
                   transform=transform, download=download)
