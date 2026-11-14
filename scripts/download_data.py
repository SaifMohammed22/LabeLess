"""
Download CIFAR 10 dataset locally into data directory
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
print(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from labeless.dataset import download_cifar10, build_transform


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ROOT / "configs/dataset.yaml")
    download = cfg.get("download", True)
    train_transform = build_transform(cfg, is_train=True)
    test_transform = build_transform(cfg, is_train=False)
    train_dataset = download_cifar10(cfg, train_transform, is_train=True, download=download)
    test_dataset = download_cifar10(cfg, test_transform, is_train=False, download=download)
    print(f"train samples: {len(train_dataset)}")
    print(f"test samples: {len(test_dataset)}")