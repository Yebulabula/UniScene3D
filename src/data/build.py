"""Builds datasets and dataloaders from config."""

import torch
from torch.utils.data import DataLoader, default_collate, ConcatDataset, WeightedRandomSampler
from fvcore.common.registry import Registry

from .datasets.dataset_wrapper import DATASETWRAPPER_REGISTRY

DATASET_REGISTRY = Registry("dataset")
DATASET_REGISTRY.__doc__ = """
Registry for datasets, which takes a list of dataset names and returns a dataset object.
Currently it performs similar as registering dataset loading functions, but remains in a
form of object class for future purposes.
"""

def get_dataset(cfg, split):
    """Build the dataset object for one split."""
    assert cfg.data.get(split), f"No valid dataset name in {split}."
    dataset_list = []
    print(split, ': ', ', '.join(cfg.data.get(split)))
    for dataset_name in cfg.data.get(split):
        _dataset = DATASET_REGISTRY.get(dataset_name)(cfg, split)
        assert len(_dataset), f"Dataset '{dataset_name}' is empty!"
        wrapper = cfg.data_wrapper.get(split, cfg.data_wrapper) if not isinstance(cfg.data_wrapper, str) else cfg.data_wrapper
        _dataset = DATASETWRAPPER_REGISTRY.get(wrapper)(cfg, _dataset, split=split)
        dataset_list.append(_dataset)

    print('=' * 50)
    print('Dataset			Size')
    total = sum(len(dataset) for dataset in dataset_list)
    for dataset_name, dataset in zip(cfg.data.get(split), dataset_list):
        print(f'{dataset_name:<20} {len(dataset):>6} ({len(dataset) / total * 100:.1f}%)')
    print(f'Total			{total}')
    print('=' * 50)
    if split in ['pretrain', 'train']:
        # Training uses one concatenated dataset so sampling happens in one loader.
        dataset_list = ConcatDataset(dataset_list)
    return dataset_list


def build_dataloader(cfg, split='train'):
    """Build the dataloader for the requested split."""
    num_workers = cfg.dataloader.num_workers
    persistent_workers = num_workers > 0
    prefetch_factor = 4 if num_workers > 0 else None

    if split == 'pretrain':
        dataset = get_dataset(cfg, split)
        collate_fn = getattr(dataset.datasets[0], 'collate_fn', default_collate)
        sampler = None
        shuffle = True

        if isinstance(dataset, ConcatDataset) and cfg.dataloader.get("balance_dataset", False):
            # Give each source dataset the same total sampling mass.
            sample_weights = []
            for sub_dataset in dataset.datasets:
                dataset_weight = 1.0 / max(len(sub_dataset), 1)
                sample_weights.extend([dataset_weight] * len(sub_dataset))

            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=cfg.dataloader.batchsize,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=shuffle,
            sampler=sampler,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True,
        )

    loader_list = []
    if split == 'train':
        dataset = get_dataset(cfg, split)
        collate_fn = getattr(dataset.datasets[0], 'collate_fn', default_collate)
        return DataLoader(
            dataset,
            batch_size=cfg.dataloader.get('batchsize_eval', cfg.dataloader.batchsize),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=persistent_workers,
            drop_last=True,
            prefetch_factor=prefetch_factor,
            shuffle=True,
        )

    for dataset in get_dataset(cfg, split):
        collate_fn = getattr(dataset, 'collate_fn', default_collate)
        loader_list.append(
            DataLoader(
                dataset,
                batch_size=cfg.dataloader.get('batchsize_eval', cfg.dataloader.batchsize),
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                shuffle=False,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
            )
        )

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list
