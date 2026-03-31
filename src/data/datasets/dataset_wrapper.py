"""Dataset wrappers for view padding and tokenization."""

import random
from pathlib import Path

import torch
from fvcore.common.registry import Registry
from transformers import AutoImageProcessor, AutoTokenizer
from torch.utils.data import Dataset


DATASETWRAPPER_REGISTRY = Registry("dataset_wrapper")
DATASETWRAPPER_REGISTRY.__doc__ = """ """


@DATASETWRAPPER_REGISTRY.register()
class SceneDatasetWrapper(Dataset):
    """Wrap pretraining scenes with tokenization and view sampling."""

    def __init__(self, cfg, dataset, split="train"):
        """Build the pretraining dataset wrapper."""
        self.dataset = dataset
        self.num_views = cfg.get("num_views", 32)
        model_root = str(Path(__file__).resolve().parents[2] / "fg-clip")
        self.tokenizer = AutoTokenizer.from_pretrained(model_root)
        self.image_processor = AutoImageProcessor.from_pretrained(model_root)
        self.use_scene_cap = cfg.data.args.get("use_scene_cap", False)

    def __len__(self):
        """Return the wrapped dataset size."""
        return len(self.dataset)

    def _build_view_indices(self, num_views):
        """Sample or pad view indices to the configured view count."""
        if num_views == self.num_views:
            return torch.arange(num_views)

        if num_views > self.num_views:
            return torch.randperm(num_views)[:self.num_views]

        pad_indices = torch.randint(0, num_views, (self.num_views - num_views,))
        return torch.cat([torch.arange(num_views), pad_indices], dim=0)

    def __getitem__(self, idx):
        """Return one wrapped pretraining sample."""
        base = self.dataset[idx]
        out = {}
        view_indices = self._build_view_indices(base['point_map'].shape[0])
        view_indices_list = view_indices.tolist()
        sentence_views = [base['sentence'][i] for i in view_indices_list]
        refer_sentence_views = [base['refer_sentence'][i] for i in view_indices_list]
        encoded_input = self.tokenizer(
            [random.choice(sens) for sens in sentence_views],
            max_length=77, padding="max_length", truncation=True, return_tensors='pt'
        )

        out['txt_ids'] = encoded_input.input_ids.squeeze(0)

        refer_encoded_input = self.tokenizer(
            [random.choice(refer_sens) for refer_sens in refer_sentence_views],
            max_length=77, padding="max_length", truncation=True, return_tensors='pt'
        )

        out['ground_txt_ids'] = refer_encoded_input.input_ids.squeeze(0)

        out['images'] = self.image_processor(
            base['images'][view_indices],
            do_center_crop=False,
            do_resize=True,
            size={"height": 224, "width": 224},
            return_tensors='pt'
        )['pixel_values'].squeeze(0)

        if self.use_scene_cap:
            enc_scene = self.tokenizer(
                base['scene_cap'], max_length=248,
                padding="max_length", truncation=True, return_tensors='pt'
            )
            out['scene_txt_ids'] = enc_scene.input_ids.squeeze(0)

        out['point_map'] = base['point_map'][view_indices].contiguous().clone()
        out['scan_id'] = base['scan_id']
        return out

    def collate_fn(self, batch_list):
        """Collate wrapped pretraining samples into a batch."""
        collated = {}
        keys = batch_list[0].keys()
        for key in keys:
            values = [sample[key] for sample in batch_list]
            if torch.is_tensor(values[0]):
                collated[key] = torch.stack([value.contiguous().clone() for value in values], dim=0)
            else:
                collated[key] = values
        return collated


@DATASETWRAPPER_REGISTRY.register()
class ScanFamilyDatasetWrapperQA(Dataset):
    """Wrap downstream QA samples with aligned view counts."""

    def __init__(self, cfg, dataset, split="train"):
        """Build the QA dataset wrapper."""
        self.dataset = dataset
        self.num_views = cfg.get("num_views", 32)
        model_root = str(Path(__file__).resolve().parents[2] / "fg-clip")
        self.image_processor = AutoImageProcessor.from_pretrained(model_root)

    def __len__(self):
        """Return the wrapped dataset size."""
        return len(self.dataset)

    def _build_view_indices(self, num_views):
        """Sample or pad view indices to the configured view count."""
        if num_views == self.num_views:
            return torch.arange(num_views)

        if num_views > self.num_views:
            return torch.randperm(num_views)[:self.num_views]

        pad_indices = torch.randint(0, num_views, (self.num_views - num_views,))
        return torch.cat([torch.arange(num_views), pad_indices], dim=0)

    def __getitem__(self, idx):
        """Return one wrapped QA sample."""
        base = self.dataset[idx]
        out = {}
        for key, value in base.items():
            if torch.is_tensor(value):
                out[key] = value.contiguous().clone()
            else:
                out[key] = value

        view_source = None
        if torch.is_tensor(out.get('point_map')):
            view_source = out['point_map']
        elif torch.is_tensor(out.get('images')):
            view_source = out['images']

        if view_source is not None:
            view_indices = self._build_view_indices(view_source.shape[0])
            if torch.is_tensor(out.get('point_map')):
                out['point_map'] = out['point_map'][view_indices].contiguous().clone()
            if torch.is_tensor(out.get('images')):
                out['images'] = out['images'][view_indices].contiguous().clone()
                out['images'] = self.image_processor(
                    out['images'],
                    do_center_crop=False,
                    do_resize=True,
                    size={"height": 224, "width": 224},
                    return_tensors='pt'
                )['pixel_values'].squeeze(0).contiguous().clone()

        if torch.is_tensor(out.get('answer_label')):
            out['answer_label'] = out['answer_label'].contiguous().clone()

        return out

    def collate_fn(self, batch_list):
        """Collate wrapped QA samples into a batch."""
        collated = {}
        keys = batch_list[0].keys()
        for key in keys:
            values = [sample[key] for sample in batch_list]
            if torch.is_tensor(values[0]):
                collated[key] = torch.stack([value.contiguous().clone() for value in values], dim=0)
            else:
                collated[key] = values
        return collated
