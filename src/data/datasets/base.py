"""Base dataset logic shared by scene-level data loaders."""

import copy
import json
import os

import jsonlines
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..data_utils import load_safetensor_from_hf


class ScanBase(Dataset):
    """Base dataset for scene-level pretraining data."""

    def __init__(self, cfg, split):
        """Store common dataset config and split state."""
        self.cfg = cfg
        self.split = split
        self.debug = cfg.debug.flag
        self.debug_size = cfg.debug.debug_size

        assert self.split in ['pretrain', 'train', 'val', 'test']
        if self.split == 'test':
            self.split = 'val'

        self.use_scene_cap = getattr(cfg.data.args, 'use_scene_cap', False)

    def _load_split(self, split):
        """Load the split file and return sorted scan ids."""
        if 'scannet' in self.__class__.__name__.lower():
            split_file = os.path.join(self.base_dir, 'annotations/splits/scannetv2_' + split + '.txt')
        else:
            split_file = os.path.join(self.base_dir, 'annotations/splits/' + split + '_split.txt')

        scan_ids = {x.strip() for x in open(split_file, 'r', encoding='utf-8')}
        return sorted(scan_ids)

    def _load_scan_pretrain(self, data, ground_lang_data):
        """Group per-view language entries into per-scene pretrain samples."""
        scans = []
        scan_id_to_idx = {}

        for i, item in enumerate(data):
            scan_id = item['scan_id']
            refer_item = ground_lang_data[i]
            if scan_id not in scan_id_to_idx:
                scan_entry = {
                    'scan_id': scan_id,
                    'safetensors_path': item['safetensors_path'],
                    'sentence': [],
                    'refer_sentence': [],
                }
                scans.append(scan_entry)
                scan_id_to_idx[scan_id] = len(scans) - 1

            scan = scans[scan_id_to_idx[scan_id]]
            scan['sentence'].append(item['utterance'][:5])
            if len(refer_item['utterance']) > 0:
                scan['refer_sentence'].append(refer_item['utterance'])
            else:
                scan['refer_sentence'].append(item['utterance'][:5])
        return scans

    def _load_lang(self, cfg, scan_ids):
        """Load language annotations for the requested scan ids."""
        caption_source = cfg.sources
        lang_data = []
        ground_lang_data = []
        valid_scan_ids = set()

        if self.use_scene_cap:
            scene_cap_file = os.path.join(self.base_dir, 'annotations/scene_cap.json')
            if not os.path.exists(scene_cap_file):
                self.scene_caps = {}
            else:
                with open(scene_cap_file, 'r', encoding='utf-8') as f:
                    self.scene_caps = json.load(f)
        else:
            self.scene_caps = None

        for anno_type in caption_source:
            if anno_type == 'scannet_view_cap':
                anno_file = os.path.join(self.base_dir, 'annotations/per_view_captions_scannet_v1.jsonl')
                with jsonlines.open(anno_file, 'r') as handle:
                    for item in handle:
                        if item['scan_id'] in scan_ids:
                            lang_data.append(item)
                            valid_scan_ids.add(item['scan_id'])
            elif anno_type == '3rscan_view_cap':
                anno_file = os.path.join(self.base_dir, 'annotations/per_view_captions_3rscan_v1.jsonl')
                with jsonlines.open(anno_file, 'r') as handle:
                    for item in handle:
                        if item['scan_id'] in scan_ids:
                            lang_data.append(item)
                            valid_scan_ids.add(item['scan_id'])
            elif anno_type == 'arkitscenes_view_cap':
                anno_file = os.path.join(self.base_dir, 'annotations/per_view_captions_arkitscenes_v1.jsonl')
                with jsonlines.open(anno_file, 'r') as handle:
                    for item in handle:
                        if item['scan_id'] in scan_ids:
                            lang_data.append(item)
                            valid_scan_ids.add(item['scan_id'])
            elif 'ssg_ref' in anno_type:
                refer_file = os.path.join('dataset/refer', f'{anno_type}.jsonl')
                with jsonlines.open(refer_file, 'r') as handle:
                    for item in handle:
                        if item['scan_id'] in scan_ids:
                            ground_lang_data.append(item)
                            valid_scan_ids.add(item['scan_id'])

        if len(ground_lang_data) == 0:
            for item in lang_data:
                ground_lang_data.append(copy.deepcopy(item))

        return lang_data, ground_lang_data, sorted(valid_scan_ids)

    def _getitem_refer(self, index):
        """Return one pretraining sample with views and language."""
        if self.cfg.mode != 'pretrain':
            raise RuntimeError('Spatial refer downstream mode was removed because it is unused in this repo.')

        item = self.scan_data[index]
        scan_id = item['scan_id']
        scene_tensor = load_safetensor_from_hf(
            repo_id='MatchLab/ScenePoint',
            filename=item['safetensors_path'],
        )

        out = {
            'point_map': scene_tensor['point_map'],
            'sentence': item['sentence'],
            'refer_sentence': item['refer_sentence'],
            'images': F.interpolate(
                scene_tensor['color_images'].permute(0, 3, 1, 2),
                size=(224, 224),
                mode='bilinear',
                align_corners=False,
            ),
            'scan_id': scan_id,
        }

        if self.use_scene_cap:
            scene_caps = self.scene_caps.get(scan_id)
            if scene_caps is not None:
                captions = scene_caps['captions']
                scene_cap = captions[np.random.choice(len(captions))]
            else:
                scene_cap = 'This is a scene.'
            out['scene_cap'] = scene_cap

        return out
