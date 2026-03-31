"""Shared ScanNet scan-loading helpers."""

import os

from torch.utils.data import Dataset
from tqdm import tqdm


SCAN_DATA = {}


class ScanNetBase(Dataset):
    """Base class for ScanNet-family downstream datasets."""

    def __init__(self, cfg, split):
        """Store shared ScanNet dataset state."""
        self.cfg = cfg
        self.split = split
        self.base_dir = cfg.data.scan_family_base
        assert self.split in ['train', 'val', 'test']

    def __len__(self):
        """Return the dataset size."""
        return len(self.lang_data)

    def __getitem__(self, index):
        """Return one item from the dataset."""
        raise NotImplementedError

    def _load_one_scan(self, scan_id):
        """Resolve one scan id to its safetensor path."""
        if scan_id.startswith('scene'):
            one_scan = {'safetensors_path': f'light_scannet/{scan_id}.safetensors'}
        else:
            one_scan = {'safetensors_path': f'light_3rscan/{scan_id}.safetensors'}
        return scan_id, one_scan

    def _load_scannet(self, scan_ids, process_num=1):
        """Load scan metadata and reuse cached entries across datasets."""
        unloaded_scan_ids = [scan_id for scan_id in scan_ids if scan_id not in SCAN_DATA]
        print(f'Loading scans: {len(unloaded_scan_ids)} / {len(scan_ids)}')
        scans = {}
        if process_num > 1:
            from joblib import Parallel, delayed

            res_all = Parallel(n_jobs=process_num)(
                delayed(self._load_one_scan)(scan_id) for scan_id in tqdm(unloaded_scan_ids)
            )
            for scan_id, one_scan in tqdm(res_all):
                scans[scan_id] = one_scan
        else:
            for scan_id in tqdm(unloaded_scan_ids):
                _, one_scan = self._load_one_scan(scan_id)
                scans[scan_id] = one_scan

        SCAN_DATA.update(scans)
        return {scan_id: SCAN_DATA[scan_id] for scan_id in scan_ids}
