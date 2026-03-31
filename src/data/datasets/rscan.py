"""3RScan pretraining dataset wrapper."""

from ..build import DATASET_REGISTRY
from .base import ScanBase


@DATASET_REGISTRY.register()
class RScanSpatialRefer(ScanBase):
    """3RScan spatial-refer dataset used for pretraining."""

    def __init__(self, cfg, split):
        """Load 3RScan language data and scan metadata."""
        super(RScanSpatialRefer, self).__init__(cfg, split)
        self.base_dir = cfg.data.rscan_base

        split_cfg = cfg.data.get(self.__class__.__name__).get(split)
        all_scan_ids = self._load_split(self.split)

        self.lang_data, self.ground_lang_data, self.scan_ids = self._load_lang(split_cfg, all_scan_ids)
        self.scan_data = self._load_scan_pretrain(self.lang_data, self.ground_lang_data)

        print(f"Loading 3RScan {split}-set scans")
        print(f"Finish loading 3RScan {split}-set scans")

    def __len__(self):
        """Return the number of scene-level samples."""
        return len(self.scan_data)

    def __getitem__(self, index):
        """Return one pretraining sample."""
        return self._getitem_refer(index)
