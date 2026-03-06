"""Custom PyTorch Dataset for sharded .npy files with lazy memory-mapped loading.

Designed for multi-worker DataLoader: mmap handles are opened lazily on first
__getitem__ so the Dataset object is safely picklable across process boundaries.
"""

import bisect
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ShardedNpyDataset(Dataset):
    """Reads samples from sharded .npy files using memory-mapped I/O.

    Each shard is a pair of .npy files (features + labels) discovered via manifest.json.
    Uses np.load(mmap_mode='r') so only the OS pages containing the requested sample
    are faulted into memory — not the entire shard.
    """

    def __init__(self, data_dir: str):
        manifest_path = os.path.join(data_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"No manifest.json in {data_dir}. Run generate_data.py first."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.data_dir = data_dir
        self.n_samples = manifest["total_samples"]
        self.shards = manifest["shards"]

        # Cumulative sample offsets for O(log N) index → shard mapping
        self._offsets = []
        running = 0
        for s in self.shards:
            self._offsets.append(running)
            running += s["n_samples"]

        # Populated lazily per-worker (not picklable → must init after spawn)
        self._feat_mmaps: dict[int, np.ndarray] = {}
        self._label_mmaps: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.n_samples

    def _ensure_mmap(self, shard_idx: int):
        if shard_idx not in self._feat_mmaps:
            s = self.shards[shard_idx]
            self._feat_mmaps[shard_idx] = np.load(
                os.path.join(self.data_dir, s["features_file"]), mmap_mode="r"
            )
            self._label_mmaps[shard_idx] = np.load(
                os.path.join(self.data_dir, s["labels_file"]), mmap_mode="r"
            )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        shard_idx = bisect.bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[shard_idx]

        self._ensure_mmap(shard_idx)

        features = torch.from_numpy(self._feat_mmaps[shard_idx][local_idx].copy())
        label = int(self._label_mmaps[shard_idx][local_idx])
        return features, label
