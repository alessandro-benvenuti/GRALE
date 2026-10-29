"""Prepared XY graphs, stored as NPZ targets in a manifest/ZIP archive."""

import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.sparse.csgraph import shortest_path
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from .dense_data import DenseData
from .dataset import custom_collate_fn


class GeometricDataset(Dataset):
    def __init__(self, manifest, split='train', n_nodes_max=64):
        self.path = Path(manifest).resolve()
        manifest = json.loads(self.path.read_text())
        split = 'validation' if split == 'valid' else split
        self.samples = [row for row in manifest['samples'] if row['split'] == split]
        self.root = Path(manifest.get('data_root', self.path.parent))
        self.n_nodes_max = n_nodes_max
        self.archives = {}
        self.pid = os.getpid()
        if not self.samples:
            raise ValueError(f'No {split} samples in {self.path}')
        if max(row['nodes'] for row in self.samples) > n_nodes_max:
            raise ValueError('Graph exceeds n_nodes_max; targets must not be truncated')

    def __len__(self):
        return len(self.samples)

    def load_target(self, index):
        row = self.samples[index]
        archive_path = self.root / row['target_archive']
        # Each DataLoader process needs its own file handle.
        if self.pid != os.getpid():
            self.archives = {}
            self.pid = os.getpid()
        if archive_path not in self.archives:
            self.archives[archive_path] = ZipFile(archive_path)
        content = self.archives[archive_path].read(row['target'])
        with np.load(io.BytesIO(content), allow_pickle=False) as target:
            return {key: target[key] for key in (
                'nodes', 'edges', 'metric', 'shape_hw', 'full_paths', 'path_offsets',
            )}

    def __getitem__(self, index):
        target = self.load_target(index)
        xy = torch.as_tensor(target['nodes'], dtype=torch.float32)
        edges = torch.as_tensor(target['edges'], dtype=torch.long).reshape(-1, 2)
        n = len(xy)
        N = self.n_nodes_max
        if xy.shape != (n, 2) or n > N:
            raise ValueError('Expected XY nodes fitting n_nodes_max')
        h = torch.arange(N) < n
        features = torch.zeros(N, 2)
        features[:n] = xy
        adjacency = torch.zeros(N, N)
        if edges.numel():
            if edges.min() < 0 or edges.max() >= n:
                raise ValueError('Invalid edge endpoints')
            adjacency[edges[:, 0], edges[:, 1]] = 1
            adjacency[edges[:, 1], edges[:, 0]] = 1
        distances = torch.full((N, N), float(N))
        if n:
            sp = shortest_path(adjacency[:n, :n].numpy(), directed=False, unweighted=True)
            sp[~np.isfinite(sp)] = N
            distances[:n, :n] = torch.as_tensor(sp, dtype=torch.float32)
        labels = torch.nn.functional.one_hot(adjacency.long(), num_classes=2).float()
        return DenseData(N, h, nodes={'features': features}, edges={
            'adjacency': adjacency, 'labels': labels, 'SP': distances,
        })


class GeometricDataModule(pl.LightningDataModule):
    def __init__(self, manifest, n_nodes_max=64, batch_size=4, n_workers=0, seed=1234):
        super().__init__()
        self.dataset_train = GeometricDataset(manifest, 'train', n_nodes_max)
        self.dataset_valid = GeometricDataset(manifest, 'valid', n_nodes_max)
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.seed = seed

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.n_workers, collate_fn=custom_collate_fn,
                          generator=torch.Generator().manual_seed(self.seed))

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size,
                          num_workers=self.n_workers, collate_fn=custom_collate_fn)
