"""Tiny CPU smoke check using the upstream COLORING sampler and GRALE model.

Run from the repository root with its dependencies available:
    python scripts/smoke_coloring.py
Generated data and checkpoints use the existing ignored data/ and logs/ paths.
"""

import ast
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightning.pytorch as pl
import numpy as np
import torch
import yaml

from datasets.coloring.utils import coloring_sample
from datasets.utils_h5 import H5DatasetBuilder
from GRALE.data.dataset import DataModule
from GRALE.main import GRALE_model


class FiniteGradients(pl.Callback):
    def __init__(self):
        self.backward_steps = 0

    def on_after_backward(self, trainer, model):
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        self.backward_steps += 1


def main():
    started = time.time()
    for source in [*Path('GRALE').rglob('*.py'), *Path('datasets').rglob('*.py'),
                   Path('train.py'), Path('prepare_data.py')]:
        ast.parse(source.read_text(), filename=str(source))

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.set_num_threads(1)
    config = yaml.safe_load(Path('GRALE/configs/demo_checkpoint.yaml').read_text())
    data_config = json.loads(Path('datasets/coloring/configs/medium.json').read_text())
    sizes = {'train': 8, 'valid': 4, 'test': 2}
    path_h5 = Path('data/h5/COLORING_smoke.h5')
    output = Path('logs/coloring-smoke')
    path_h5.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    dataset = H5DatasetBuilder.create(
        str(path_h5), n_max_nodes=data_config['n_max_nodes'],
        node_labels=['BLUE', 'GREEN', 'RED', 'YELLOW'],
        edge_labels=['NOEDGE', 'EDGE'], chunk_size_storage=8, overwrite=True,
    )
    node_counts = {}
    for split, count in sizes.items():
        n_max = data_config['n_max_nodes']
        arrays = {
            'node_labels': np.zeros((count, n_max), dtype=np.uint8),
            'edge_labels': np.zeros((count, n_max, n_max), dtype=np.uint8),
            'node_mask': np.ones((count, n_max), dtype=bool),
            'SP_matrix': np.zeros((count, n_max, n_max), dtype=np.uint8),
        }
        node_counts[split] = []
        for i in range(count):
            _, graph = coloring_sample(
                n_min_nodes=data_config['n_min_nodes'], n_max_nodes=n_max,
                n_pixels=data_config['n_pixels'], sigma_filter=0.4,
                sigma_noise=0.02, node_weight_treshold=None,
                edge_weight_treshold=None, precompute_shortest_paths=True,
            )
            n = len(graph['node_colors'])
            node_counts[split].append(n)
            arrays['node_labels'][i, :n] = graph['node_colors']
            arrays['edge_labels'][i, :n, :n] = graph['adjacency_matrix']
            arrays['node_mask'][i, :n] = False
            arrays['SP_matrix'][i, :n, :n] = graph['SP_matrix']
        for field, values in arrays.items():
            dataset.append_data(f'{split}/{field}', values)

    # One worker keeps this bounded and satisfies upstream prefetch_factor usage.
    data = DataModule(str(path_h5), batch_size=2, n_data_epoch=4,
                      n_data_valid=4, n_workers=1)
    batch = next(iter(data.train_dataloader()))
    model = GRALE_model(config=config)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    gradient_check = FiniteGradients()
    trainer = pl.Trainer(
        accelerator='cpu', devices=1, max_steps=3,
        limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=str(output),
        gradient_clip_val=config['max_grad_norm'], precision='32-true',
        reload_dataloaders_every_n_epochs=1,
        callbacks=[gradient_check],
    )
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == 3
    assert all(torch.isfinite(p).all() for p in model.parameters())
    changed = sum(not torch.equal(initial[name], p.detach())
                  for name, p in model.named_parameters())
    assert changed > 0, 'No parameters changed during optimizer steps'
    assert gradient_check.backward_steps == 3
    model.eval()
    with torch.no_grad():
        latent = model.encode(batch.clone())
        decoded = model.decode(latent)
        reconstructed = model(batch.clone(), hard_matcher=True)
    expected = (2, config['n_nodes_max'])
    assert tuple(decoded.h.shape) == expected
    assert tuple(reconstructed.h.shape) == expected
    assert torch.isfinite(latent).all()
    for graph in (decoded, reconstructed):
        assert torch.isfinite(graph.h).all()
        assert all(torch.isfinite(v).all() for v in graph.nodes.values())
        assert all(torch.isfinite(v).all() for v in graph.edges.values())

    checkpoint = output / 'last.ckpt'
    trainer.save_checkpoint(str(checkpoint))
    restored = GRALE_model.load_from_checkpoint(str(checkpoint), map_location='cpu')
    restored.eval()
    with torch.no_grad():
        restored_latent = restored.encode(batch.clone())
    assert torch.allclose(latent, restored_latent)
    metrics = {name: float(value) for name, value in trainer.callback_metrics.items()}
    assert metrics and all(np.isfinite(value) for value in metrics.values())
    summary = {
        'status': 'passed', 'device': 'cpu', 'python': sys.version,
        'torch': torch.__version__, 'lightning': pl.__version__,
        'model_config': 'GRALE/configs/demo_checkpoint.yaml',
        'generator_config': 'datasets/coloring/configs/medium.json',
        'split_sizes': sizes, 'node_counts': node_counts,
        'batch_size': 2, 'n_data_epoch': 4, 'optimizer_steps': trainer.global_step,
        'finite_backward_steps': gradient_check.backward_steps,
        'changed_parameter_tensors': changed,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'latent_shape': list(latent.shape),
        'reconstruction_h_shape': list(reconstructed.h.shape),
        'checkpoint_reload_matches': True, 'metrics': metrics,
        'dataset': str(path_h5), 'checkpoint': str(checkpoint),
        'elapsed_seconds': time.time() - started,
    }
    (output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
