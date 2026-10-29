"""Bounded OCTA smoke, tiny-fit and held-out pilot runs. No images enter the model."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from GRALE.data.dataset import custom_collate_fn
from GRALE.data.geometry import GeometricDataModule
from GRALE.loss.geometry import decode_graph, score_graph, summarize_groups, chord_paths, path_distance
from GRALE.main import GRALE_model
from GRALE.model.utils import build_laplacian_node_pos

DEFAULT_MANIFEST = ROOT.parent / 'GNBM/gnbm-native-2d-20261027/experiments/native2d_long_20261028/octasynth/manifest.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select_rows(rows, count, seed, empty_count=None):
    rng = random.Random(seed)
    empty = [row for row in rows if row['nodes'] == 0]
    nonempty = [row for row in rows if row['nodes'] > 0]
    empty_count = round(count * len(empty) / len(rows)) if empty_count is None else empty_count
    if empty_count > len(empty) or count - empty_count > len(nonempty):
        raise ValueError('Not enough graphs for the requested subset')
    rng.shuffle(empty)
    groups = defaultdict(list)
    for row in nonempty:
        groups[(row['subject_id'], min(row['nodes'] // 8, 7))].append(row)
    for group in groups.values():
        rng.shuffle(group)
    keys = list(groups)
    rng.shuffle(keys)
    chosen = empty[:empty_count]
    while len(chosen) < count:
        for key in keys:
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop())
    rng.shuffle(chosen)
    return chosen


def prepare_subset(source, destination, mode, seed):
    source = Path(source).resolve()
    identity = digest(source)
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved['source_sha256'] != identity or saved['mode'] != mode or saved['seed'] != seed:
            raise ValueError('Frozen subset differs; use a new output directory')
        for name, expected in saved['archives_sha256'].items():
            if digest(Path(saved['data_root']) / name) != expected:
                raise ValueError('Target archive changed since subset selection')
        return saved
    original = json.loads(source.read_text())
    train = [row for row in original['samples'] if row['split'] == 'train']
    valid = [row for row in original['samples'] if row['split'] == 'validation']
    if {row['subject_id'] for row in train} & {row['subject_id'] for row in valid}:
        raise ValueError('Training and validation source identities overlap')
    sizes = {'smoke': (8, 4), 'tiny': (10, 4), 'pilot': (256, 128)}
    n_train, n_valid = sizes[mode]
    selected = select_rows(train, n_train, seed, 2 if mode in ('smoke', 'tiny') else None)
    if mode == 'smoke':
        # Exercise the observed large, small, disconnected and empty inputs.
        required = [max(train, key=lambda row: row['nodes']),
                    min((row for row in train if row['nodes']), key=lambda row: row['nodes']),
                    next(row for row in train if row['nodes'] > 1 and row['edges'] < row['nodes'] - 1)]
        for row in required:
            if row not in selected:
                index = next(i for i, old in enumerate(selected) if old['nodes'] and old not in required)
                selected[index] = row
    selected += select_rows(valid, n_valid, seed + 1, 1 if mode in ('smoke', 'tiny') else None)
    root = Path(original.get('data_root', source.parent)).resolve()
    archives = {row['target_archive'] for row in selected}
    saved = {'data_root': str(root), 'source_manifest': str(source),
             'source_sha256': identity, 'mode': mode, 'seed': seed,
             'annotation': 'supplied XY graph, contracted branches rendered as straight chords',
             'normalization': 'XY/[W,H]; pixel metric retained; no augmentation',
             'archives_sha256': {name: digest(root / name) for name in archives},
             'samples': selected}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(saved, indent=2) + '\n')
    return saved


class RuntimeCheck(pl.Callback):
    def __init__(self):
        self.backward_steps = 0
        self.started = None
        self.batch_seconds = []

    def on_train_start(self, trainer, model):
        self.started = time.time()

    def on_after_backward(self, trainer, model):
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Missing or nonfinite gradients')
        self.backward_steps += 1

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        if model.device.type == 'cuda':
            torch.cuda.synchronize()
        self.batch_started = time.time()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if model.device.type == 'cuda':
            torch.cuda.synchronize()
        self.batch_seconds.append(time.time() - self.batch_started)


def sanity(model, data, device):
    batch = next(iter(data.train_dataloader())).to(device)
    model.eval()
    with torch.no_grad():
        latent = model.encode(batch.clone())
        outputs = model.decode(latent)
        targets = model.target_builder(model.input_builder(batch.clone()))
        aligned = model(batch.clone(), hard_matcher=True, logits=True)
        loss, _ = model.training_objective(aligned, targets, None)
    if not torch.isfinite(latent).all() or not torch.isfinite(loss).all():
        raise RuntimeError('Nonfinite forward/loss')
    assert all(torch.isfinite(value).all() for value in [outputs.h, *outputs.nodes.values(), *outputs.edges.values()])
    for i, row in enumerate(data.dataset_train.samples):
        graph = data.dataset_train[i]
        target = data.dataset_train.load_target(i)
        assert np.allclose(graph.nodes.features[:row['nodes']].numpy() @ target['metric'].T,
                           target['nodes'] @ target['metric'].T, atol=1e-4)
        expected_edges = {tuple(sorted(edge)) for edge in target['edges'].tolist()}
        assert {tuple(edge) for edge in decode_graph(graph)['edges'].tolist()} == expected_edges
        assert score_graph(decode_graph(graph), decode_graph(graph))['exact']
    # An explicitly all-empty batch must work independently of sampler order.
    empty = custom_collate_fn([data.dataset_train[i] for i, row in enumerate(data.dataset_train.samples)
                              if row['nodes'] == 0][:2]).to(device)
    empty_targets = model.target_builder(empty)
    inputs = model.input_builder(empty.clone())
    embeddings, latent_empty = model.encoder(inputs)
    decoded_embeddings, output_empty = model.decoder(latent_empty)
    T, _ = model.matcher(embeddings, ~inputs.h, decoded_embeddings)
    loss_empty, logs = model.training_objective(output_empty, empty_targets, T)
    assert torch.isfinite(loss_empty).all()
    assert logs['loss_node_features'] == logs['loss_edge_features'] == logs['loss_adjacency'] == 0
    loss_empty.mean().backward()
    model.zero_grad(set_to_none=True)
    h = batch.h
    A = batch.edges.adjacency
    pe = build_laplacian_node_pos(A, 16, h=h)
    larger_A = torch.nn.functional.pad(A, (0, 3, 0, 3))
    larger_h = torch.nn.functional.pad(h, (0, 3))
    padded_pe = build_laplacian_node_pos(larger_A, 16, h=larger_h)
    assert torch.allclose(pe, padded_pe[:, :A.shape[1]], atol=1e-6)
    return {'latent_shape': list(latent.shape), 'slots': list(outputs.h.shape),
            'empty_loss_finite': True, 'pe_padding_invariant': True}


def overlay(path, sample, target, prediction, guides):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, graph, title in zip(axes, [target, prediction], ['Target chords', 'Decoded chords']):
        image = Path(sample['image'])
        if image.exists():
            ax.imshow(plt.imread(image), cmap='gray', extent=[0, 128, 128, 0])
        for guide in guides:
            xy = guide * 128
            ax.plot(xy[:, 0], xy[:, 1], color='yellow', linewidth=.7)
        for edge in graph['edges']:
            xy = graph['nodes'][edge] * 128
            ax.plot(xy[:, 0], xy[:, 1], color='cyan', linewidth=1)
        ax.scatter(graph['nodes'][:, 0] * 128, graph['nodes'][:, 1] * 128, s=8, color='red')
        ax.set(xlim=(0, 128), ylim=(128, 0), title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(model, dataset, output, prototype):
    output.mkdir(parents=True, exist_ok=True)
    records, empty_records, prototype_records = [], [], []
    shape_scores = []
    empty = {'nodes': np.empty((0, 2)), 'edges': np.empty((0, 2), dtype=int)}
    model.eval()
    matcher_squared, matcher_nodes = 0., 0
    with torch.no_grad():
        for i, sample in enumerate(dataset.samples):
            data = dataset[i]
            batch = custom_collate_fn([data]).to(model.device)
            latent = model.encode(batch.clone())
            decoded = model.decode(latent)
            prediction = decode_graph(decoded[0])
            target = decode_graph(data)
            raw = dataset.load_target(i)
            metric = raw['metric']
            record = score_graph(prediction, target, metric)
            record['sample_id'] = sample['sample_id']
            records.append(record)
            empty_records.append(score_graph(empty, target, metric))
            prototype_records.append(score_graph(prototype, target, metric))
            aligned = model(batch.clone(), hard_matcher=True)
            active = batch.h[0]
            error = (aligned.nodes.features[0, active] - batch.nodes.features[0, active]).cpu().numpy() @ metric.T
            matcher_squared += float((error ** 2).sum())
            matcher_nodes += int(active.sum())
            guides = [raw['full_paths'][a:b] for a, b in zip(raw['path_offsets'][:-1], raw['path_offsets'][1:])]
            shape_scores.append({'sample_id': sample['sample_id'],
                'prediction_vs_target_chords': path_distance(chord_paths(prediction), chord_paths(target), metric),
                'target_chords_vs_guides': path_distance(chord_paths(target), guides, metric)})
            np.savez_compressed(output / f'{sample["sample_id"]}.npz', **prediction,
                existence=decoded.h[0].cpu().numpy(), source=np.array(sample['source']),
                crop_start_xy=np.asarray(sample['crop_start_xy']), metric=metric)
            if i < 6:
                overlay(output / f'{sample["sample_id"]}.png', sample, target, prediction, guides)
    report = {'reconstruction': summarize_groups(records, dataset.samples),
              'all_empty_baseline': summarize_groups(empty_records, dataset.samples),
              'training_prototype_baseline': summarize_groups(prototype_records, dataset.samples),
              'matcher_coordinate_rmse_px': (matcher_squared / matcher_nodes) ** .5 if matcher_nodes else None,
              'per_graph': records, 'shape_scores': shape_scores}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report['reconstruction']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'tiny', 'pilot'], default='smoke')
    parser.add_argument('--config', default='octa_straight')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=Path('logs/geometric'))
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / f'GRALE/configs/{args.config}.yaml').read_text())
    seed = config['seed']
    pl.seed_everything(seed, workers=True)
    torch.set_num_threads(1)
    subset = args.output / 'subsets' / f'{args.mode}-{seed}.json'
    identity = prepare_subset(args.manifest, subset, args.mode, seed)
    output = args.output / args.mode / args.config
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.ckpt').exists():
        raise ValueError('Run already exists; select a new --output rather than overwrite evidence')
    steps = args.steps or {'smoke': 2, 'tiny': 1000, 'pilot': 2000}[args.mode]
    config['n_grad_steps'] = steps
    config['n_warmup_steps'] = 0 if args.mode == 'smoke' else min(50, steps // 10)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    device = 'cpu' if device == 'auto' else device
    if device == 'cpu':
        config['batchsize_effective'] = min(config['batchsize_effective'], 2)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    data = GeometricDataModule(subset, config['n_nodes_max'], config['batchsize_effective'], 0)
    model = GRALE_model(config).to(device)
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    checks = sanity(model, data, device)
    callback = RuntimeCheck()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    trainer = pl.Trainer(accelerator='gpu' if device == 'cuda' else 'cpu', devices=1,
        max_steps=steps, max_time='00:00:30:00', callbacks=[callback],
        logger=CSVLogger(str(output), name='csv'), enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        gradient_clip_val=config['max_grad_norm'], precision='32-true',
        log_every_n_steps=1, num_sanity_val_steps=0)
    trainer.fit(model, datamodule=data)
    trainer.validate(model, datamodule=data, verbose=False)
    assert callback.backward_steps > 0
    changed = sum(not torch.equal(initial[name], value.detach().cpu()) for name, value in model.named_parameters())
    assert changed > 0
    trainer.save_checkpoint(str(output / 'last.ckpt'))
    restored = GRALE_model.load_from_checkpoint(str(output / 'last.ckpt'), map_location=device).to(device)
    restored.eval()
    model.eval()
    batch = next(iter(data.train_dataloader())).to(device)
    with torch.no_grad():
        assert torch.allclose(model.encode(batch.clone()), restored.encode(batch.clone()), atol=1e-6)
    prototype_id = next(i for i, sample in enumerate(data.dataset_train.samples) if sample['nodes'])
    prototype = decode_graph(data.dataset_train[prototype_id])
    reports = {split: evaluate(model, dataset, output / split, prototype)
               for split, dataset in [('train', data.dataset_train), ('valid', data.dataset_valid)]}
    nonempty = reports['train']['nonempty']
    empty = reports['train']['empty']
    tiny_gate = (nonempty.get('count_correct') == nonempty.get('exact') == 1
                 and nonempty.get('coordinate_rmse_px') is not None
                 and nonempty['coordinate_rmse_px'] <= 1 and empty.get('exact') == 1)
    state = {'mode': args.mode, 'config': args.config, 'sanity': checks,
             'steps': trainer.global_step, 'finite_backward_steps': callback.backward_steps,
             'changed_parameter_tensors': changed, 'checkpoint_roundtrip': True,
             'runtime_passed': True,
             'tiny_fit_gate': tiny_gate if args.mode == 'tiny' else None,
             'interpretation': ('tiny fit gate passed' if tiny_gate else
                 'short runtime check; reconstruction quality not established') if args.mode != 'pilot' else
                 'held-out pilot; assess provisional gates in EXPERIMENT_PLAN.md',
             'parameters': sum(p.numel() for p in model.parameters()),
             'seconds': time.time() - callback.started,
             'seconds_per_step': float(np.mean(callback.batch_seconds)),
             'peak_gpu_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
             'device': device, 'python': sys.version, 'torch': torch.__version__, 'lightning': pl.__version__,
             'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'working_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest(),
             'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in
                             [*sorted((ROOT / 'GRALE').rglob('*.py')), Path(__file__).resolve()]},
             'subset': str(subset.resolve()), 'subset_sha256': digest(subset),
             'source_sha256': identity['source_sha256'], 'reports': reports,
             'prototype_sample_id': data.dataset_train.samples[prototype_id]['sample_id']}
    (output / 'result.json').write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: state[key] for key in ['mode', 'config', 'sanity', 'steps',
        'finite_backward_steps', 'checkpoint_roundtrip', 'seconds_per_step', 'device']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
