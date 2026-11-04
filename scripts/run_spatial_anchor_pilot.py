"""Paired learned-L1 experiment: default, centered, spread, and spread warmup.

All arms use the same frozen graphs, shared initial weights and minibatch order.
No fixed correspondence, image input or coordinate term is added to matching.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import yaml

from GRALE.data.dataset import custom_collate_fn
from GRALE.data.geometry import GeometricDataset
from GRALE.loss.geometry import decode_graph, summarize_groups
from GRALE.main import GRALE_model
from scripts.inspect_geometric_run import digest, geometry, plot_graphs, spread, transport_stats

ARMS = {'default': 'default', 'center': 'center',
        'spread': 'jittered_grid', 'spread_warmup': 'jittered_grid'}


class InspectionDataset(GeometricDataset):
    """Recover training inputs from the old diagnostic's saved GT arrays.

    These are the exact float32 nodes and simple adjacency used by that run.
    Original images/curved paths are unavailable and are not used by this pilot.
    The frozen subset itself is neither edited nor resampled.
    """
    def __init__(self, manifest, split, n_nodes_max, inspection):
        super().__init__(manifest, split, n_nodes_max)
        self.inspection = inspection
        self.diagnostic_split = split

    def target_path(self, index):
        return self.inspection / f'{self.diagnostic_split}-{self.samples[index]["sample_id"]}.npz'

    def load_target(self, index):
        with np.load(self.target_path(index), allow_pickle=False) as target:
            return {'nodes': target['target_nodes'], 'edges': target['target_edges'],
                    'metric': target['metric']}


def inspection_datasets(manifest, frozen, config, inspection, reference_run=None):
    """Validate identity/counts/topology and retain source hashes for recovery."""
    report_path = inspection / 'report.json'
    report = json.loads(report_path.read_text())
    if report['subset_sha256'] != digest(manifest):
        raise ValueError('Inspection belongs to a different frozen subset')
    if reference_run and Path(report['run_dir']).resolve() != reference_run.resolve():
        raise ValueError('Inspection belongs to a different reference run')
    if report.get('saved_prediction_mismatches') != []:
        raise ValueError('Inspection did not reproduce saved reference predictions')
    records = {(r['split'], r['sample_id']): r for r in report['graphs']}
    datasets = {split: InspectionDataset(manifest, split, config['n_nodes_max'], inspection)
                for split in ['train', 'valid']}
    expected = {(split, row['sample_id']) for split, dataset in datasets.items() for row in dataset.samples}
    if set(records) != expected or len(report['graphs']) != len(expected):
        raise ValueError('Inspection and frozen cohort sample identities differ')
    sources = {}
    for split, dataset in datasets.items():
        for index, sample in enumerate(dataset.samples):
            target = dataset.load_target(index)
            nodes, edges, metric = (target[key] for key in ['nodes', 'edges', 'metric'])
            record = records[split, sample['sample_id']]
            if (nodes.dtype != np.float32 or nodes.shape != (sample['nodes'], 2)
                    or edges.shape != (sample['edges'], 2)
                    or not np.issubdtype(edges.dtype, np.integer)
                    or not np.isfinite(nodes).all()
                    or not np.array_equal(metric, np.eye(2) * 128)
                    or record['n_true'] != len(nodes) or record['target_edges'] != len(edges)):
                raise ValueError(f'Invalid recovered target: {sample["sample_id"]}')
            if edges.size and (edges.min() < 0 or edges.max() >= len(nodes)
                               or np.any(edges[:, 0] >= edges[:, 1])
                               or len(np.unique(edges, axis=0)) != len(edges)):
                raise ValueError('Recovered edges must be unique, simple and undirected')
            measured = spread(nodes @ metric.T)
            saved = record['target_spread_px']
            if (measured is None) != (saved is None) or (measured is not None and
                    not np.isclose(measured, saved, atol=1e-6, rtol=1e-7)):
                raise ValueError('Recovered geometry does not match the saved inspection')
            # Exercise the inherited graph builder before creating any run output.
            dataset[index]
            sources[str(dataset.target_path(index))] = digest(dataset.target_path(index))
    return datasets, {'mode': 'saved_inspection_gt_arrays', 'report': str(report_path),
                      'report_sha256': digest(report_path), 'target_artifacts_sha256': sources,
                      'original_archives_sha256_unverified': frozen['archives_sha256'],
                      'unavailable_unused_fields': ['images', 'full_paths', 'path_offsets'],
                      'note': 'Recovered original float32 training nodes, simple edges and metric; no resampling'}


def initialize(config, arm, device):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    cfg = dict(config, coordinate_init=ARMS[arm], coordinate_anchor_seed=config['seed'])
    model = GRALE_model(cfg).to(device)
    final = model.decoder.head_node_features.mlp
    final = final if isinstance(final, torch.nn.Linear) else final[-1]
    excluded = {id(p) for p in final.parameters()}
    common = hashlib.sha256()
    all_parameters = hashlib.sha256()
    for name, parameter in model.named_parameters():
        data = name.encode() + parameter.detach().cpu().numpy().tobytes()
        all_parameters.update(data)
        if id(parameter) not in excluded:
            common.update(data)
    return model, cfg, common.hexdigest(), all_parameters.hexdigest()


def forward(model, batch):
    inputs = model.input_builder(batch.clone())
    encoded, latent = model.encoder(inputs)
    embeddings, output = model.decoder(latent)
    T, _ = model.matcher(encoded, ~inputs.h, embeddings)
    loss, terms = model.training_objective(output, model.target_builder(batch.clone()), T)
    return output, T, loss.mean(), terms


def residual_scale(arm, step, hold, ramp):
    return min(1., max(0., (step - hold) / max(ramp, 1))) if arm == 'spread_warmup' else 1.


def set_scale(model, value):
    if model.decoder.coordinate_residual_scale is not None:
        model.decoder.coordinate_residual_scale.fill_(value)


def evaluate(model, datasets, arm, step, output, forward_fn=None):
    predict = forward if forward_fn is None else forward_fn
    model.eval()
    folder = output / 'snapshots' / f'step-{step:06d}'
    folder.mkdir(parents=True)
    records, groups = [], {}
    train = datasets['train']
    selected = {('train', next(i for i, row in enumerate(train.samples) if row['nodes'])),
                ('train', max(range(len(train)), key=lambda i: train.samples[i]['nodes'])),
                ('valid', next(i for i, row in enumerate(datasets['valid'].samples) if row['nodes']))}
    with torch.no_grad():
        for split, dataset in datasets.items():
            rows = []
            for index, sample in enumerate(dataset.samples):
                graph = dataset[index]
                batch = custom_collate_fn([graph]).to(model.device)
                output_graph, T, _, terms = predict(model, batch)
                prediction = decode_graph(model.format_logits(output_graph.clone())[0])
                target = decode_graph(graph)
                metric = np.asarray(dataset.load_target(index)['metric'])
                xy = output_graph.nodes.features[0].cpu().numpy()
                real = T[0].cpu().numpy()[:, graph.h.numpy()]
                matching, bary, _, _ = transport_stats(real, xy, target['nodes'], prediction['slots'], metric)
                row = {'split': split, 'sample_id': sample['sample_id'], 'loss_terms': terms,
                       **geometry(prediction, target, metric), **matching,
                       'all_slot_spread_px': spread(xy @ metric.T)}
                records.append(row)
                rows.append(row)
                np.savez_compressed(folder / f'{split}-{sample["sample_id"]}.npz',
                                    **prediction, all_slot_coordinates=xy, transport=T[0].cpu().numpy(), metric=metric)
                if (split, index) in selected:
                    plot_graphs(folder / f'{split}-{sample["sample_id"]}.png', target, prediction,
                                bary, real, metric, row)
            aggregate = summarize_groups(rows, dataset.samples)
            nonempty = [r for r in rows if r['n_true']]
            aggregate['diagnostics'] = {
                key: float(np.mean([r[key] for r in nonempty if r.get(key) is not None]))
                for key in ['mean_max_assignment_weight', 'effective_output_slots_per_target',
                            'barycenter_residual_rmse_px_mass_weighted', 'all_slot_spread_px']}
            ratios = [r['spread_ratio'] for r in nonempty if r['spread_ratio'] is not None]
            aggregate['diagnostics']['median_spread_ratio'] = float(np.median(ratios)) if ratios else None
            groups[split] = aggregate
    fit = groups['train']
    gate = (fit['nonempty'].get('exact') == 1 and fit['empty'].get('exact') == 1
            and fit['nonempty'].get('coordinate_rmse_px') is not None
            and fit['nonempty']['coordinate_rmse_px'] <= 1)
    result = {'arm': arm, 'step': step, 'tiny_fit_gate': gate, 'groups': groups, 'per_graph': records}
    (folder / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'arm': arm, 'step': step, 'tiny_fit_gate': gate,
                      'train': fit['nonempty'], 'diagnostics': fit['diagnostics']}), flush=True)
    model.train()
    return result


def plot_comparison(results, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for arm, state in results.items():
        records = state['snapshots']
        for ax, source, key in [(axes[0, 0], 'nonempty', 'coordinate_rmse_px'),
                                (axes[0, 1], 'nonempty', 'node_f1'),
                                (axes[1, 0], 'diagnostics', 'mean_max_assignment_weight'),
                                (axes[1, 1], 'diagnostics', 'median_spread_ratio')]:
            points = [(r['step'], r['groups']['train'][source].get(key)) for r in records]
            points = [(x, y) for x, y in points if y is not None]
            if points:
                x, y = np.array(points).T
                ax.plot(x, y, '.-', label=arm)
    for ax, title in zip(axes.flat, ['Training localization RMSE (pixels)', 'Training node F1',
                                    'Real-target assignment mean maximum', 'Median predicted / GT spatial spread']):
        ax.set(title=title, xlabel='Optimizer update')
        if ax.lines:
            ax.legend(fontsize=8)
    axes[1, 1].axhline(1, color='gray', linestyle='--', linewidth=.7)
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--reference-run', type=Path, help='Completed run containing result.json and config.yaml')
    source.add_argument('--manifest', type=Path, help='Frozen manifest for a local runtime check')
    parser.add_argument('--inspection-targets', type=Path,
                        help='Explicit recovery from this reference inspection when its source archive is unavailable')
    parser.add_argument('--config', type=Path, default=ROOT / 'GRALE/configs/octa_straight_l1_200k.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--hold-steps', type=int, default=500)
    parser.add_argument('--ramp-steps', type=int, default=500)
    args = parser.parse_args()
    if args.steps < 1 or args.hold_steps < 0 or args.ramp_steps < 1:
        parser.error('Steps/ramp must be positive and hold steps nonnegative')
    torch.set_num_threads(1)
    reference = None
    if args.reference_run:
        reference = json.loads((args.reference_run / 'result.json').read_text())
        manifest = Path(reference['subset'])
        if digest(manifest) != reference['subset_sha256']:
            raise ValueError('Reference subset identity changed')
        config = yaml.safe_load((args.reference_run / 'config.yaml').read_text())
    else:
        manifest = args.manifest
        config = yaml.safe_load(args.config.read_text())
    frozen = json.loads(manifest.read_text())
    config.update(n_grad_steps=args.steps, n_warmup_steps=min(50, args.steps // 10),
                  matcher_model='sinkhorn', matcher_cost='l1', batchsize_effective=4)
    if config['node_pos_dim'] != 0 or config['node_features_dim'] != 2:
        raise ValueError('This paired pilot requires the coordinate-only XY configuration')
    if args.inspection_targets:
        datasets, target_provenance = inspection_datasets(
            manifest, frozen, config, args.inspection_targets, args.reference_run)
    else:
        for archive, expected in frozen['archives_sha256'].items():
            if digest(Path(frozen['data_root']) / archive) != expected:
                raise ValueError(f'Target archive identity changed: {archive}')
        datasets = {split: GeometricDataset(manifest, split, config['n_nodes_max']) for split in ['train', 'valid']}
        target_provenance = {'mode': 'original_verified_archives', 'archives_sha256': frozen['archives_sha256']}
    if len(datasets['train']) != 10 or len(datasets['valid']) != 4:
        raise ValueError('Use the existing frozen 10-train/4-validation cohort')
    if {r['subject_id'] for r in datasets['train'].samples} & {r['subject_id'] for r in datasets['valid'].samples}:
        raise ValueError('Source identities overlap')
    if any(r['nodes'] > 20 for dataset in datasets.values() for r in dataset.samples):
        raise ValueError('Frozen cohort contains a graph exceeding 20 nodes')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {'purpose': 'Paired initialization experiment; one seed, tiny cohort; not a generalization study',
             'reference_run': str(args.reference_run) if args.reference_run else None,
             'subset': str(manifest), 'subset_sha256': digest(manifest), 'config': config,
             'target_provenance': target_provenance,
             'hold_steps': args.hold_steps, 'ramp_steps': args.ramp_steps,
             'torch': torch.__version__, 'device': args.device,
             'source_commit': (ROOT / 'SOURCE_COMMIT').read_text().strip() if (ROOT / 'SOURCE_COMMIT').exists() else None,
             'source_archive_sha256': (ROOT / 'SOURCE_ARCHIVE_SHA256').read_text().strip() if (ROOT / 'SOURCE_ARCHIVE_SHA256').exists() else None,
             'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                             [*sorted((ROOT / 'GRALE').rglob('*.py')), Path(__file__).resolve()]},
             'arms': {}}
    schedule = {0, 50, 200, 500, 1000, 2000, 3000, args.steps}
    schedule = {step for step in schedule if step <= args.steps}
    common_hash = None
    order_hash = None
    for arm in ARMS:
        arm_dir = output / arm
        arm_dir.mkdir()
        model, arm_config, shared_sha, full_sha = initialize(config, arm, args.device)
        if common_hash is not None and shared_sha != common_hash:
            raise AssertionError('Shared initialization differs across arms')
        common_hash = shared_sha
        optimizers, schedulers = model.configure_optimizers()
        optimizer, scheduler = optimizers[0], schedulers[0]['scheduler']
        initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        set_scale(model, residual_scale(arm, 0, args.hold_steps, args.ramp_steps))
        snapshots = [evaluate(model, datasets, arm, 0, arm_dir)]
        dataset = datasets['train']
        loader = DataLoader(list(range(len(dataset))), batch_size=4, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(config['seed']),
                            collate_fn=lambda ids: (ids, custom_collate_fn([dataset[i] for i in ids])))
        iterator = iter(loader)
        order = hashlib.sha256()
        started = time.time()
        with (arm_dir / 'losses.csv').open('w', newline='') as stream:
            writer = None
            for step in range(1, args.steps + 1):
                try:
                    ids, batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    ids, batch = next(iterator)
                order.update((json.dumps(ids) + '\n').encode())
                scale = residual_scale(arm, step - 1, args.hold_steps, args.ramp_steps)
                set_scale(model, scale)
                optimizer.zero_grad(set_to_none=True)
                _, _, loss, terms = forward(model, batch.to(args.device))
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss in {arm}, step {step}')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'], error_if_nonfinite=True)
                lr = optimizer.param_groups[0]['lr']
                optimizer.step()
                scheduler.step()
                row = {'step': step, 'batch_size': len(ids), 'learning_rate': lr,
                       'residual_scale': scale, 'gradient_norm': float(norm), **terms}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if step in schedule:
                    stream.flush()
                    snapshots.append(evaluate(model, datasets, arm, step, arm_dir))
                    torch.save({'state_dict': model.state_dict(), 'hyper_parameters': arm_config,
                                'pytorch-lightning_version': pl.__version__,
                                'hparams_name': 'config',
                                'global_step': step, 'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict()}, arm_dir / 'last.ckpt')
                    (arm_dir / 'snapshots.json').write_text(json.dumps(snapshots, indent=2, allow_nan=False) + '\n')
        if order_hash is not None and order.hexdigest() != order_hash:
            raise AssertionError('Training minibatch order differs across arms')
        order_hash = order.hexdigest()
        changed = sum(not torch.equal(initial[name], p.detach().cpu()) for name, p in model.named_parameters())
        if not changed:
            raise AssertionError('No model parameters changed')
        state['arms'][arm] = {'shared_parameter_sha256': shared_sha, 'initial_parameter_sha256': full_sha,
                              'minibatch_order_sha256': order_hash, 'steps': args.steps,
                              'effective_weight_decay': optimizer.param_groups[0]['weight_decay'],
                              'changed_parameter_tensors': changed, 'seconds': time.time() - started,
                              'snapshots': snapshots}
        (output / 'result.json').write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        plot_comparison(state['arms'], output)
        del optimizer, scheduler, model, initial
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    print('SPATIAL_ANCHOR_PILOT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
