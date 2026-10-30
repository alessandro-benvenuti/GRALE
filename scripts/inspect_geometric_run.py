"""Inspect geometry, learned matching, barycentric averaging and logged losses on CPU.

No optimizer is created. Checkpoints, datasets and original run artifacts are read-only.
"""
import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from GRALE.data.dataset import custom_collate_fn
from GRALE.data.geometry import GeometricDataset
from GRALE.loss.geometry import decode_graph, score_graph, summarize_groups
from GRALE.loss.utils import pairwise_L1_norm, pairwise_squared_norm
from GRALE.main import GRALE_model


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def spread(points):
    return float(np.sqrt(np.square(points - points.mean(0)).sum(1).mean())) if len(points) else None


def geometry(prediction, target, metric):
    pred = prediction['nodes'] @ metric.T
    true = target['nodes'] @ metric.T
    ps, ts = spread(pred), spread(true)
    score = score_graph(prediction, target, metric)
    return {**score, 'predicted_edges': len(prediction['edges']), 'target_edges': len(target['edges']),
            'predicted_spread_px': ps, 'target_spread_px': ts,
            'spread_ratio': ps / ts if ps is not None and ts else None,
            'centroid_error_px': float(np.linalg.norm(pred.mean(0) - true.mean(0))) if len(pred) and len(true) else None,
            'coordinate_rmse_px': math.sqrt(score['squared_error'] / score['assigned']) if score['assigned'] else None}


def transport_stats(T, prediction_xy, target_xy, selected, metric):
    """Rows are output slots, columns are real targets (padding columns excluded)."""
    n, N = len(target_xy), len(prediction_xy)
    if not n:
        return {}, np.zeros((N, 2)), np.zeros((N, 0)), np.zeros(N)
    real = np.asarray(T, dtype=np.float64)
    mass = real.sum(1)
    q = np.divide(real, mass[:, None], out=np.zeros_like(real), where=mass[:, None] > 0)
    barycenters = q @ target_xy
    columns = real / real.sum(0, keepdims=True)
    column_entropy = -(columns * np.log(np.clip(columns, 1e-300, None))).sum(0)
    row_entropy = -(q * np.log(np.clip(q, 1e-300, None))).sum(1)
    residual_squared = np.square((prediction_xy - barycenters) @ metric.T).sum(1)
    bary_rmse = math.sqrt(float(residual_squared @ mass / mass.sum()))
    # Exact squared-loss decomposition at fixed T: residual to each barycenter
    # plus target variance within that row's conditional distribution.
    pair_squared = np.square((prediction_xy[:, None] - target_xy[None]) @ metric.T).sum(2)
    variance = (q * np.square((target_xy[None] - barycenters[:, None]) @ metric.T).sum(2)).sum(1)
    total = float((real * pair_squared).sum() / mass.sum())
    residual = float(residual_squared @ mass / mass.sum())
    mixed_variance = float(variance @ mass / mass.sum())
    stats = {
        'mean_max_assignment_weight': float(columns.max(0).mean()),
        'target_assignment_entropy': float(column_entropy.mean()),
        'target_assignment_entropy_normalized_log_capacity': float(column_entropy.mean() / math.log(N)) if N > 1 else 0.,
        'effective_output_slots_per_target': float(np.exp(column_entropy).mean()),
        'effective_real_targets_per_output_mass_weighted': float(np.exp(row_entropy) @ mass / mass.sum()),
        'row_max_target_weight_mass_weighted': float(q.max(1) @ mass / mass.sum()),
        'real_target_mass_on_decoded_slots_fraction': float(mass[selected].sum() / mass.sum()),
        'real_mass_output_rows_gt_half': int((mass > .5).sum()),
        'uniform_reference_all_slots': 1 / N,
        'uniform_reference_n_real_targets': 1 / n,
        'barycenter_residual_rmse_px_mass_weighted': bary_rmse,
        'barycenter_residual_rmse_px_decoded': float(np.sqrt(residual_squared[selected].mean())) if len(selected) else None,
        'transport_pair_coordinate_mse_px2': total,
        'barycenter_residual_mse_px2': residual,
        'within_row_target_variance_px2': mixed_variance,
        'coordinate_decomposition_absolute_error': abs(total - residual - mixed_variance),
    }
    return stats, barycenters, q, mass


def plot_graphs(path, target, prediction, barycenters, transport, metric, record):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    point_sets = [target['nodes'] @ metric.T, prediction['nodes'] @ metric.T]
    selected_bary = (barycenters[prediction['slots']] @ metric.T
                     if len(target['nodes']) else np.empty((0, 2)))
    points = np.concatenate([*point_sets, selected_bary])
    low = np.minimum(points.min(0), 0) - 3 if len(points) else np.full(2, -3)
    high = np.maximum(points.max(0), 128) + 3 if len(points) else np.full(2, 131)
    for ax, graph, color, title in zip(axes[:2], [target, prediction], ['tab:blue', 'tab:orange'],
                                        ['Ground truth', 'Decoded prediction']):
        xy = graph['nodes'] @ metric.T
        for edge in graph['edges']:
            ax.plot(xy[edge, 0], xy[edge, 1], color=color, lw=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=20, color=color)
        ax.set_title(f'{title}: {len(xy)} nodes, {len(graph["edges"])} edges')
    pred = point_sets[1]
    axes[2].scatter(point_sets[0][:, 0], point_sets[0][:, 1], color='lightgray', s=14, label='GT nodes')
    axes[2].scatter(pred[:, 0], pred[:, 1], color='tab:orange', s=20, label='Predicted nodes')
    axes[2].scatter(selected_bary[:, 0], selected_bary[:, 1], color='tab:blue', marker='x', s=30,
                    label='Transport target averages')
    for a, b in zip(pred, selected_bary):
        axes[2].plot([a[0], b[0]], [a[1], b[1]], color='gray', lw=.6)
    axes[2].set_title('Prediction vs transport target averages')
    axes[2].legend(fontsize=7)
    for ax in axes[:3]:
        ax.set(xlim=(low[0], high[0]), ylim=(high[1], low[1]), aspect='equal', xlabel='X (pixels)', ylabel='Y (pixels)')
    if transport.shape[1]:
        order = np.argsort(-transport.sum(1), kind='stable')
        im = axes[3].imshow(transport[order], aspect='auto', interpolation='nearest', vmin=0,
                            vmax=max(float(transport.max()), 1e-12))
        fig.colorbar(im, ax=axes[3], fraction=.05)
        axes[3].set(xlabel='Real GT node column', ylabel='Output slots sorted by real-target mass',
                    title=f'Transport: mean max = {record["mean_max_assignment_weight"]:.3g}')
    else:
        axes[3].text(.5, .5, 'Empty target: no real-node matching', ha='center', va='center', wrap=True)
        axes[3].axis('off')
    fig.suptitle(f'{record["split"]}: {record["sample_id"]}', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def inspect_graph(model, dataset, index, split, run_dir, output):
    sample = dataset.samples[index]
    graph = dataset[index]
    raw = dataset.load_target(index)
    metric = np.asarray(raw['metric'], dtype=np.float64)
    batch = custom_collate_fn([graph])
    with torch.no_grad():
        inputs = model.input_builder(batch.clone())
        encoded, latent = model.encoder(inputs)
        embeddings, logits = model.decoder(latent)
        T, _ = model.matcher(encoded, ~inputs.h, embeddings)
        _, loss_terms = model.training_objective(logits, model.target_builder(batch.clone()), T)
        decoded = model.format_logits(logits.clone())
        prediction, target = decode_graph(decoded[0]), decode_graph(graph)
        # Reconstruct the actual learned cost for the L1 tie diagnostic.
        a = model.matcher.linear_outputs(embeddings + model.matcher.positionnal_encoding_outputs)
        padded = torch.where(inputs.h[..., None], encoded, model.matcher.node_padding_features)
        b = model.matcher.linear_inputs(padded)
        costs = (pairwise_L1_norm(a, b) if model.matcher.matcher_cost == 'l1'
                 else pairwise_squared_norm(a, b).clamp_min(0))[0]
    full_T = T[0].numpy()
    if not np.isfinite(full_T).all():
        raise ValueError(f'Nonfinite transport: {sample["sample_id"]}')
    real = full_T[:, graph.h.numpy()]
    stats, barycenters, q, mass = transport_stats(
        real, logits.nodes.features[0].numpy(), target['nodes'], prediction['slots'], metric)
    record = {'split': split, 'sample_id': sample['sample_id'],
              **geometry(prediction, target, metric), **stats, 'loss_terms': loss_terms,
              'marginal_row_max_error': float(np.abs(full_T.sum(1) - 1).max()),
              'marginal_column_max_error': float(np.abs(full_T.sum(0) - 1).max())}
    rows = np.flatnonzero(mass > .5)
    if len(rows) > 1 and len(target['nodes']) > 1:
        block = costs.numpy()[rows][:, graph.h.numpy()]
        centered = block - block.mean(0, keepdims=True) - block.mean(1, keepdims=True) + block.mean()
        signs = (a[0, rows, None] - b[0, None, graph.h]).sign().numpy()
        record.update(raw_cost_doublecenter_max_abs=float(np.abs(centered).max()),
                      constant_sign_matching_dimensions=int((signs.max((0, 1)) == signs.min((0, 1))).sum()),
                      matching_dimensions=a.shape[-1])
    saved = run_dir / split / f'{sample["sample_id"]}.npz'
    if saved.exists():
        with np.load(saved, allow_pickle=False) as archive:
            record['reproduces_saved_prediction'] = bool(
                np.array_equal(prediction['slots'], archive['slots'])
                and np.array_equal(prediction['edges'], archive['edges'])
                and prediction['nodes'].shape == archive['nodes'].shape
                and np.allclose(prediction['nodes'], archive['nodes'], atol=1e-5, rtol=1e-5))
    else:
        record['reproduces_saved_prediction'] = None
    stem = f'{split}-{sample["sample_id"]}'
    np.savez_compressed(output / f'{stem}.npz', transport=full_T, real_target_transport=real,
                        raw_matching_cost=costs.numpy(), projected_output_embeddings=a[0].numpy(),
                        projected_input_embeddings=b[0].numpy(),
                        row_conditionals=q, real_target_mass=mass, target_nodes=target['nodes'],
                        target_edges=target['edges'], predicted_nodes=prediction['nodes'],
                        predicted_edges=prediction['edges'], decoded_slots=prediction['slots'],
                        all_predicted_nodes=logits.nodes.features[0].numpy(),
                        target_averages=barycenters, metric=metric)
    plot_graphs(output / f'{stem}.png', target, prediction, barycenters, real, metric, record)
    return record, target, metric


def inspect_losses(run_dir, output, config, window, bin_width, csv_override=None):
    candidates = [csv_override] if csv_override else sorted((run_dir / 'csv').glob('version_*/metrics.csv'))
    training = []
    for path in candidates:
        with path.open(newline='') as stream:
            fields = next(csv.reader(stream))
        if any(name.startswith('train/loss') for name in fields):
            training.append((path, fields))
    if len(training) != 1:
        raise ValueError('Expected one training metrics.csv; select it explicitly with --metrics-csv')
    path, fields = training[0]
    terms = ['loss', 'loss_h', 'loss_node_features', 'loss_adjacency', 'loss_edge_labels',
             'loss_edge_features', 'loss_marginals']
    columns = {}
    for term in terms:
        for candidate in [f'train/{term}_step', f'train/{term}', f'train/{term}_epoch']:
            if candidate in fields:
                columns[f'train/{term}'] = candidate
                break
        if term in fields:
            columns[f'valid/{term}'] = term
    if 'learning_rate' in fields:
        columns['learning_rate'] = 'learning_rate'
    first, last = defaultdict(list), defaultdict(lambda: deque(maxlen=2 * window))
    counts, bins = defaultdict(int), defaultdict(lambda: defaultdict(lambda: [0., 0]))
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if not row.get('step'):
                continue
            step = int(float(row['step']))
            for name, column in columns.items():
                if not row.get(column):
                    continue
                value = float(row[column])
                if not math.isfinite(value):
                    raise ValueError(f'Nonfinite logged value: {column}, step {step}')
                if len(first[name]) < window:
                    first[name].append(value)
                last[name].append(value)
                counts[name] += 1
                slot = bins[step // bin_width][name]
                slot[0] += value
                slot[1] += 1
    summaries = {}
    for name, values in last.items():
        values = list(values)
        tail, previous = values[-window:], values[:-window]
        factor = config.get('alpha_' + name.split('/')[-1].removeprefix('loss_'), 1.) if name.startswith('train/') else None
        summaries[name] = {'column': columns[name], 'logged_values': counts[name],
                           'first_window_mean': float(np.mean(first[name])),
                           'last_window_mean': float(np.mean(tail)),
                           'previous_window_mean': float(np.mean(previous)) if previous else None,
                           'last_vs_previous_change': float(np.mean(tail) - np.mean(previous)) if previous else None,
                           'training_coefficient': factor}
    rows = [{'step_bin_start': index * bin_width,
             **{name: total / count for name, (total, count) in values.items()}}
            for index, values in sorted(bins.items())]
    with (output / 'loss-curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step_bin_start', *columns])
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {term: f'C{index}' for index, term in enumerate(terms)}
    for name in columns:
        points = [(row['step_bin_start'], row[name]) for row in rows if name in row]
        if not points:
            continue
        x, y = np.array(points).T
        if name == 'learning_rate':
            axes[1, 1].plot(x, y)
        elif name.startswith('train/'):
            term = name.removeprefix('train/')
            axes[0, 0].plot(x, y, '.-', color=colors[term], label=term)
            if name != 'train/loss':
                axes[0, 1].plot(x, y * summaries[name]['training_coefficient'], '.-', color=colors[term], label=term)
        else:
            term = name.removeprefix('valid/')
            axes[1, 0].plot(x, y, '.-', color=colors[term], label=term)
    for ax, title in zip(axes.flat, ['Training losses before coefficients', 'Weighted training contributions',
                                    'Validation losses before coefficients', 'Learning rate']):
        ax.set(title=title, xlabel='Logged optimizer step')
        if title != 'Learning rate':
            ax.set_yscale('symlog', linthresh=1e-6)
            if ax.lines:
                ax.legend(fontsize=7)
    fig.suptitle(f'Means within {bin_width}-step bins; curves are diagnostics, not aggregate graph metrics')
    fig.tight_layout()
    fig.savefig(output / 'loss-curves.png', dpi=140)
    plt.close(fig)
    return {'csv': str(path), 'csv_sha256': digest(path), 'window_logged_values': window,
            'bin_width_steps': bin_width, 'series': summaries}


def snapshot_history(run_dir, targets, output):
    rows = []
    for folder in sorted((run_dir / 'intermediate').glob('step-*')):
        step = int(folder.name.removeprefix('step-'))
        for key, (target, metric) in targets.items():
            path = folder / f'{key}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    prediction = {'nodes': saved['nodes'], 'edges': saved['edges']}
                rows.append({'example': key, 'step': step, **geometry(prediction, target, metric)})
    if not rows:
        return {'examples': 0, 'records': 0}
    with (output / 'geometry-over-steps.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in sorted({row['example'] for row in rows}):
        for ax, field in zip(axes, ['spread_ratio', 'centroid_error_px']):
            points = [(row['step'], row[field]) for row in rows if row['example'] == key and row[field] is not None]
            if points:
                x, y = np.array(points).T
                ax.plot(x, y, '.-', label=key)
    axes[0].axhline(1, color='gray', ls='--', lw=.7)
    axes[0].set(title='Predicted / GT spatial spread', xlabel='Optimizer step')
    axes[1].set(title='Centroid error (pixels)', xlabel='Optimizer step')
    for ax in axes:
        if ax.lines:
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / 'geometry-over-steps.png', dpi=140)
    plt.close(fig)
    return {'examples': len({row['example'] for row in rows}), 'records': len(rows)}


def summary_text(report):
    def number(value):
        return '-' if value is None else f'{value:.4g}'
    lines = [f'CPU inspection: step {report["step"]}, matcher {report["matcher_cost"]}',
             'No training or checkpoint changes. All frozen train/validation graphs inspected.',
             'Spread is RMS distance from each graph\'s own centroid, in pixels.',
             'Target max T is averaged over real GT columns, over all output slots.',
             'Barycenter residual is weighted by transport mass from real targets.', '']
    for split in ['train', 'valid']:
        lines += [split.upper(), 'sample | nodes GT/pred | edges GT/pred | spread GT/pred | centroid error | localization RMSE | max T | effective outputs/target | barycenter residual']
        for row in report['graphs']:
            if row['split'] != split:
                continue
            values = [row['sample_id'], f'{row["n_true"]}/{row["n_pred"]}',
                      f'{row["target_edges"]}/{row["predicted_edges"]}',
                      f'{number(row["target_spread_px"])}/{number(row["predicted_spread_px"])}',
                      *[number(row.get(key)) for key in ['centroid_error_px', 'coordinate_rmse_px',
                          'mean_max_assignment_weight', 'effective_output_slots_per_target',
                          'barycenter_residual_rmse_px_mass_weighted']]]
            lines.append(' | '.join(values))
        lines.append('')
    lines += ['LOSS CURVES: first / last window means, and last minus preceding window',
              'Step-log means use equal weight per optimizer update; batches have differing sizes.']
    for name, value in report['loss_curves']['series'].items():
        if name.startswith('train/'):
            lines.append(f'{name}: {number(value["first_window_mean"])} / {number(value["last_window_mean"])}; '
                         f'late change {number(value["last_vs_previous_change"])}; coefficient {number(value["training_coefficient"])}')
    lines += ['', 'Geometry alone cannot prove collapse. Diffuse T plus a small barycenter residual',
              'supports transport averaging; a large spatial spread can still conceal group-average collapse.',
              'CPU vs saved prediction mismatches: ' + str(report['saved_prediction_mismatches'])]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Directory containing result.json and last.ckpt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--metrics-csv', type=Path)
    parser.add_argument('--window', type=int, default=300)
    parser.add_argument('--bin-width', type=int, default=1000)
    args = parser.parse_args()
    if args.window < 1 or args.bin_width < 1:
        parser.error('--window and --bin-width must be positive')
    torch.set_num_threads(1)
    run_dir = args.run_dir.resolve()
    original = json.loads((run_dir / 'result.json').read_text())
    subset = Path(original['subset'])
    if not subset.exists():
        subset = run_dir.parent.parent / 'subsets' / subset.name
    if digest(subset) != original['subset_sha256']:
        raise ValueError('Frozen subset hash differs from the training result')
    frozen = json.loads(subset.read_text())
    for name, expected in frozen['archives_sha256'].items():
        if digest(Path(frozen['data_root']) / name) != expected:
            raise ValueError(f'Target archive changed: {name}')
    for name, expected in original.get('code_sha256', {}).items():
        if name.startswith('GRALE/') and digest(ROOT / name) != expected:
            raise ValueError(f'Model source differs from the training run: {name}')
    checkpoint = run_dir / 'last.ckpt'
    if digest(checkpoint) != original['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs from the original result')
    output = (args.output or run_dir / 'inspection').resolve()
    bundle = output.parent / (output.name + '.zip')
    if bundle.exists():
        raise FileExistsError(f'Choose a fresh output; bundle already exists: {bundle}')
    output.mkdir(parents=True, exist_ok=False)
    model = GRALE_model.load_from_checkpoint(str(checkpoint), map_location='cpu').eval()
    model.requires_grad_(False)
    records, targets, aggregates = [], {}, {}
    for split in ['train', 'valid']:
        data = GeometricDataset(subset, split, model.hparams['n_nodes_max'])
        group = []
        for index, sample in enumerate(data.samples):
            record, target, metric = inspect_graph(model, data, index, split, run_dir, output)
            records.append(record)
            group.append(record)
            targets[f'{split}-{sample["sample_id"]}'] = (target, metric)
            print(f'Inspected {split} {index + 1}/{len(data)}: {sample["sample_id"]}', flush=True)
        aggregates[split] = summarize_groups(group, data.samples)
    report = {'run_dir': str(run_dir), 'step': original['steps'], 'mode': 'CPU inference only',
              'checkpoint_sha256': digest(checkpoint), 'subset_sha256': digest(subset),
              'diagnostic_script_sha256': digest(Path(__file__)), 'torch': torch.__version__,
              'model_hyperparameters': dict(model.hparams),
              'matcher_cost': model.matcher.matcher_cost, 'graphs': records, 'reconstruction': aggregates,
              'saved_prediction_mismatches': [f'{row["split"]}/{row["sample_id"]}' for row in records
                                               if row['reproduces_saved_prediction'] is False],
              'loss_curves': inspect_losses(run_dir, output, dict(model.hparams), args.window,
                                            args.bin_width, args.metrics_csv),
              'snapshot_history': snapshot_history(run_dir, targets, output)}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    summary = summary_text(report)
    (output / 'summary.txt').write_text(summary)
    with ZipFile(bundle, 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, arcname=path.name)
    print(summary, flush=True)
    print(f'Report: {output / "report.json"}\nResults bundle: {bundle}', flush=True)


if __name__ == '__main__':
    main()
